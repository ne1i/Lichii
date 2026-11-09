// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon {

/// Guest memory a cache uploaded from, which the guest rarely reads again: every 600 frames it is
/// marked cold (DeviceMemoryManager::HintCold, madvise(MADV_COLD) on Linux), so the OS reclaims it
/// before anything else. The first tick after a load hinted everything the load uploaded at once,
/// on the GPU thread (frame 1802 after it: 66-75 ms in two OoT runs).
///
/// EDEN_NX2_FIX_COLD_HINT_BUDGET (default on; =0 hints every range at the tick as before): the
/// ranges collected by a tick are hinted over the following frames, at most
/// EDEN_NX2_COLD_HINT_MB_PER_FRAME (default 16) MiB a frame. EDEN_NX2_COLD_HINT=0 hints nothing.
class ColdHintQueue {
public:
    void Add(DAddr address, u64 size) {
        collected.emplace_back(address, size);
    }

    /// Once a frame, before frame_tick advances. `cache` names the caller in the log.
    template <typename DeviceMemory>
    void Tick(u64 frame_tick, DeviceMemory& device_memory, const char* cache) {
        const Config& config = GetConfig();
        if (frame_tick % 600 == 0 && !collected.empty()) {
            if (!config.enabled) {
                collected.clear();
            } else if (config.budget == 0) {
                const auto begin = std::chrono::steady_clock::now();
                for (const auto& [address, size] : collected) {
                    device_memory.HintCold(address, size);
                    hinted_bytes += size;
                }
                hinted_time += std::chrono::steady_clock::now() - begin;
                hinted_ranges += collected.size();
                collected.clear();
                Report(cache);
            } else {
                pending.insert(pending.end(), collected.begin(), collected.end());
                collected.clear();
            }
        }
        if (pending.empty()) {
            return;
        }
        const auto begin = std::chrono::steady_clock::now();
        u64 left = config.budget;
        while (!pending.empty() && left != 0) {
            auto& [address, size] = pending.front();
            const u64 step = (std::min)(size, left);
            device_memory.HintCold(address, step);
            hinted_bytes += step;
            left -= step;
            address += step;
            size -= step;
            if (size == 0) {
                pending.pop_front();
                ++hinted_ranges;
            }
        }
        hinted_time += std::chrono::steady_clock::now() - begin;
        ++hinted_frames;
        if (pending.empty()) {
            Report(cache);
        }
    }

private:
    struct Config {
        bool enabled;
        u64 budget; ///< Bytes a frame, 0 for all at the tick
    };

    static const Config& GetConfig() {
        static const Config config = [] {
            const char* const hint = std::getenv("EDEN_NX2_COLD_HINT");
            const bool enabled = hint == nullptr || hint[0] != '0';
            const char* const fix = std::getenv("EDEN_NX2_FIX_COLD_HINT_BUDGET");
            const bool budgeted = fix == nullptr || fix[0] != '0';
            u64 mib = 16;
            if (const char* const value = std::getenv("EDEN_NX2_COLD_HINT_MB_PER_FRAME")) {
                mib = std::strtoull(value, nullptr, 10);
            }
            const u64 budget = budgeted ? (std::max)(mib, u64{1}) << 20 : 0;
            if (!enabled) {
                LOG_INFO(HW_GPU, "EDEN_NX2_COLD_HINT=0: uploaded guest memory is not marked cold");
            } else if (budget != 0) {
                LOG_INFO(HW_GPU,
                         "EDEN_NX2_FIX_COLD_HINT_BUDGET: uploaded guest memory is marked cold over "
                         "the frames after each 600-frame tick, {} MiB a frame",
                         budget >> 20);
            } else {
                LOG_INFO(HW_GPU, "EDEN_NX2_FIX_COLD_HINT_BUDGET=0: uploaded guest memory is marked "
                                 "cold all at once every 600 frames");
            }
            return Config{enabled, budget};
        }();
        return config;
    }

    void Report(const char* cache) {
        LOG_INFO(HW_GPU, "{} cold hints: {} ranges, {} MiB over {} frames, {:.2f} ms", cache,
                 hinted_ranges, hinted_bytes >> 20, (std::max)(hinted_frames, u64{1}),
                 std::chrono::duration<double, std::milli>(hinted_time).count());
        hinted_ranges = 0;
        hinted_bytes = 0;
        hinted_frames = 0;
        hinted_time = {};
    }

    std::vector<std::pair<DAddr, u64>> collected;
    std::deque<std::pair<DAddr, u64>> pending;
    u64 hinted_ranges{};
    u64 hinted_bytes{};
    u64 hinted_frames{};
    std::chrono::steady_clock::duration hinted_time{};
};

} // namespace VideoCommon
