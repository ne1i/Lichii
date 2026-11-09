// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "common/container/unordered_set.h"
#include "core/hardware_properties.h"

namespace Dynarmic::A64 {
class Jit;
}

namespace Core {

/// JIT warm-up (EDEN_NX2_JIT_WARMUP, default on; =0 off). The blocks execution compiles in the
/// application's code image are recorded, relative to the image, in
/// <cache dir>/jit/<program id>-<build id>.blocks. The next launch compiles them on each guest
/// core's host thread while that core has nothing to run, before the game reaches them. Blocks
/// one core compiles are queued for the others as well (EDEN_NX2_JIT_PROPAGATE=0: not).
/// EDEN_NX2_JIT_WARMUP_RECORD=0: the file is read, not written.
class JitWarmup {
public:
    struct Image {
        u64 program_id{};
        std::array<u8, 0x20> build_id{};
        u64 base{}; ///< Start of the code image (rtld, main, subsdk*, sdk)
        u64 size{};
    };

    /// nullptr when switched off.
    [[nodiscard]] static std::shared_ptr<JitWarmup> Create(const Image& image);

    explicit JitWarmup(const Image& image);
    ~JitWarmup();
    JitWarmup(const JitWarmup&) = delete;
    JitWarmup& operator=(const JitWarmup&) = delete;

    /// Execution compiled the block at location (an IR::LocationDescriptor value) on core.
    void OnCompiled(size_t core, u64 location);

    /// On core's host thread while it has nothing to run: compiles listed blocks into jit until
    /// stop is set, the time slice ends or nothing is left. Returns whether it compiled or checked
    /// any block.
    bool Run(size_t core, Dynarmic::A64::Jit& jit, const std::atomic<bool>& stop);

private:
    struct alignas(64) CoreWork {
        size_t cursor{}; ///< Next index of list to compile (the core's thread only)
        bool full{};     ///< Out of code space: no more precompiles (the core's thread only)
        bool reported_done{};
        std::atomic<u64> compiled{};
        std::atomic<u64> present{};
        std::atomic<u64> unreadable{};
        std::atomic<u64> compile_ns{};
        std::atomic<u64> cache_bytes{};
        std::atomic<u64> late{};    ///< Execution compiled a listed block this core had not yet
        std::atomic<u64> unlisted{}; ///< Execution compiled a block no list had
        std::atomic<u64> outside{}; ///< Execution compiled a block outside the code image
    };

    void Load();
    void WriterLoop();
    void MaybeLog(bool force);

    [[nodiscard]] bool InImage(u64 location) const;
    [[nodiscard]] u64 ToEntry(u64 location) const;
    [[nodiscard]] u64 ToLocation(u64 entry) const;

    Image image;
    std::filesystem::path path;
    bool record{};
    bool propagate{};
    u64 slice_us{};
    size_t keep_free{};
    size_t max_entries{};
    size_t loaded{};

    std::mutex mutex;
    Common::unordered_set<u64> known; ///< Entries listed or recorded
    std::vector<u64> list;            ///< Locations to precompile, in first-compiled order
    std::atomic<size_t> list_size{};
    std::vector<u64> pending;         ///< Entries to append to the file
    std::condition_variable pending_cv;
    bool file_ok{true};
    bool stopping{};

    std::array<CoreWork, Hardware::NUM_CPU_CORES> cores{};
    std::atomic<u64> last_log_ns{};
    std::atomic<u32> done_cores{};

    std::thread writer; ///< Joined (after a last write) by the destructor
};

} // namespace Core
