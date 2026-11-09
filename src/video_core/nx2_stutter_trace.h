// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "common/common_types.h"

// EDEN_NX2_STUTTER_TRACE=1: per presented frame, the wall time each GPU thread (front, back,
// VulkanWorker) spent by category, from the EDEN_NX2_GPU_PROFILE zones (nx2_profile.h maps each
// zone to a category). Frames presented more than 33 ms after the previous one are logged with
// their top category (Nx2Stutter); every 10 s the frame-time histogram, pipeline builds and
// lookups (Nx2StutterSummary); and the boot phases with the process memory (Nx2Boot).
// tools/nx2/cache_metrics.py reads them. Frames longer than EDEN_NX2_STUTTER_DETAIL_MS (default
// 50) also get an Nx2StutterDetail line: each thread's largest zones and the sites below that ran.
namespace VideoCommon::StutterTrace {

enum class Category : u8 {
    GraphicsMiss,  ///< Graphics pipeline built on the GPU thread (environments, translation, SPIR-V)
    ComputeMiss,   ///< Compute pipeline built on the GPU thread
    PipelineWait,  ///< VulkanWorker holding draws and dispatches for pipelines still building
    ShaderCapture, ///< Programs captured, lowered and hashed when the shader cache first sees them
    TextureCache,  ///< Uploads, decoding, downloads, joins and blits, view and target lookups
    BufferCache,   ///< Buffer binding and uploads
    QueryFence,    ///< Waits for GPU ticks, fences and command list readbacks
    Present,       ///< Swapchain frame acquire and present
    Idle,          ///< Waiting for work or for the other GPU thread (never a cause)
    Count,
    None = Count,
};

inline constexpr size_t CategoryCount = static_cast<size_t>(Category::Count);
inline constexpr std::array<std::string_view, CategoryCount> category_names{
    "gfx_miss", "compute_miss", "pipeline_wait", "shader_capture", "texture",
    "buffer",   "query_fence",  "present",       "idle",
};

enum class Thread : u8 {
    Front,  ///< The GPU thread (all of the work when the split is off)
    Back,   ///< GPUBack (gpu_split.h)
    Worker, ///< VulkanWorker
    Count,
};

inline constexpr size_t ThreadCount = static_cast<size_t>(Thread::Count);

/// Pipeline lookups, counted at each pipeline's first use in the session, and native program
/// lookups (shader cache v2), at each program's first bind.
enum class Lookup : u8 {
    GraphicsPrebuilt,   ///< Built from the disk cache at boot
    GraphicsBackground, ///< Built from the disk cache by the background prebuild
    GraphicsRecorded,   ///< In the disk cache but not built (skipped at boot, or evicted)
    GraphicsNew,        ///< Not in the disk cache
    ComputePrebuilt,
    ComputeNew,
    ProgramHit,   ///< Native program from a shader cache v2 record, not lowered
    ProgramMiss,  ///< Native program lowered: no record
    ProgramStale, ///< Native program lowered: a record of another build's lowering
    Count,
};

inline constexpr size_t LookupCount = static_cast<size_t>(Lookup::Count);

/// Pipelines created through shader cache v2's driver layer (VK_KHR_pipeline_binary).
enum class DriverBuild : u8 {
    Binary,   ///< From the binaries stored for it
    Compiled, ///< Compiled by the driver, its binaries stored
    Rejected, ///< Stored binaries that were stale or the driver refused (it is then compiled)
    Count,
};

inline constexpr size_t DriverBuildCount = static_cast<size_t>(DriverBuild::Count);

/// Room for every Nx2Profile::Zone (nx2_profile.h checks it).
inline constexpr size_t ZoneSlots = 64;

/// Rare operations that can make one frame long on their own, timed apart from the zones they run
/// in and on every thread, guest threads included.
enum class Site : u8 {
    MemoryAllocate,  ///< Device memory blocks allocated (vkAllocateMemory under VMA)
    StagingCreate,   ///< A staging buffer the pool didn't have
    BufferCreate,    ///< A buffer cache buffer, with the overlaps it joins
    BufferDelete,    ///< A buffer cache buffer deleted
    BufferCollect,   ///< Buffer cache garbage collection, its downloads included
    BufferDownload,  ///< Buffer data downloaded while the caller waits for the host GPU
    ImageCreate,     ///< A texture cache image, with the images it joins
    ImageDelete,     ///< A texture cache image deleted
    TextureCollect,  ///< Texture cache garbage collection
    TextureDownload, ///< Image data downloaded while the caller waits for the host GPU
    SchedulerFinish, ///< Submitting and waiting for the host GPU to finish all of it
    MemoryFree,      ///< Device memory blocks given back to the driver
    MacroDmaFlush,   ///< A macro's DMA read writing out GPU-written memory before reading it
    GpuInvalidate,   ///< Gathering the CPU's writes and invalidating the caches for them
    WaitForIdle,     ///< The rasterizer's wait for idle
    RasterizerFlushRegion, ///< RasterizerVulkan::FlushRegion, the downloads it makes included
    Count,
};

inline constexpr size_t SiteCount = static_cast<size_t>(Site::Count);
inline constexpr std::array<std::string_view, SiteCount> site_names{
    "memory_allocate", "staging_create", "buffer_create",    "buffer_delete",
    "buffer_collect",  "buffer_download", "image_create",    "image_delete",
    "texture_collect", "texture_download", "scheduler_finish", "memory_free",
    "macro_dma_flush", "gpu_invalidate",  "wait_for_idle",    "flush_region",
};

[[nodiscard]] inline bool ReadEnv() {
    const char* const value = std::getenv("EDEN_NX2_STUTTER_TRACE");
    return value != nullptr && value[0] == '1';
}

inline bool g_enabled = ReadEnv();

/// Called with Nx2Profile::ReadSwitch, once nx2_switches.txt is applied.
inline void ReadSwitch() {
    g_enabled = ReadEnv();
}

[[nodiscard]] inline bool Enabled() {
    return g_enabled;
}

struct Totals {
    /// Monotonic, each written only by its own thread; frames take the differences.
    std::array<std::array<std::atomic<u64>, CategoryCount>, ThreadCount> ticks{};
    std::array<std::array<std::atomic<u64>, CategoryCount>, ThreadCount> counts{};
    /// Each thread's innermost open zone and the Nx2Profile::Ticks from which its time is not in
    /// ticks yet, so that a frame also counts the zones still open when it is presented.
    std::array<std::atomic<u8>, ThreadCount> open{};
    std::array<std::atomic<u64>, ThreadCount> open_since{};
    /// The innermost open Nx2Profile::Zone (ZoneSlots: none), timed from open_since like open.
    std::array<std::atomic<u8>, ThreadCount> open_zone{};
    /// Odd while the thread updates ticks and its open zone (a sequence lock for Snapshot)
    std::array<std::atomic<u32>, ThreadCount> sequence{};
    std::array<std::atomic<bool>, ThreadCount> running{};
    std::array<std::atomic<u64>, LookupCount> lookups{};
    std::array<std::atomic<u64>, DriverBuildCount> driver_builds{};
    /// Each zone's exclusive time (nested zones excluded) and closes, by thread.
    std::array<std::array<std::atomic<u64>, ZoneSlots>, ThreadCount> zone_ticks{};
    std::array<std::array<std::atomic<u64>, ZoneSlots>, ThreadCount> zone_counts{};
    /// Sites run on any thread: added atomically.
    std::array<std::atomic<u64>, SiteCount> site_ticks{};
    std::array<std::atomic<u64>, SiteCount> site_counts{};
};

inline Totals g_totals;
inline thread_local size_t t_thread = ThreadCount;
inline thread_local Category t_open = Category::None;
inline thread_local u64 t_open_since{};
inline thread_local size_t t_open_zone = ZoneSlots;

/// Makes category (and zone) the calling thread's open zone from now on, adding the time of the
/// zone open until now to its ticks (and one to its count when that zone closes).
inline void SwitchOpen(Category category, size_t zone, u64 now, bool closing) {
    const size_t thread = t_thread;
    auto& sequence = g_totals.sequence[thread];
    const u32 begin = sequence.load(std::memory_order_relaxed);
    sequence.store(begin + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    if (t_open != Category::None) {
        const auto index = static_cast<size_t>(t_open);
        auto& total = g_totals.ticks[thread][index];
        total.store(total.load(std::memory_order_relaxed) + (now - t_open_since),
                    std::memory_order_relaxed);
        if (closing) {
            auto& count = g_totals.counts[thread][index];
            count.store(count.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        }
    }
    if (t_open_zone < ZoneSlots) {
        auto& total = g_totals.zone_ticks[thread][t_open_zone];
        total.store(total.load(std::memory_order_relaxed) + (now - t_open_since),
                    std::memory_order_relaxed);
        if (closing) {
            auto& count = g_totals.zone_counts[thread][t_open_zone];
            count.store(count.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        }
    }
    g_totals.open[thread].store(static_cast<u8>(category), std::memory_order_relaxed);
    g_totals.open_zone[thread].store(static_cast<u8>(zone), std::memory_order_relaxed);
    g_totals.open_since[thread].store(now, std::memory_order_relaxed);
    sequence.store(begin + 2, std::memory_order_release);
    t_open = category;
    t_open_zone = zone;
    t_open_since = now;
}

/// Names the calling thread for its lifetime.
class ThreadScope {
public:
    explicit ThreadScope(Thread thread) : index{static_cast<size_t>(thread)} {
        t_thread = index;
        t_open = Category::None;
        t_open_zone = ZoneSlots;
        g_totals.open[index].store(static_cast<u8>(Category::None), std::memory_order_relaxed);
        g_totals.open_zone[index].store(static_cast<u8>(ZoneSlots), std::memory_order_relaxed);
        g_totals.running[index].store(true, std::memory_order_relaxed);
    }
    ~ThreadScope() {
        g_totals.running[index].store(false, std::memory_order_relaxed);
        g_totals.open[index].store(static_cast<u8>(Category::None), std::memory_order_relaxed);
        g_totals.open_zone[index].store(static_cast<u8>(ZoneSlots), std::memory_order_relaxed);
        t_thread = ThreadCount;
    }
    ThreadScope(const ThreadScope&) = delete;
    ThreadScope& operator=(const ThreadScope&) = delete;

private:
    size_t index;
};

/// A zone of the category opens at now (Nx2Profile::Ticks) on the calling thread. Returns the
/// category it nests in, which Close reopens. zone is its Nx2Profile::Zone (ZoneSlots: none).
/// Unnamed threads are ignored.
inline Category Open(Category category, u64 now, size_t zone = ZoneSlots) {
    if (t_thread >= ThreadCount) {
        return Category::None;
    }
    const Category outer = t_open;
    SwitchOpen(category, zone, now, false);
    return outer;
}

/// The innermost open zone closes at now, back in outer (what its Open returned) and outer_zone.
inline void Close(Category outer, u64 now, size_t outer_zone = ZoneSlots) {
    if (t_thread < ThreadCount) {
        SwitchOpen(outer, outer_zone, now, true);
    }
}

/// A site ran from start to end (Nx2Profile::Ticks); long ones are kept for the frame's detail.
void AddSite(Site site, u64 start, u64 end);

/// Times a site while the trace is enabled.
class SiteScope {
public:
    explicit SiteScope(Site site_) : site{site_} {
        if (Enabled()) {
            start = Now();
        }
    }
    ~SiteScope() {
        if (start != 0) {
            AddSite(site, start, Now());
        }
    }
    SiteScope(const SiteScope&) = delete;
    SiteScope& operator=(const SiteScope&) = delete;

    /// Nx2Profile::Ticks without including nx2_profile.h (which includes this header).
    static u64 Now();

private:
    Site site;
    u64 start{};
};

inline void CountLookup(Lookup lookup) {
    g_totals.lookups[static_cast<size_t>(lookup)].fetch_add(1, std::memory_order_relaxed);
}

inline void CountDriverBuild(DriverBuild build) {
    g_totals.driver_builds[static_cast<size_t>(build)].fetch_add(1, std::memory_order_relaxed);
}

using ThreadTicks = std::array<std::array<u64, CategoryCount>, ThreadCount>;

/// Each thread's ticks at now (Nx2Profile::Ticks), its open zone's time so far included.
[[nodiscard]] ThreadTicks Snapshot(u64 now);

struct Attribution {
    std::array<double, CategoryCount> category_ms{}; ///< Summed over the threads
    std::array<std::optional<double>, ThreadCount> busy_ms{}; ///< Running threads: not idle
    double other_ms{}; ///< Busy time no category explains, on the thread with the most of it
    std::string_view cause; ///< The largest of the categories (Idle excluded) and other
};

/// What one frame's time went to, from the ticks each thread added during it.
[[nodiscard]] Attribution Attribute(const ThreadTicks& ticks,
                                    const std::array<bool, ThreadCount>& running,
                                    double interval_ms, double ms_per_tick);

/// Called by the renderer for each presented frame, by one thread at a time.
void OnPresent();

/// Logs a boot phase with the time and the process memory. Reads the environment itself: the
/// boot starts before the video core applies nx2_switches.txt.
void BootMarker(std::string_view event, std::string_view details = {});

/// The disk pipeline cache file, whose size the summaries log.
void SetCacheFile(std::string path);

/// Called when a GPU is created: frames and summaries start over (the totals stay monotonic).
void StartSession();

} // namespace VideoCommon::StutterTrace
