// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <string_view>
#ifdef _MSC_VER
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

#include "common/common_types.h"
#include "common/cpu_features.h"
#include "common/thread_cpu_clock.h"

// The guest side of EDEN_NX2_STUTTER_TRACE=1 (video_core/nx2_stutter_trace.h): what the host
// threads of the guest cores and the guest threads did in each presented frame. Frames over 33 ms
// get an Nx2StutterGuest line after their Nx2Stutter line, and every summary an
// Nx2StutterGuestSummary line. EDEN_NX2_GUEST_TRACE=0 leaves this part out of the trace.
// Multicore only (a host thread per guest core).
namespace Common::Nx2GuestTrace {

[[nodiscard]] inline bool ReadEnv() {
    const char* const trace = std::getenv("EDEN_NX2_STUTTER_TRACE");
    if (trace == nullptr || trace[0] != '1') {
        return false;
    }
    const char* const guest = std::getenv("EDEN_NX2_GUEST_TRACE");
    return guest == nullptr || guest[0] != '0';
}

// A plain flag: the hooks check it on every SVC and scheduler lock.
inline bool g_enabled = ReadEnv();

/// Called with StutterTrace::ReadSwitch, once nx2_switches.txt is applied.
inline void ReadSwitch() {
    g_enabled = ReadEnv();
}

[[nodiscard]] inline bool Enabled() {
    return g_enabled;
}

/// What everything here measures in, the same as Nx2Profile::Ticks: the time stamp counter on
/// x86-64. From the compiler's own macros: every library that includes this must agree on it.
[[nodiscard]] inline u64 Ticks() {
#if defined(__x86_64__) || defined(_M_X64)
    return __rdtsc();
#else
    return static_cast<u64>(g_wall_clock.GetTimeNS().count());
#endif
}

/// What the host thread of a guest core is doing. Each core's wall time is split between them.
enum class CoreState : u8 {
    Kernel, ///< Scheduler, interrupts, fiber switches and everything not below
    Guest,  ///< Guest code (the JIT's Run), Jit and Fault excluded
    Idle,   ///< No runnable guest thread (PhysicalCore::Idle, its spin included)
    Svc,    ///< SVC handlers, on the core (a thread blocked in an SVC is off the core)
    Hle,    ///< HLE service requests handled by a service thread that runs on a guest core
    Jit,    ///< Blocks compiled by the JIT (translate, optimize, emit)
    Fault,  ///< The fastmem signal handler and guest reads of GPU-written memory (GPU::OnCPURead)
    Count,
};

inline constexpr size_t CoreStateCount = static_cast<size_t>(CoreState::Count);
inline constexpr std::array<std::string_view, CoreStateCount> core_state_names{
    "kernel", "guest", "idle", "svc", "hle", "jit", "fault",
};

inline constexpr size_t MaxCores = 8;
/// What CurrentCore returns on a thread that is not a guest core's.
inline constexpr size_t NoCore = MaxCores;

/// Host-side stalls: timed where they happen, on any thread. The frame lines give those of the
/// guest cores' threads apart from those of the other threads (services, GPU).
enum class Stall : u8 {
    Sigsegv,        ///< The JIT's SIGSEGV/SIGBUS handler (fastmem faults): time in the handler
    JitCompile,     ///< A block compiled by the JIT
    JitInvalidate,  ///< Code cache ranges invalidated, or the whole cache cleared
    MemoryWrite,    ///< A write to GPU-tracked memory (Memory::HandleRasterizerWrite)
    MemoryRead,     ///< A read of GPU-tracked memory (Memory::HandleRasterizerDownload): count only
    CpuRead,        ///< GPU::OnCPURead flushing GPU-written memory for a guest read
    GpuInvalidate,  ///< GPU::InvalidateRegion, its flush on the GPU thread included
    GpuFlush,       ///< GPU::FlushAndInvalidateRegion
    SchedulerLock,  ///< The kernel's scheduler lock, when another thread held it
    GpuSync,        ///< Nested: GPU::WaitForSyncOperation (the GPU thread running a request)
    ReadbackFence,  ///< Nested: a reading thread waiting for the host GPU to run its download
    DownloadLanding, ///< Nested: a write waiting for a download landing on its bytes
    ServiceLock,    ///< A service's lock, taken for each of its requests
    HleRequest,     ///< HLE service requests (ServiceFrameworkBase::HandleSyncRequest)
    GpuSubmit,      ///< nvhost_gpu::SubmitGPFIFOImpl (the pushbuffer snapshot included)
    Count,
};

inline constexpr size_t StallCount = static_cast<size_t>(Stall::Count);
inline constexpr std::array<std::string_view, StallCount> stall_names{
    "sigsegv",  "jit",            "jit_invalidate",   "mem_write",    "mem_read",
    "cpu_read", "gpu_invalidate", "gpu_flush",        "sched_lock",   "gpu_sync",
    "readback_fence", "download_landing", "service_lock", "hle_request", "gpu_submit",
};

/// The calling thread's guest core, NoCore when it is not a core's host thread. Never inlined:
/// a fiber resumes on another host thread, where an address computed before the switch would
/// still name the first thread's storage.
[[nodiscard]] size_t CurrentCore();

/// Names the calling thread as a guest core's host thread for its lifetime.
class CoreRegistration {
public:
    explicit CoreRegistration(size_t core_);
    ~CoreRegistration();
    CoreRegistration(const CoreRegistration&) = delete;
    CoreRegistration& operator=(const CoreRegistration&) = delete;

private:
    ThreadCpuClock clock;
    size_t core;
};

/// The core's host thread enters and leaves guest code running the guest thread.
void EnterGuest(size_t core, u64 thread_id);
void LeaveGuest(size_t core, u64 thread_id);

/// The core's host thread starts and stops waiting for a runnable guest thread.
void EnterIdle(size_t core);
void LeaveIdle(size_t core);

/// The guest thread's fiber leaves its core (runnable: preempted, otherwise blocked). Returns
/// the Ticks to give OnCore when the fiber runs again, on whichever core that is.
[[nodiscard]] u64 OffCore(u64 thread_id, bool runnable);
void OnCore(u64 thread_id, u64 off_start);

struct SvcStart {
    u64 start{};   ///< Ticks, 0: not traced
    u64 off{};     ///< The thread's time off its core so far
    u64 preempt{}; ///< Of off: while runnable
};

struct SvcEnd {
    u64 end{};
    u64 host{};      ///< On the core
    u64 off{};       ///< Off the core: blocked or preempted
    u64 preempt{};   ///< Of off: preempted while runnable
    bool longest{};  ///< The thread's longest wait of the frame so far: give it a detail
    bool slow{};     ///< Over the slow threshold on the core: give it a detail
};

/// The guest thread enters an SVC, and returns from it.
[[nodiscard]] SvcStart EnterSvc(u64 thread_id, u32 svc);
[[nodiscard]] SvcEnd LeaveSvc(u64 thread_id, u32 svc, const SvcStart& start);
/// What the SVC waited on, for the SVCs LeaveSvc asked a detail of.
void SvcDetail(u64 thread_id, u32 svc, const SvcEnd& end, std::string_view detail);

/// The SVC's name if it is one that waits or is otherwise worth a name, empty otherwise.
[[nodiscard]] std::string_view SvcName(u32 svc);

/// Whether the thread has no name yet and it is time to look for one.
[[nodiscard]] bool WantsName(u64 thread_id);
void SetName(u64 thread_id, std::string_view name);

/// The service and command of the request the thread last sent, if it sent it since `since`
/// (Ticks): "service:command", empty otherwise.
[[nodiscard]] std::array<char, 40> LastRequest(u64 thread_id, u64 since);

/// An HLE service request from the client thread, handled by the handler thread.
class IpcScope {
public:
    IpcScope(std::string_view service, u32 command_, u64 handler_, u64 client_);
    ~IpcScope();
    IpcScope(const IpcScope&) = delete;
    IpcScope& operator=(const IpcScope&) = delete;

private:
    u64 start;
    u64 handler;
    u64 client;
    u64 outer_client;
    u64 off; ///< The handler's time off its core before the request
    u32 command;
    CoreState outer_state;
    std::array<char, 24> name{};
};

/// The thread submitted GPU commands, or it handles the request of the thread that did.
void OnGpuSubmit(u64 thread_id);
/// The thread presented a frame (nx2:vi2 PresentClientBuffer).
void OnGuestPresent(u64 thread_id);

/// A stall ran from start to end (Ticks). in_handler: from a signal handler, where a long one
/// is not listed.
void AddStall(Stall stall, u64 start, u64 end, bool in_handler = false);
void CountStall(Stall stall);
/// Makes state the calling core thread's state and returns the one it replaces (Count when the
/// thread is no core's). For work that cannot switch fibers before RestoreState.
[[nodiscard]] CoreState ReplaceState(CoreState state, u64 now);
void RestoreState(CoreState state, u64 now);

/// Ticks while the trace is enabled, otherwise 0 (which End ignores).
[[nodiscard]] inline u64 Begin() {
    return Enabled() ? Ticks() : 0;
}

inline void End(Stall stall, u64 start) {
    if (start != 0) {
        AddStall(stall, start, Ticks());
    }
}

/// Times a stall while the trace is enabled, as the given core state if any. No fiber switch may
/// happen inside it.
class StallScope {
public:
    explicit StallScope(Stall stall_, CoreState state = CoreState::Count, bool in_handler_ = false)
        : stall{stall_}, in_handler{in_handler_} {
        if (Enabled()) {
            start = Ticks();
            if (state != CoreState::Count) {
                outer = ReplaceState(state, start);
            }
        }
    }
    ~StallScope() {
        if (start != 0) {
            const u64 end = Ticks();
            if (outer != CoreState::Count) {
                RestoreState(outer, end);
            }
            AddStall(stall, start, end, in_handler);
        }
    }
    StallScope(const StallScope&) = delete;
    StallScope& operator=(const StallScope&) = delete;

private:
    Stall stall;
    bool in_handler;
    CoreState outer{CoreState::Count};
    u64 start{};
};

struct Frame {
    u64 number{};
    u64 start_ticks{};
    u64 end_ticks{};
    u64 start_mono_ns{}; ///< steady_clock (CLOCK_MONOTONIC) at the frame's first and last present
    u64 end_mono_ns{};
    double ms_per_tick{};
    double interval_ms{};
    bool slow{}; ///< Logged as Nx2Stutter: gets an Nx2StutterGuest line
};

/// Called by the stutter trace for each presented frame, by one thread at a time.
void OnFrame(const Frame& frame);
/// Called with each Nx2StutterSummary: logs the window's totals and starts the next window.
void Summarize(double window_s);
/// Called when a GPU is created: frames and windows start over.
void StartSession();

} // namespace Common::Nx2GuestTrace
