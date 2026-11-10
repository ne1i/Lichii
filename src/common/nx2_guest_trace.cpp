// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/assert.h"
#include "common/logging.h"
#include "common/nx2_guest_trace.h"

namespace Common::Nx2GuestTrace {
namespace {

constexpr auto Relaxed = std::memory_order_relaxed;

/// Guest threads by id modulo this; a thread that finds another's data in its slot clears it.
constexpr size_t SlotCount = 512;
/// SVCs over this on their core, and stalls over it, are listed one by one in the frame's line.
constexpr double SlowMs = 2.0;
/// Waits under this are not worth resolving what they waited on.
constexpr double MinWaitMs = 0.2;
constexpr size_t MaxLongEvents = 128;
constexpr size_t FrameThreads = 6;
constexpr size_t SummarySvcs = 8;
constexpr size_t SvcSlots = 256;
/// The stalls of the guest cores' threads, then those of every other thread.
constexpr size_t StallGroups = 2;
constexpr std::array<std::string_view, StallGroups> stall_group_names{"cores", "other"};
constexpr u64 Never = ~u64{0};

/// Text a thread writes while another may read it: words, so a read is torn at worst.
template <size_t Words>
struct Text {
    std::array<std::atomic<u64>, Words> words{};

    void Set(std::string_view text) {
        for (size_t word = 0; word < Words; ++word) {
            u64 value = 0;
            const size_t offset = word * sizeof(u64);
            if (offset < text.size()) {
                std::memcpy(&value, text.data() + offset,
                            (std::min)(sizeof(u64), text.size() - offset));
            }
            words[word].store(value, Relaxed);
        }
    }

    [[nodiscard]] std::array<char, Words * sizeof(u64) + 1> Get() const {
        std::array<char, Words * sizeof(u64) + 1> text{};
        for (size_t word = 0; word < Words; ++word) {
            const u64 value = words[word].load(Relaxed);
            std::memcpy(text.data() + word * sizeof(u64), &value, sizeof(u64));
        }
        return text;
    }
};

struct alignas(64) CoreData {
    /// Odd while the core's thread updates ticks and its state (a sequence lock for the frames)
    std::atomic<u32> sequence{};
    std::array<std::atomic<u64>, CoreStateCount> ticks{};
    std::atomic<u8> state{};
    std::atomic<u64> since{};
    std::atomic<bool> running{};
};

/// A thread's longest wait of a frame.
struct WaitRecord {
    std::atomic<u32> epoch{}; ///< The frame epoch it belongs to, plus one
    std::atomic<u32> svc{};
    std::atomic<u64> end{};
    std::atomic<u64> off{};
    std::atomic<u64> preempt{};
    Text<6> detail;
};

/// A guest thread. Written by the thread itself, whichever host thread runs it; request* by the
/// thread handling its request while it waits.
struct Slot {
    std::atomic<u64> tag{}; ///< Thread id plus one
    std::atomic<u64> run{}; ///< In guest code
    std::atomic<u64> run_since{};
    std::atomic<u64> off{}; ///< Off its core: blocked or preempted
    std::atomic<u64> preempt{};
    std::atomic<u64> off_since{};
    std::atomic<bool> off_runnable{};
    std::atomic<u8> resume_state{};
    std::atomic<u8> core{};
    std::atomic<u8> named{};
    std::atomic<u64> svc_ticks{}; ///< In SVCs, on its core
    std::atomic<u64> svc_count{};
    std::atomic<u32> svc_open{}; ///< The SVC it is in, plus one
    std::atomic<u64> submits{};
    std::atomic<u64> serving{}; ///< The client whose request it handles, plus one
    std::atomic<u64> request_time{};
    std::atomic<u32> request_command{};
    Text<3> request;
    Text<3> name;
    /// By frame epoch parity: the frame's reader takes the one of the epoch that just ended.
    std::array<WaitRecord, 2> waits;
};

struct StallBucket {
    std::array<std::atomic<u64>, StallCount> ticks{};
    std::array<std::atomic<u64>, StallCount> counts{};
    std::array<std::atomic<u64>, StallCount> longest{}; ///< Since the last frame
};

/// The longest time between two events of a frame, and who made them.
struct GapTracker {
    struct Longest {
        std::atomic<u32> epoch{};
        std::atomic<u64> start{};
        std::atomic<u64> end{};
        std::atomic<u64> before{};
        std::atomic<u64> after{};
    };
    std::atomic<u64> last{};
    std::atomic<u64> last_thread{};
    std::atomic<u64> count{};
    std::array<Longest, 2> longest;
};

struct LongEvent {
    u64 end{};
    u64 ticks{};
    std::array<char, 96> text{};
};

std::array<CoreData, MaxCores> g_cores;
std::array<Slot, SlotCount> g_slots;
std::array<StallBucket, MaxCores + 1> g_stalls;
std::array<std::array<std::atomic<u64>, SvcSlots>, MaxCores + 1> g_svc_ticks;
std::array<std::array<std::atomic<u64>, SvcSlots>, MaxCores + 1> g_svc_counts;
GapTracker g_submits;
GapTracker g_presents;
std::atomic<u32> g_epoch{};
/// In Ticks, known from the second frame on.
std::atomic<u64> g_slow_ticks{Never};
std::atomic<u64> g_min_wait_ticks{Never};

std::mutex g_long_mutex;
std::array<LongEvent, MaxLongEvents> g_long_events;
size_t g_long_count{};

std::mutex g_clock_mutex;
std::array<const ThreadCpuClock*, MaxCores> g_clocks{};

thread_local size_t t_core = NoCore;

void Add(std::atomic<u64>& total, u64 value) {
    total.store(total.load(Relaxed) + value, Relaxed);
}

[[nodiscard]] constexpr u64 Sub(u64 a, u64 b) {
    return a > b ? a - b : 0;
}

void Switch(CoreData& core, CoreState state, u64 now) {
    const u32 begin = core.sequence.load(Relaxed);
    core.sequence.store(begin + 1, Relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    Add(core.ticks[core.state.load(Relaxed)], Sub(now, core.since.load(Relaxed)));
    core.state.store(static_cast<u8>(state), Relaxed);
    core.since.store(now, Relaxed);
    core.sequence.store(begin + 2, std::memory_order_release);
}

void Clear(Slot& slot, u64 tag) {
    slot.run.store(0, Relaxed);
    slot.run_since.store(0, Relaxed);
    slot.off.store(0, Relaxed);
    slot.preempt.store(0, Relaxed);
    slot.off_since.store(0, Relaxed);
    slot.off_runnable.store(false, Relaxed);
    slot.resume_state.store(0, Relaxed);
    slot.core.store(static_cast<u8>(NoCore), Relaxed);
    slot.named.store(0, Relaxed);
    slot.svc_ticks.store(0, Relaxed);
    slot.svc_count.store(0, Relaxed);
    slot.svc_open.store(0, Relaxed);
    slot.submits.store(0, Relaxed);
    slot.serving.store(0, Relaxed);
    slot.request_time.store(0, Relaxed);
    slot.request_command.store(0, Relaxed);
    slot.request.Set({});
    slot.name.Set({});
    for (WaitRecord& wait : slot.waits) {
        wait.epoch.store(0, Relaxed);
    }
    slot.tag.store(tag, Relaxed);
}

Slot& SlotOf(u64 thread_id) {
    Slot& slot = g_slots[thread_id % SlotCount];
    if (slot.tag.load(Relaxed) != thread_id + 1) {
        Clear(slot, thread_id + 1);
    }
    return slot;
}

/// Without clearing: for the threads other than thread_id's own.
Slot* FindSlot(u64 thread_id) {
    Slot& slot = g_slots[thread_id % SlotCount];
    return slot.tag.load(Relaxed) == thread_id + 1 ? &slot : nullptr;
}

/// Only what a field of the log lines can hold: no spaces or separators.
std::string Clean(std::string_view text) {
    std::string clean;
    for (const char c : text) {
        if (c == '\0') {
            break;
        }
        const bool keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9') ||
                          std::string_view{"_.-:()+<>"}.find(c) != std::string_view::npos;
        clean += keep ? c : '_';
    }
    return clean.empty() ? std::string{"-"} : clean;
}

std::string ThreadLabel(u64 thread_id) {
    const Slot* const slot = FindSlot(thread_id);
    return fmt::format("{}#{}", slot ? Clean(slot->name.Get().data()) : std::string{"-"},
                       thread_id);
}

std::string SvcLabel(u32 svc) {
    const std::string_view name = SvcName(svc);
    return name.empty() ? fmt::format("svc0x{:X}", svc) : std::string{name};
}

template <typename... Args>
void PushLong(u64 start, u64 end, fmt::format_string<Args...> format, Args&&... args) {
    LongEvent event{.end = end, .ticks = Sub(end, start)};
    const auto result = fmt::format_to_n(event.text.data(), event.text.size() - 1, format,
                                         std::forward<Args>(args)...);
    *result.out = '\0';
    std::scoped_lock lock{g_long_mutex};
    if (g_long_count < MaxLongEvents) {
        g_long_events[g_long_count++] = event;
    }
}

void AddStallTo(Stall stall, u64 start, u64 end, bool list) {
    const size_t core = CurrentCore();
    const auto index = static_cast<size_t>(stall);
    const u64 ticks = Sub(end, start);
    StallBucket& bucket = g_stalls[core];
    bucket.ticks[index].fetch_add(ticks, Relaxed);
    bucket.counts[index].fetch_add(1, Relaxed);
    if (ticks > bucket.longest[index].load(Relaxed)) {
        bucket.longest[index].store(ticks, Relaxed);
    }
    if (list && ticks > g_slow_ticks.load(Relaxed)) {
        if (core < MaxCores) {
            PushLong(start, end, "stall:core{}:{}", core, stall_names[index]);
        } else {
            PushLong(start, end, "stall:other:{}", stall_names[index]);
        }
    }
}

void OnGap(GapTracker& tracker, u64 thread_id, u64 now) {
    const u64 last = tracker.last.exchange(now, Relaxed);
    const u64 before = tracker.last_thread.exchange(thread_id + 1, Relaxed);
    tracker.count.fetch_add(1, Relaxed);
    const u32 epoch = g_epoch.load(Relaxed);
    auto& longest = tracker.longest[epoch % 2];
    const bool fresh = longest.epoch.load(Relaxed) != epoch + 1;
    if (last == 0 || (!fresh && Sub(now, last) <= Sub(longest.end.load(Relaxed),
                                                       longest.start.load(Relaxed)))) {
        return;
    }
    longest.start.store(last, Relaxed);
    longest.end.store(now, Relaxed);
    longest.before.store(before, Relaxed);
    longest.after.store(thread_id + 1, Relaxed);
    longest.epoch.store(epoch + 1, Relaxed);
}

// What follows is the frames' side: one thread at a time (the one presenting).

struct SlotTotals {
    u64 tag{};
    u64 run{};
    u64 svc{};
    u64 off{};
    u64 preempt{};
    u64 svcs{};
    u64 submits{};
};

using CoreTicks = std::array<std::array<u64, CoreStateCount>, MaxCores>;
using StallTotals = std::array<std::array<u64, StallCount>, StallGroups>;
using SvcTotals = std::array<u64, SvcSlots>;

struct Reader {
    bool primed{};
    CoreTicks core_ticks{};
    std::array<u64, MaxCores> cpu_ns{};
    std::array<SlotTotals, SlotCount> slots{};
    StallTotals stall_ticks{};
    StallTotals stall_counts{};
    u64 submits{};
    u64 presents{};

    // The summary's window
    std::array<std::array<double, CoreStateCount>, MaxCores> window_core_ms{};
    std::array<double, MaxCores> window_cpu_ms{};
    std::array<SlotTotals, SlotCount> window_slots{};
    std::array<std::array<double, StallCount>, StallGroups> window_stall_ms{};
    StallTotals window_stall_counts{};
    SvcTotals window_svc_ticks{};
    SvcTotals window_svc_counts{};
    u64 window_frames{};
    u64 window_slow_frames{};
    u64 window_long_events{};
    u64 window_submits{};
    u64 window_presents{};
    double window_submit_gap_ms{};
    double window_present_gap_ms{};
    double ms_per_tick{};
};

Reader g_reader;

CoreTicks SnapshotCores(u64 now) {
    CoreTicks ticks{};
    for (size_t index = 0; index < MaxCores; ++index) {
        const CoreData& core = g_cores[index];
        for (;;) {
            const u32 begin = core.sequence.load(std::memory_order_acquire);
            if (begin % 2 != 0) {
                std::this_thread::yield();
                continue;
            }
            for (size_t state = 0; state < CoreStateCount; ++state) {
                ticks[index][state] = core.ticks[state].load(Relaxed);
            }
            const size_t open = core.state.load(Relaxed);
            const u64 since = core.since.load(Relaxed);
            const bool running = core.running.load(Relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (core.sequence.load(Relaxed) != begin) {
                continue;
            }
            if (running && open < CoreStateCount) {
                ticks[index][open] += Sub(now, since);
            }
            break;
        }
    }
    return ticks;
}

/// The thread's totals at now, the time it has been running or off its core included.
SlotTotals SnapshotSlot(const Slot& slot, u64 now) {
    SlotTotals totals{
        .tag = slot.tag.load(Relaxed),
        .run = slot.run.load(Relaxed),
        .svc = slot.svc_ticks.load(Relaxed),
        .off = slot.off.load(Relaxed),
        .preempt = slot.preempt.load(Relaxed),
        .svcs = slot.svc_count.load(Relaxed),
        .submits = slot.submits.load(Relaxed),
    };
    if (const u64 since = slot.run_since.load(Relaxed); since != 0) {
        totals.run += Sub(now, since);
    }
    if (const u64 since = slot.off_since.load(Relaxed); since != 0) {
        totals.off += Sub(now, since);
        if (slot.off_runnable.load(Relaxed)) {
            totals.preempt += Sub(now, since);
        }
    }
    return totals;
}

SlotTotals Difference(const SlotTotals& now, const SlotTotals& last) {
    if (now.tag != last.tag) {
        return now;
    }
    return {
        .tag = now.tag,
        .run = Sub(now.run, last.run),
        .svc = Sub(now.svc, last.svc),
        .off = Sub(now.off, last.off),
        .preempt = Sub(now.preempt, last.preempt),
        .svcs = Sub(now.svcs, last.svcs),
        .submits = Sub(now.submits, last.submits),
    };
}

SvcTotals SumSvcs(const std::array<std::array<std::atomic<u64>, SvcSlots>, MaxCores + 1>& table) {
    SvcTotals totals{};
    for (const auto& core : table) {
        for (size_t svc = 0; svc < SvcSlots; ++svc) {
            totals[svc] += core[svc].load(Relaxed);
        }
    }
    return totals;
}

struct FrameWait {
    double ms{};
    double preempt_ms{};
    std::string kind;
};

/// The thread's longest wait that ended in the frame, or the one it is still in if longer; only
/// the part inside the frame.
FrameWait LongestWait(const Slot& slot, const Frame& frame, u32 epoch) {
    FrameWait wait;
    const WaitRecord& record = slot.waits[epoch % 2];
    if (record.epoch.load(Relaxed) == epoch + 1) {
        const u64 end = record.end.load(Relaxed);
        const u64 off = (std::min)(record.off.load(Relaxed), Sub(end, frame.start_ticks));
        wait.ms = static_cast<double>(off) * frame.ms_per_tick;
        wait.preempt_ms = static_cast<double>((std::min)(record.preempt.load(Relaxed), off)) *
                          frame.ms_per_tick;
        wait.kind = fmt::format("{}({})", SvcLabel(record.svc.load(Relaxed)),
                                Clean(record.detail.Get().data()));
    }
    const u64 since = slot.off_since.load(Relaxed);
    if (since == 0 || since >= frame.end_ticks) {
        return wait;
    }
    const double open_ms = static_cast<double>(frame.end_ticks -
                                               (std::max)(since, frame.start_ticks)) *
                           frame.ms_per_tick;
    if (open_ms > wait.ms) {
        const bool runnable = slot.off_runnable.load(Relaxed);
        const u32 svc = slot.svc_open.load(Relaxed);
        wait.ms = open_ms;
        wait.preempt_ms = runnable ? open_ms : 0.0;
        wait.kind = fmt::format("{}(open)", svc != 0   ? SvcLabel(svc - 1)
                                            : runnable ? std::string{"preempted"}
                                                       : std::string{"blocked"});
    }
    return wait;
}

struct FrameGap {
    double ms{};
    std::string text;
};

/// "ms:start-end:before>after", start and end in steady_clock ns; "open" when the frame ended
/// before the next event.
FrameGap LongestGap(const GapTracker& tracker, const Frame& frame, u32 epoch) {
    const auto mono = [&frame](u64 ticks) {
        const double ns = (static_cast<double>(frame.end_ticks) - static_cast<double>(ticks)) *
                          frame.ms_per_tick * 1e6;
        return static_cast<s64>(frame.end_mono_ns) - static_cast<s64>(ns);
    };
    const auto label = [](u64 tag) { return tag == 0 ? std::string{"-"} : ThreadLabel(tag - 1); };
    u64 start = 0;
    u64 end = 0;
    std::string threads = "->-";
    const auto& longest = tracker.longest[epoch % 2];
    if (longest.epoch.load(Relaxed) == epoch + 1) {
        start = longest.start.load(Relaxed);
        end = longest.end.load(Relaxed);
        threads = fmt::format("{}>{}", label(longest.before.load(Relaxed)),
                              label(longest.after.load(Relaxed)));
    }
    const u64 last = tracker.last.load(Relaxed);
    if (last != 0 && last < frame.end_ticks && frame.end_ticks - last > Sub(end, start)) {
        start = last;
        end = frame.end_ticks;
        threads = fmt::format("{}>open", label(tracker.last_thread.load(Relaxed)));
    }
    if (start == 0) {
        return {.ms = 0.0, .text = "none"};
    }
    const double ms = static_cast<double>(Sub(end, start)) * frame.ms_per_tick;
    return {.ms = ms, .text = fmt::format("{:.1f}:{}-{}:{}", ms, mono(start), mono(end), threads)};
}

} // Anonymous namespace

YUZU_NO_INLINE size_t CurrentCore() {
    return t_core;
}

CoreRegistration::CoreRegistration(size_t core_)
    : clock{Enabled()}, core{core_ < MaxCores ? core_ : NoCore} {
    t_core = core;
    if (core == NoCore) {
        return;
    }
    CoreData& data = g_cores[core];
    data.state.store(static_cast<u8>(CoreState::Kernel), Relaxed);
    data.since.store(Ticks(), Relaxed);
    data.running.store(true, Relaxed);
    std::scoped_lock lock{g_clock_mutex};
    g_clocks[core] = &clock;
}

CoreRegistration::~CoreRegistration() {
    t_core = NoCore;
    if (core == NoCore) {
        return;
    }
    CoreData& data = g_cores[core];
    Switch(data, CoreState::Kernel, Ticks());
    data.running.store(false, Relaxed);
    std::scoped_lock lock{g_clock_mutex};
    g_clocks[core] = nullptr;
}

void EnterGuest(size_t core, u64 thread_id) {
    if (core >= MaxCores) {
        return;
    }
    const u64 now = Ticks();
    Slot& slot = SlotOf(thread_id);
    slot.core.store(static_cast<u8>(core), Relaxed);
    slot.run_since.store(now, Relaxed);
    Switch(g_cores[core], CoreState::Guest, now);
}

void LeaveGuest(size_t core, u64 thread_id) {
    if (core >= MaxCores) {
        return;
    }
    const u64 now = Ticks();
    Switch(g_cores[core], CoreState::Kernel, now);
    Slot& slot = SlotOf(thread_id);
    if (const u64 since = slot.run_since.load(Relaxed); since != 0) {
        Add(slot.run, Sub(now, since));
        slot.run_since.store(0, Relaxed);
    }
}

void EnterIdle(size_t core) {
    if (core < MaxCores) {
        Switch(g_cores[core], CoreState::Idle, Ticks());
    }
}

void LeaveIdle(size_t core) {
    if (core < MaxCores) {
        Switch(g_cores[core], CoreState::Kernel, Ticks());
    }
}

u64 OffCore(u64 thread_id, bool runnable) {
    const u64 now = Ticks();
    Slot& slot = SlotOf(thread_id);
    const size_t core = CurrentCore();
    if (core < MaxCores) {
        slot.resume_state.store(g_cores[core].state.load(Relaxed), Relaxed);
        Switch(g_cores[core], CoreState::Kernel, now);
    }
    slot.off_runnable.store(runnable, Relaxed);
    slot.off_since.store(now, Relaxed);
    return now;
}

void OnCore(u64 thread_id, u64 off_start) {
    if (off_start == 0) {
        return;
    }
    const u64 now = Ticks();
    Slot& slot = SlotOf(thread_id);
    if (slot.off_since.load(Relaxed) != 0) {
        Add(slot.off, Sub(now, off_start));
        if (slot.off_runnable.load(Relaxed)) {
            Add(slot.preempt, Sub(now, off_start));
        }
        slot.off_since.store(0, Relaxed);
    }
    // The fiber may have resumed on another host thread than it left.
    if (const size_t core = CurrentCore(); core < MaxCores) {
        const u8 state = slot.resume_state.load(Relaxed);
        Switch(g_cores[core], state < CoreStateCount ? static_cast<CoreState>(state)
                                                     : CoreState::Kernel,
               now);
    }
}

SvcStart EnterSvc(u64 thread_id, u32 svc) {
    const u64 now = Ticks();
    Slot& slot = SlotOf(thread_id);
    slot.svc_open.store(svc + 1, Relaxed);
    if (const size_t core = CurrentCore(); core < MaxCores) {
        Switch(g_cores[core], CoreState::Svc, now);
    }
    return {.start = now, .off = slot.off.load(Relaxed), .preempt = slot.preempt.load(Relaxed)};
}

SvcEnd LeaveSvc(u64 thread_id, u32 svc, const SvcStart& start) {
    if (start.start == 0) {
        return {};
    }
    const u64 now = Ticks();
    Slot& slot = SlotOf(thread_id);
    const size_t core = CurrentCore();
    if (core < MaxCores) {
        Switch(g_cores[core], CoreState::Kernel, now);
    }
    SvcEnd end{.end = now};
    end.off = (std::min)(Sub(slot.off.load(Relaxed), start.off), Sub(now, start.start));
    end.preempt = (std::min)(Sub(slot.preempt.load(Relaxed), start.preempt), end.off);
    end.host = Sub(now, start.start) - end.off;
    slot.svc_open.store(0, Relaxed);
    Add(slot.svc_ticks, end.host);
    Add(slot.svc_count, 1);
    const size_t index = (std::min<size_t>)(svc, SvcSlots - 1);
    Add(g_svc_ticks[core][index], end.host);
    Add(g_svc_counts[core][index], 1);

    const u32 epoch = g_epoch.load(Relaxed);
    const WaitRecord& record = slot.waits[epoch % 2];
    const u64 longest = record.epoch.load(Relaxed) == epoch + 1 ? record.off.load(Relaxed) : 0;
    end.longest = end.off >= g_min_wait_ticks.load(Relaxed) && end.off > longest;
    end.slow = end.host > g_slow_ticks.load(Relaxed);
    return end;
}

void SvcDetail(u64 thread_id, u32 svc, const SvcEnd& end, std::string_view detail) {
    Slot& slot = SlotOf(thread_id);
    if (end.longest) {
        const u32 epoch = g_epoch.load(Relaxed);
        WaitRecord& record = slot.waits[epoch % 2];
        record.svc.store(svc, Relaxed);
        record.end.store(end.end, Relaxed);
        record.off.store(end.off, Relaxed);
        record.preempt.store(end.preempt, Relaxed);
        record.detail.Set(detail);
        record.epoch.store(epoch + 1, Relaxed);
    }
    if (end.slow) {
        // Listed as if it ran last: the frames only need its length and that it ended in them.
        PushLong(Sub(end.end, end.host), end.end, "svc:{}#{}:{}({})",
                 Clean(slot.name.Get().data()), thread_id, SvcLabel(svc), Clean(detail));
    }
}

std::string_view SvcName(u32 svc) {
    switch (svc) {
    case 0x01:
        return "SetHeapSize";
    case 0x0B:
        return "SleepThread";
    case 0x18:
        return "WaitSynchronization";
    case 0x1A:
        return "ArbitrateLock";
    case 0x1B:
        return "ArbitrateUnlock";
    case 0x1C:
        return "WaitProcessWideKeyAtomic";
    case 0x1D:
        return "SignalProcessWideKey";
    case 0x21:
        return "SendSyncRequest";
    case 0x22:
        return "SendSyncRequestWithUserBuffer";
    case 0x34:
        return "WaitForAddress";
    case 0x35:
        return "SignalToAddress";
    case 0x43:
        return "ReplyAndReceive";
    case 0x7F5A:
        return "OotZstd";
    default:
        return {};
    }
}

bool WantsName(u64 thread_id) {
    Slot& slot = SlotOf(thread_id);
    // Threads are often named after they start: look again now and then, a few times.
    const u8 tries = slot.named.load(Relaxed);
    return tries < 4 && slot.svc_count.load(Relaxed) % 1024 == 1;
}

void SetName(u64 thread_id, std::string_view name) {
    Slot& slot = SlotOf(thread_id);
    slot.name.Set(name);
    slot.named.store(name.empty() ? static_cast<u8>(slot.named.load(Relaxed) + 1) : u8{4},
                     Relaxed);
}

std::array<char, 40> LastRequest(u64 thread_id, u64 since) {
    std::array<char, 40> text{};
    const Slot& slot = SlotOf(thread_id);
    if (slot.request_time.load(Relaxed) >= since && since != 0) {
        const auto result = fmt::format_to_n(text.data(), text.size() - 1, "{}:{}",
                                             slot.request.Get().data(),
                                             slot.request_command.load(Relaxed));
        *result.out = '\0';
    }
    return text;
}

IpcScope::IpcScope(std::string_view service, u32 command_, u64 handler_, u64 client_)
    : start{Enabled() ? Ticks() : 0}, handler{handler_}, client{client_}, outer_client{},
      off{}, command{command_}, outer_state{CoreState::Count} {
    if (start == 0) {
        return;
    }
    if (!service.empty()) {
        std::memcpy(name.data(), service.data(), (std::min)(service.size(), name.size() - 1));
    }
    Slot& client_slot = SlotOf(client);
    client_slot.request.Set(service);
    client_slot.request_command.store(command, Relaxed);
    client_slot.request_time.store(start, Relaxed);
    Slot& slot = SlotOf(handler);
    outer_client = slot.serving.exchange(client + 1, Relaxed);
    off = slot.off.load(Relaxed);
    outer_state = ReplaceState(CoreState::Hle, start);
}

IpcScope::~IpcScope() {
    if (start == 0) {
        return;
    }
    const u64 end = Ticks();
    Slot& slot = SlotOf(handler);
    slot.serving.store(outer_client, Relaxed);
    RestoreState(outer_state, end);
    // On a guest core the handler can wait as a fiber: that time is not the request's.
    const u64 host = Sub(Sub(end, start), Sub(slot.off.load(Relaxed), off));
    AddStallTo(Stall::HleRequest, end - host, end, false);
    if (host > g_slow_ticks.load(Relaxed)) {
        PushLong(end - host, end, "ipc:{}:{}:{}#{}", Clean(name.data()), command,
                 Clean(SlotOf(client).name.Get().data()), client);
    }
}

void OnGpuSubmit(u64 thread_id) {
    const Slot& slot = SlotOf(thread_id);
    const u64 serving = slot.serving.load(Relaxed);
    const u64 submitter = serving != 0 ? serving - 1 : thread_id;
    SlotOf(submitter).submits.fetch_add(1, Relaxed);
    OnGap(g_submits, submitter, Ticks());
}

void OnGuestPresent(u64 thread_id) {
    const Slot& slot = SlotOf(thread_id);
    const u64 serving = slot.serving.load(Relaxed);
    OnGap(g_presents, serving != 0 ? serving - 1 : thread_id, Ticks());
}

void AddStall(Stall stall, u64 start, u64 end, bool in_handler) {
    AddStallTo(stall, start, end, !in_handler);
}

void CountStall(Stall stall) {
    g_stalls[CurrentCore()].counts[static_cast<size_t>(stall)].fetch_add(1, Relaxed);
}

CoreState ReplaceState(CoreState state, u64 now) {
    const size_t core = CurrentCore();
    if (core >= MaxCores) {
        return CoreState::Count;
    }
    const u8 outer = g_cores[core].state.load(Relaxed);
    Switch(g_cores[core], state, now);
    return static_cast<CoreState>(outer);
}

void RestoreState(CoreState state, u64 now) {
    if (state == CoreState::Count) {
        return;
    }
    if (const size_t core = CurrentCore(); core < MaxCores) {
        Switch(g_cores[core], state, now);
    }
}

void OnFrame(const Frame& frame) {
    auto& reader = g_reader;
    const double ms_per_tick = frame.ms_per_tick;
    const auto to_ms = [ms_per_tick](u64 ticks) { return static_cast<double>(ticks) * ms_per_tick; };
    if (ms_per_tick > 0.0) {
        g_slow_ticks.store(static_cast<u64>(SlowMs / ms_per_tick), Relaxed);
        g_min_wait_ticks.store(static_cast<u64>(MinWaitMs / ms_per_tick), Relaxed);
        reader.ms_per_tick = ms_per_tick;
    }
    // Waits and gaps recorded from here on are the next frame's.
    const u32 epoch = g_epoch.fetch_add(1, Relaxed);

    const CoreTicks core_ticks = SnapshotCores(frame.end_ticks);
    const CoreTicks last_core_ticks = std::exchange(reader.core_ticks, core_ticks);
    std::array<u64, MaxCores> cpu_ns{};
    {
        std::scoped_lock lock{g_clock_mutex};
        for (size_t core = 0; core < MaxCores; ++core) {
            cpu_ns[core] = g_clocks[core] ? g_clocks[core]->Read().value_or(0) : 0;
        }
    }
    const auto last_cpu_ns = std::exchange(reader.cpu_ns, cpu_ns);

    StallTotals stall_ticks{};
    StallTotals stall_counts{};
    StallTotals stall_longest{};
    for (size_t bucket = 0; bucket <= MaxCores; ++bucket) {
        const size_t group = bucket < MaxCores ? 0 : 1;
        for (size_t stall = 0; stall < StallCount; ++stall) {
            stall_ticks[group][stall] += g_stalls[bucket].ticks[stall].load(Relaxed);
            stall_counts[group][stall] += g_stalls[bucket].counts[stall].load(Relaxed);
            stall_longest[group][stall] =
                (std::max)(stall_longest[group][stall],
                           g_stalls[bucket].longest[stall].exchange(0, Relaxed));
        }
    }
    const StallTotals last_stall_ticks = std::exchange(reader.stall_ticks, stall_ticks);
    const StallTotals last_stall_counts = std::exchange(reader.stall_counts, stall_counts);
    const u64 submits = g_submits.count.load(Relaxed);
    const u64 presents = g_presents.count.load(Relaxed);
    const u64 frame_submits = Sub(submits, std::exchange(reader.submits, submits));
    const u64 frame_presents = Sub(presents, std::exchange(reader.presents, presents));

    std::array<LongEvent, MaxLongEvents> long_events;
    size_t long_count;
    {
        std::scoped_lock lock{g_long_mutex};
        long_count = std::exchange(g_long_count, size_t{0});
        std::copy_n(g_long_events.begin(), long_count, long_events.begin());
    }

    struct FrameThread {
        size_t slot;
        SlotTotals frame;
    };
    std::vector<FrameThread> threads;
    for (size_t index = 0; index < SlotCount; ++index) {
        const SlotTotals totals = SnapshotSlot(g_slots[index], frame.end_ticks);
        const SlotTotals last = std::exchange(reader.slots[index], totals);
        if (!reader.primed) {
            reader.window_slots[index] = totals;
        }
        if (frame.slow && reader.primed && totals.tag != 0) {
            threads.push_back({index, Difference(totals, last)});
        }
    }
    if (!reader.primed) {
        reader.primed = true;
        reader.window_svc_ticks = SumSvcs(g_svc_ticks);
        reader.window_svc_counts = SumSvcs(g_svc_counts);
        return;
    }

    ++reader.window_frames;
    reader.window_submits += frame_submits;
    reader.window_presents += frame_presents;
    reader.window_long_events += long_count;
    std::string cores;
    for (size_t core = 0; core < MaxCores; ++core) {
        if (!g_cores[core].running.load(Relaxed)) {
            continue;
        }
        cores += fmt::format("{}{}", cores.empty() ? "" : ",", core);
        for (size_t state = 0; state < CoreStateCount; ++state) {
            const double ms = to_ms(Sub(core_ticks[core][state], last_core_ticks[core][state]));
            reader.window_core_ms[core][state] += ms;
            cores += fmt::format("{}{}:{:.1f}", state == 0 ? ":" : "/", core_state_names[state],
                                 ms);
        }
        const double cpu_ms = static_cast<double>(Sub(cpu_ns[core], last_cpu_ns[core])) / 1e6;
        reader.window_cpu_ms[core] += cpu_ms;
        cores += fmt::format("/cpu:{:.1f}", cpu_ms);
    }
    std::string stalls;
    for (size_t group = 0; group < StallGroups; ++group) {
        std::string text;
        for (size_t stall = 0; stall < StallCount; ++stall) {
            const u64 count = Sub(stall_counts[group][stall], last_stall_counts[group][stall]);
            const double ms = to_ms(Sub(stall_ticks[group][stall], last_stall_ticks[group][stall]));
            reader.window_stall_counts[group][stall] += count;
            reader.window_stall_ms[group][stall] += ms;
            if (count != 0) {
                text += fmt::format("{}{}:{}x{:.1f}^{:.1f}", text.empty() ? "" : ",",
                                    stall_names[stall], count, ms,
                                    to_ms(stall_longest[group][stall]));
            }
        }
        if (!text.empty()) {
            stalls += fmt::format("{}{}:{}", stalls.empty() ? "" : ";", stall_group_names[group],
                                  text);
        }
    }
    const FrameGap submit_gap = LongestGap(g_submits, frame, epoch);
    const FrameGap present_gap = LongestGap(g_presents, frame, epoch);
    reader.window_submit_gap_ms = (std::max)(reader.window_submit_gap_ms, submit_gap.ms);
    reader.window_present_gap_ms = (std::max)(reader.window_present_gap_ms, present_gap.ms);
    if (!frame.slow) {
        return;
    }
    ++reader.window_slow_frames;

    // The threads most on their core, then the ones submitting to the GPU and presenting.
    const auto on_core = [](const FrameThread& thread) { return thread.frame.run + thread.frame.svc; };
    std::sort(threads.begin(), threads.end(),
              [&on_core](const FrameThread& a, const FrameThread& b) {
                  return on_core(a) > on_core(b);
              });
    const u64 submitter = g_submits.last_thread.load(Relaxed);
    const u64 presenter = g_presents.last_thread.load(Relaxed);
    std::string thread_text;
    for (size_t rank = 0; rank < threads.size(); ++rank) {
        const FrameThread& thread = threads[rank];
        const bool gpu = thread.frame.tag == submitter || thread.frame.submits != 0;
        const bool present = thread.frame.tag == presenter;
        if ((rank >= FrameThreads || on_core(thread) == 0) && !gpu && !present) {
            continue;
        }
        const Slot& slot = g_slots[thread.slot];
        const FrameWait wait = LongestWait(slot, frame, epoch);
        thread_text += fmt::format(
            "{}{}/c{}/run:{:.1f}/svc:{:.1f}/blocked:{:.1f}/preempt:{:.1f}/svcs:{}/submits:{}",
            thread_text.empty() ? "" : ",", ThreadLabel(thread.frame.tag - 1),
            u32{slot.core.load(Relaxed)}, to_ms(thread.frame.run), to_ms(thread.frame.svc),
            to_ms(Sub(thread.frame.off, thread.frame.preempt)), to_ms(thread.frame.preempt),
            thread.frame.svcs, thread.frame.submits);
        if (wait.ms > 0.0) {
            thread_text += fmt::format("/wait:{:.1f}:{:.1f}:{}", wait.ms, wait.preempt_ms,
                                       wait.kind);
        }
        if (gpu) {
            thread_text += "/gpu";
        }
        if (present) {
            thread_text += "/present";
        }
    }
    std::string slow;
    for (size_t index = 0; index < long_count; ++index) {
        const LongEvent& event = long_events[index];
        if (event.end <= frame.start_ticks) {
            continue;
        }
        slow += fmt::format("{}{}:{:.1f}", slow.empty() ? "" : ",", event.text.data(),
                            to_ms(event.ticks));
    }
    LOG_INFO(HW_GPU,
             "Nx2StutterGuest frame={} ms={:.1f} mono={}-{} cores={} threads={} stalls={} slow={} "
             "submits={} submit_gap={} presents={} present_gap={}",
             frame.number, frame.interval_ms, frame.start_mono_ns, frame.end_mono_ns,
             cores.empty() ? "none" : cores, thread_text.empty() ? "none" : thread_text,
             stalls.empty() ? "none" : stalls, slow.empty() ? "none" : slow, frame_submits,
             submit_gap.text, frame_presents, present_gap.text);
}

void Summarize(double window_s) {
    auto& reader = g_reader;
    if (!reader.primed || window_s <= 0.0) {
        return;
    }
    const double ms_per_tick = reader.ms_per_tick;
    // Per second of the window: ms/s and events/s.
    const auto rate = [window_s](double value) { return value / window_s; };
    std::string cores;
    for (size_t core = 0; core < MaxCores; ++core) {
        if (!g_cores[core].running.load(Relaxed)) {
            continue;
        }
        cores += fmt::format("{}{}", cores.empty() ? "" : ",", core);
        for (size_t state = 0; state < CoreStateCount; ++state) {
            cores += fmt::format("{}{}:{:.1f}", state == 0 ? ":" : "/", core_state_names[state],
                                 rate(reader.window_core_ms[core][state]));
        }
        cores += fmt::format("/cpu:{:.1f}", rate(reader.window_cpu_ms[core]));
    }
    struct WindowThread {
        SlotTotals window;
    };
    std::vector<WindowThread> threads;
    for (size_t index = 0; index < SlotCount; ++index) {
        const SlotTotals& totals = reader.slots[index];
        if (totals.tag != 0) {
            threads.push_back({Difference(totals, reader.window_slots[index])});
        }
        reader.window_slots[index] = totals;
    }
    const auto on_core = [](const WindowThread& thread) {
        return thread.window.run + thread.window.svc;
    };
    std::sort(threads.begin(), threads.end(),
              [&on_core](const WindowThread& a, const WindowThread& b) {
                  return on_core(a) > on_core(b);
              });
    std::string thread_text;
    for (size_t rank = 0; rank < threads.size(); ++rank) {
        const SlotTotals& window = threads[rank].window;
        if ((rank >= FrameThreads || on_core(threads[rank]) == 0) && window.submits == 0) {
            continue;
        }
        thread_text += fmt::format(
            "{}{}/run:{:.1f}/svc:{:.1f}/blocked:{:.1f}/preempt:{:.1f}/svcs:{:.0f}/submits:{:.1f}",
            thread_text.empty() ? "" : ",", ThreadLabel(window.tag - 1),
            rate(static_cast<double>(window.run) * ms_per_tick),
            rate(static_cast<double>(window.svc) * ms_per_tick),
            rate(static_cast<double>(Sub(window.off, window.preempt)) * ms_per_tick),
            rate(static_cast<double>(window.preempt) * ms_per_tick),
            rate(static_cast<double>(window.svcs)), rate(static_cast<double>(window.submits)));
    }
    const SvcTotals svc_ticks = SumSvcs(g_svc_ticks);
    const SvcTotals svc_counts = SumSvcs(g_svc_counts);
    std::array<size_t, SvcSlots> order;
    for (size_t svc = 0; svc < SvcSlots; ++svc) {
        order[svc] = svc;
    }
    const auto svc_window = [&svc_ticks](size_t svc) {
        return Sub(svc_ticks[svc], reader.window_svc_ticks[svc]);
    };
    std::partial_sort(order.begin(), order.begin() + SummarySvcs, order.end(),
                      [&svc_window](size_t a, size_t b) { return svc_window(a) > svc_window(b); });
    std::string svcs;
    for (size_t rank = 0; rank < SummarySvcs; ++rank) {
        const size_t svc = order[rank];
        const u64 count = Sub(svc_counts[svc], reader.window_svc_counts[svc]);
        if (count == 0) {
            break;
        }
        // The last slot holds every SVC number from it up.
        svcs += fmt::format("{}{}:{:.0f}x{:.1f}", svcs.empty() ? "" : ",",
                            svc == SvcSlots - 1 ? std::string{"other"}
                                                : SvcLabel(static_cast<u32>(svc)),
                            rate(static_cast<double>(count)),
                            rate(static_cast<double>(svc_window(svc)) * ms_per_tick));
    }
    reader.window_svc_ticks = svc_ticks;
    reader.window_svc_counts = svc_counts;
    std::string stalls;
    for (size_t group = 0; group < StallGroups; ++group) {
        std::string text;
        for (size_t stall = 0; stall < StallCount; ++stall) {
            if (reader.window_stall_counts[group][stall] == 0) {
                continue;
            }
            text += fmt::format("{}{}:{:.0f}x{:.1f}", text.empty() ? "" : ",", stall_names[stall],
                                rate(static_cast<double>(reader.window_stall_counts[group][stall])),
                                rate(reader.window_stall_ms[group][stall]));
        }
        if (!text.empty()) {
            stalls += fmt::format("{}{}:{}", stalls.empty() ? "" : ";", stall_group_names[group],
                                  text);
        }
    }
    LOG_INFO(HW_GPU,
             "Nx2StutterGuestSummary window_s={:.2f} frames={} slow_frames={} cores={} threads={} "
             "svcs={} stalls={} long_events={:.1f} submits={:.1f} presents={:.1f} "
             "max_submit_gap_ms={:.1f} max_present_gap_ms={:.1f}",
             window_s, reader.window_frames, reader.window_slow_frames,
             cores.empty() ? "none" : cores, thread_text.empty() ? "none" : thread_text,
             svcs.empty() ? "none" : svcs, stalls.empty() ? "none" : stalls,
             rate(static_cast<double>(reader.window_long_events)),
             rate(static_cast<double>(reader.window_submits)),
             rate(static_cast<double>(reader.window_presents)), reader.window_submit_gap_ms,
             reader.window_present_gap_ms);
    reader.window_core_ms = {};
    reader.window_cpu_ms = {};
    reader.window_stall_ms = {};
    reader.window_stall_counts = {};
    reader.window_frames = 0;
    reader.window_slow_frames = 0;
    reader.window_long_events = 0;
    reader.window_submits = 0;
    reader.window_presents = 0;
    reader.window_submit_gap_ms = 0.0;
    reader.window_present_gap_ms = 0.0;
}

void StartSession() {
    // The totals stay monotonic; the first frame takes them as the session's start.
    g_reader.primed = false;
    g_reader.window_core_ms = {};
    g_reader.window_cpu_ms = {};
    g_reader.window_stall_ms = {};
    g_reader.window_stall_counts = {};
    g_reader.window_frames = 0;
    g_reader.window_slow_frames = 0;
    g_reader.window_long_events = 0;
    g_reader.window_submits = 0;
    g_reader.window_presents = 0;
    g_reader.window_submit_gap_ms = 0.0;
    g_reader.window_present_gap_ms = 0.0;
    std::scoped_lock lock{g_long_mutex};
    g_long_count = 0;
}

} // namespace Common::Nx2GuestTrace
