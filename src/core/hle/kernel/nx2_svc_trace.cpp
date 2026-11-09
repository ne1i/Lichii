// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <string>

#include <fmt/format.h>

#include "core/arm/debug.h"
#include "core/core.h"
#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/kernel/k_handle_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/nx2_svc_trace.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/memory.h"

namespace Kernel::Svc::Nx2SvcTrace {
namespace {

/// Handles of a WaitSynchronization named in its detail.
constexpr s32 MaxHandles = 3;

/// "KReadableEvent=<the name its service gave it>", or the object's type.
std::string HandleName(KernelCore& kernel, KProcess& process, u32 handle) {
    KScopedAutoObject object = process.GetHandleTable().GetObject<KAutoObject>(kernel, handle);
    if (object.IsNull()) {
        return "closed";
    }
    if (auto* const event = object->DynamicCast<KReadableEvent*>(); event != nullptr) {
        return fmt::format("Event:{}", Service::KernelHelpers::DebugEventName(event));
    }
    if (auto* const thread = object->DynamicCast<KThread*>(); thread != nullptr) {
        return fmt::format("Thread:{}", thread->GetThreadId());
    }
    return object->GetTypeName();
}

} // Anonymous namespace

Token Begin(Core::System& system, u32 imm, const std::array<u64, 8>& args) {
    Token token;
    token.thread_id = GetCurrentThread(system.Kernel()).GetThreadId();
    std::copy_n(args.begin(), token.args.size(), token.args.begin());
    token.start = Common::Nx2GuestTrace::EnterSvc(token.thread_id, imm);
    return token;
}

void Detail(Core::System& system, u32 imm, const Token& token,
            const Common::Nx2GuestTrace::SvcEnd& end, u64 result, u64 out) {
    auto& kernel = system.Kernel();
    auto& process = GetCurrentProcess(kernel);
    const std::string_view timed_out =
        static_cast<u32>(result) == ResultTimedOut.raw ? ":timeout" : "";
    std::string detail;
    switch (imm) {
    case 0x0B: // SleepThread(ns)
        detail = fmt::format("{}us", static_cast<s64>(token.args[0]) / 1000);
        break;
    case 0x18: { // WaitSynchronization(out index, handles, count, timeout): the woken one first
        const s32 args_count = static_cast<s32>(token.args[2]);
        const s32 count =
            args_count > 0 && process.GetMemory().IsValidVirtualAddressRange(
                                  token.args[1], 4 * static_cast<u64>(args_count))
                ? args_count
                : 0;
        const s32 woken = result == 0 ? static_cast<s32>(out) : -1;
        const auto handle_at = [&process, &token](s32 index) {
            return process.GetMemory().Read32(token.args[1] + 4 * static_cast<u64>(index));
        };
        if (woken >= 0 && woken < count) {
            detail = HandleName(kernel, process, handle_at(woken));
        }
        for (s32 index = 0; index < (std::min)(count, MaxHandles); ++index) {
            if (index != woken) {
                detail += fmt::format("{}{}", detail.empty() ? "" : "+",
                                      HandleName(kernel, process, handle_at(index)));
            }
        }
        detail += fmt::format("{}:of{}{}", detail.empty() ? "none" : "", count, timed_out);
        break;
    }
    case 0x1A: { // ArbitrateLock(owner thread handle, address, tag)
        KScopedAutoObject owner = process.GetHandleTable().GetObject<KThread>(
            kernel, static_cast<u32>(token.args[0]));
        detail = fmt::format("owner#{}:mutex:{:x}",
                             owner.IsNotNull() ? std::to_string(owner->GetThreadId())
                                               : std::string{"closed"},
                             token.args[1]);
        break;
    }
    case 0x1C: // WaitProcessWideKeyAtomic(mutex address, condvar key, tag, timeout)
        detail = fmt::format("cv:{:x}:mutex:{:x}{}", token.args[1], token.args[0], timed_out);
        break;
    case 0x21: // SendSyncRequest: the HLE service names itself as it handles the request
    case 0x22: {
        const auto request =
            Common::Nx2GuestTrace::LastRequest(token.thread_id, token.start.start);
        detail = request[0] != '\0' ? request.data() : "session";
        break;
    }
    case 0x34: // WaitForAddress(address, type, value, timeout)
        detail = fmt::format("addr:{:x}:type{}{}", token.args[0], token.args[1], timed_out);
        break;
    default:
        break;
    }
    Common::Nx2GuestTrace::SvcDetail(token.thread_id, imm, end, detail);
}

void NameThread(Core::System& system, u64 thread_id) {
    const auto name = Core::GetThreadName(GetCurrentThreadPointer(system.Kernel()));
    Common::Nx2GuestTrace::SetName(thread_id, name ? std::string_view{*name} : std::string_view{});
}

} // namespace Kernel::Svc::Nx2SvcTrace
