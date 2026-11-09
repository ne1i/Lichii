// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>

#include "common/common_types.h"
#include "common/nx2_guest_trace.h"

namespace Core {
class System;
}

// The SVC side of the guest trace (common/nx2_guest_trace.h): each SVC's time on its core and
// off it, and what the long ones waited on.
namespace Kernel::Svc::Nx2SvcTrace {

/// Lives on the stack of the SVC: the thread can return from it on another core.
struct Token {
    Common::Nx2GuestTrace::SvcStart start{};
    u64 thread_id{};
    std::array<u64, 4> args{}; ///< The SVC's first arguments (the results overwrite them)
};

[[nodiscard]] Token Begin(Core::System& system, u32 imm, const std::array<u64, 8>& args);

/// Out of line: only the thread's longest waits and SVCs slow on their core get here.
void Detail(Core::System& system, u32 imm, const Token& token,
            const Common::Nx2GuestTrace::SvcEnd& end, u64 result, u64 out);
void NameThread(Core::System& system, u64 thread_id);

/// result and out: the SVC's first two results.
inline void End(Core::System& system, u32 imm, const Token& token, u64 result, u64 out) {
    if (token.start.start == 0) {
        return;
    }
    const auto end = Common::Nx2GuestTrace::LeaveSvc(token.thread_id, imm, token.start);
    if (end.longest || end.slow) [[unlikely]] {
        Detail(system, imm, token, end, result, out);
    }
    if (Common::Nx2GuestTrace::WantsName(token.thread_id)) [[unlikely]] {
        NameThread(system, token.thread_id);
    }
}

} // namespace Kernel::Svc::Nx2SvcTrace
