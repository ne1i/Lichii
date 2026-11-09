// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <string_view>

#include "common/common_types.h"

namespace Core {
class System;
}

// The Legend of Zelda: Ocarina of Time (0400486024DEC000) decompresses its streamed data with a
// statically linked zstd while it plays; in Kokiri Forest the JIT-compiled sequence decoder takes
// about a quarter of the two busiest guest cores. The loader replaces the first instruction of the
// game's ZSTD_decompressMultiFrame (a pacibsp, which the JIT treats as a no-op) with an SVC
// Horizon doesn't have. Svc::Call hands that SVC to Handle, which decompresses with the host's
// zstd (with the game's dictionaries) and returns to the caller. A call the host's zstd refuses
// continues into the game's own code after the SVC, so results never differ from the game's.
// While it plays, the game streams through its own chunked decompressor, which calls zstd's
// block internals: their entries are replaced too, and the host keeps a twin of each of the
// game's ZSTD_DCtx for the frames it decodes.
namespace Core::OotZstd {

/// An SVC number outside Horizon's range.
constexpr u32 SvcNumber = 0x7f5a;

/// Patches the module image when it is Ocarina's main of the known build.
void PatchModule(u64 program_id, std::span<const u8> build_id, std::string_view name,
                 std::span<u8> image, u64 load_base);

/// Handles SvcNumber for the current thread.
void Handle(Core::System& system);

} // namespace Core::OotZstd
