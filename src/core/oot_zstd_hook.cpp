// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#define ZSTD_STATIC_LINKING_ONLY
#define ZSTD_DISABLE_DEPRECATE_WARNINGS
#include <zstd.h>

#include "common/hex_util.h"
#include "common/logging.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc_types.h"
#include "core/memory.h"
#include "core/oot_zstd_hook.h"

namespace Core::OotZstd {
namespace {

constexpr u64 ProgramId = 0x0400486024DEC000ULL;
constexpr std::string_view MainBuildId = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348";

/// ZSTD_decompressMultiFrame(dctx, dst, dstCapacity, src, srcSize, dict, dictSize, ddict) in
/// that build's main, and its first nine instructions.
constexpr u64 MultiFrameOffset = 0xEB380;
constexpr std::array<u32, 9> MultiFramePrologue{
    0xd503237f, // pacibsp
    0xa9ba7bfd, // stp x29, x30, [sp, #-0x60]!
    0xa9016ffc, // stp x28, x27, [sp, #0x10]
    0xa90267fa, // stp x26, x25, [sp, #0x20]
    0x910003fd, // mov x29, sp
    0xa9035ff8, // stp x24, x23, [sp, #0x30]
    0xa90457f6, // stp x22, x21, [sp, #0x40]
    0xa9054ff4, // stp x20, x19, [sp, #0x50]
    0xd11a43ff, // sub sp, sp, #0x690
};

// While it plays, the game streams its data through its own chunked decompressor
// (main+0xF5C80), which drives zstd 1.4's internals block by block:
// ZSTD_decompressBlock_internal(dctx, dst, dstCapacity, src, srcSize, frame) for a block that is
// whole in the chunk, and ZSTD_decompressContinue(dctx, dst, dstCapacity, src, srcSize) for one
// it had to stage. Their first instructions, which Handle redoes when it leaves a call to them.
constexpr u64 BlockOffset = 0xF6490;
constexpr std::array<u32, 3> BlockCode{0xd351fc88, 0xb5002de8, 0xd503237f}; // lsr x8, x4, #17
constexpr u64 ContinueOffset = 0xF3880;
constexpr std::array<u32, 4> ContinueCode{0xf9783408, 0xeb08009f, 0x54004a81, 0xd503237f}; // ldr x8, [x0, #0x7068]

// The game's ZSTD_DCtx (zstd 1.4).
constexpr u64 PreviousDstEnd = 0x7048;
constexpr u64 PrefixStart = 0x7050;
constexpr u64 VirtualStart = 0x7058;
constexpr u64 DictEnd = 0x7060;
constexpr u64 Expected = 0x7068;
constexpr u64 FrameContentSize = 0x7070;
constexpr u64 WindowSize = 0x7078;
constexpr u64 FrameDictId = 0x708C;
constexpr u64 ChecksumFlag = 0x7090;
constexpr u64 DecodedSize = 0x7098;
constexpr u64 BlockType = 0x70A0;
constexpr u64 Stage = 0x70A4;
constexpr u64 XxhState = 0x70B0;
constexpr u64 HeaderSize = 0x7108;
constexpr u64 HeaderBuffer = 0x271E0;
constexpr u64 HeaderBufferSize = 18; // ZSTD_FRAMEHEADERSIZE_MAX
constexpr size_t TailCheck = 64;

constexpr u32 BlockTypeCompressed = 2;
constexpr u32 StageDecodeBlockHeader = 2;
constexpr u32 StageDecompressBlock = 3;
constexpr u32 StageDecompressLastBlock = 4;
constexpr u32 StageCheckChecksum = 5;
constexpr u64 BlockSizeMax = 128 * 1024;
constexpr u64 MaxWindowSize = 32ULL << 20;
constexpr u64 ErrorCorruption = static_cast<u64>(-20);
constexpr u64 ErrorDstSizeTooSmall = static_cast<u64>(-70);

/// Larger requests stay with the game's code.
constexpr u64 MaxSourceSize = 64ULL << 20;
constexpr u64 MaxDictionarySize = 16ULL << 20;
constexpr u64 MaxOutputSize = 512ULL << 20;

std::atomic<u64> module_base{};
std::atomic<bool> streaming_hooked{};
std::atomic<u64> native_calls{};
std::atomic<u64> native_bytes{};
std::atomic<u64> fallbacks{};
std::atomic<u64> stream_blocks{};
std::atomic<u64> stream_bytes{};
std::atomic<u64> stream_guest_frames{};

struct DDictDeleter {
    void operator()(ZSTD_DDict* ddict) const {
        ZSTD_freeDDict(ddict);
    }
};

/// A host ZSTD_DDict made from the bytes of a guest dictionary. The bytes are kept to notice a
/// guest address that holds another dictionary later.
struct Dictionary {
    std::vector<u8> content;
    std::unique_ptr<ZSTD_DDict, DDictDeleter> ddict;
};

std::mutex dictionaries_mutex;
std::map<std::tuple<u64, u64, u64>, std::shared_ptr<const Dictionary>> dictionaries;

/// The host dictionary for guest dictionary content at address with size bytes. key_owner is the
/// guest ZSTD_DDict (0 for raw dictionary content).
std::shared_ptr<const Dictionary> GetDictionary(Core::Memory::Memory& memory, u64 key_owner,
                                                u64 address, u64 size) {
    thread_local std::vector<u8> bytes;
    bytes.resize(size);
    if (!memory.ReadBlock(address, bytes.data(), size)) {
        return nullptr;
    }
    const auto key = std::make_tuple(key_owner, address, size);
    std::scoped_lock lock{dictionaries_mutex};
    if (const auto it = dictionaries.find(key); it != dictionaries.end()) {
        if (it->second->content == bytes) {
            return it->second;
        }
        dictionaries.erase(it);
    }
    auto dictionary = std::make_shared<Dictionary>();
    dictionary->content = bytes;
    dictionary->ddict.reset(ZSTD_createDDict_byReference(dictionary->content.data(), size));
    if (!dictionary->ddict) {
        return nullptr;
    }
    if (dictionaries.size() >= 64) {
        dictionaries.clear();
    }
    LOG_INFO(Core, "Ocarina zstd: dictionary {:#x} ({} bytes, id {}) loaded", address, size,
             ZSTD_getDictID_fromDDict(dictionary->ddict.get()));
    return dictionaries.emplace(key, std::move(dictionary)).first->second;
}

void Fallback(const char* reason, u64 src_size, u64 capacity) {
    const u64 count = fallbacks.fetch_add(1, std::memory_order_relaxed);
    if (count < 4 || (count & (count - 1)) == 0) {
        LOG_INFO(Core, "Ocarina zstd: the game's decoder runs ({}, {} bytes into {}), {} so far",
                 reason, src_size, capacity, count + 1);
    }
}

/// ZSTD_decompressMultiFrame natively; false hands the call to the game's code unchanged.
bool DecompressFrames(Core::Memory::Memory& memory, const Kernel::Svc::ThreadContext& ctx,
                      u64& result) {
    const u64 dst = ctx.r[1];
    const u64 capacity = ctx.r[2];
    const u64 src = ctx.r[3];
    const u64 src_size = ctx.r[4];
    const u64 dict = ctx.r[5];
    const u64 dict_size = ctx.r[6];
    const u64 ddict = ctx.r[7];
    std::shared_ptr<const Dictionary> dictionary;
    if (ddict != 0) {
        // struct ZSTD_DDict_s { void* dictBuffer; const void* dictContent; size_t dictSize; ... }
        std::array<u64, 3> header{};
        if (!memory.ReadBlock(ddict, header.data(), sizeof(header)) || header[1] == 0 ||
            header[2] == 0 || header[2] > MaxDictionarySize) {
            Fallback("unreadable ZSTD_DDict", src_size, capacity);
            return false;
        }
        dictionary = GetDictionary(memory, ddict, header[1], header[2]);
    } else if (dict != 0 && dict_size != 0) {
        if (dict_size > MaxDictionarySize) {
            Fallback("dictionary size", src_size, capacity);
            return false;
        }
        dictionary = GetDictionary(memory, 0, dict, dict_size);
    }
    if ((ddict != 0 || (dict != 0 && dict_size != 0)) && !dictionary) {
        Fallback("dictionary", src_size, capacity);
        return false;
    }
    if (src_size == 0 || src_size > MaxSourceSize) {
        Fallback("source size", src_size, capacity);
        return false;
    }
    thread_local std::vector<u8> input;
    thread_local std::vector<u8> output;
    thread_local ZSTD_DCtx* const context = ZSTD_createDCtx();
    if (context == nullptr) {
        return false;
    }
    input.resize(src_size);
    if (!memory.ReadBlock(src, input.data(), src_size)) {
        Fallback("unreadable source", src_size, capacity);
        return false;
    }
    const unsigned long long bound = ZSTD_decompressBound(input.data(), src_size);
    if (bound == ZSTD_CONTENTSIZE_ERROR) {
        Fallback("not a zstd frame", src_size, capacity);
        return false;
    }
    const u64 size = std::min<u64>(bound, capacity);
    if (size > MaxOutputSize) {
        Fallback("output size", src_size, capacity);
        return false;
    }
    output.resize(size);
    const size_t written =
        dictionary ? ZSTD_decompress_usingDDict(context, output.data(), output.size(),
                                                input.data(), src_size, dictionary->ddict.get())
                   : ZSTD_decompressDCtx(context, output.data(), output.size(), input.data(),
                                         src_size);
    if (ZSTD_isError(written)) {
        Fallback(ZSTD_getErrorName(written), src_size, capacity);
        return false;
    }
    if (written != 0 && !memory.WriteBlock(dst, output.data(), written)) {
        Fallback("unwritable destination", src_size, capacity);
        return false;
    }
    result = written;
    if (output.capacity() > (64ULL << 20)) {
        output = {}; // keep no huge buffer per guest thread
    }
    const u64 calls = native_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 bytes = native_bytes.fetch_add(written, std::memory_order_relaxed) + written;
    if (calls == 1 || (calls & (calls - 1)) == 0) {
        LOG_INFO(Core, "Ocarina zstd: {} host decompressions, {} MiB out, {} game fallbacks",
                 calls, bytes >> 20, fallbacks.load(std::memory_order_relaxed));
    }
    return true;
}

// XXH64_update on the game's XXH64_state_t (zstd 1.4: total_len, v1-v4, mem64[4], memsize).
constexpr u64 Prime1 = 11400714785074694791ULL;
constexpr u64 Prime2 = 14029467366897019727ULL;

u64 XxhRound(u64 acc, u64 input) {
    acc += input * Prime2;
    acc = std::rotl(acc, 31);
    return acc * Prime1;
}

struct XxhGuestState {
    u64 total_len;
    std::array<u64, 4> v;
    std::array<u8, 32> mem;
    u32 memsize;
    u32 reserved32;
    u64 reserved64;
};
static_assert(sizeof(XxhGuestState) == 88);

void XxhUpdate(XxhGuestState& state, const u8* input, size_t length) {
    state.total_len += length;
    if (state.memsize + length < 32) {
        std::memcpy(state.mem.data() + state.memsize, input, length);
        state.memsize += static_cast<u32>(length);
        return;
    }
    const u8* const end = input + length;
    const auto lane = [](const u8* p) {
        u64 value;
        std::memcpy(&value, p, sizeof(value));
        return value;
    };
    if (state.memsize != 0) {
        const size_t fill = 32 - state.memsize;
        std::memcpy(state.mem.data() + state.memsize, input, fill);
        for (size_t i = 0; i < 4; ++i) {
            state.v[i] = XxhRound(state.v[i], lane(state.mem.data() + i * 8));
        }
        input += fill;
        state.memsize = 0;
    }
    while (input + 32 <= end) {
        for (size_t i = 0; i < 4; ++i) {
            state.v[i] = XxhRound(state.v[i], lane(input + i * 8));
        }
        input += 32;
    }
    if (input < end) {
        std::memcpy(state.mem.data(), input, static_cast<size_t>(end - input));
        state.memsize = static_cast<u32>(end - input);
    }
}

struct DCtxDeleter {
    void operator()(ZSTD_DCtx* context) const {
        ZSTD_freeDCtx(context);
    }
};

/// The host's twin of one guest ZSTD_DCtx while the host decodes the blocks of its frame. The host
/// writes the frame's output in stream order into one half of mirror at a time; a half holds at
/// least the window and a block, so after a switch the other half is the history zstd still
/// needs. Once the host decodes a block of a frame, it has to decode every compressed block of
/// that frame: the game's entropy tables and repeat offsets are not updated meanwhile.
struct Stream {
    enum class Mode : u8 { Idle, Host, Guest };
    std::atomic<Mode> mode{Mode::Idle};
    std::array<u8, HeaderBufferSize + 8> header{}; ///< the frame's header size and bytes
    std::unique_ptr<ZSTD_DCtx, DCtxDeleter> context;
    std::shared_ptr<const Dictionary> dictionary;
    std::vector<u8> mirror;
    size_t half{};
    size_t current{};
    size_t position{};
    u64 produced{};     ///< frame bytes the host has seen (the game's decodedSize)
    u64 guest_end{};    ///< guest address after the last bytes the host wrote
};

std::mutex streams_mutex;
std::map<u64, std::shared_ptr<Stream>> streams;

/// Kept alive by the caller: another thread may drop idle streams from the map meanwhile.
std::shared_ptr<Stream> GetStream(u64 dctx) {
    std::scoped_lock lock{streams_mutex};
    if (const auto it = streams.find(dctx); it != streams.end()) {
        return it->second;
    }
    if (streams.size() >= 64) {
        // Only streams no thread holds and that are not in a host frame.
        std::erase_if(streams, [](const auto& entry) {
            return entry.second.use_count() == 1 && entry.second->mode != Stream::Mode::Host;
        });
    }
    return streams.emplace(dctx, std::make_shared<Stream>()).first->second;
}

void GuestFrame(Stream& stream, const char* reason) {
    stream.mode = Stream::Mode::Guest;
    const u64 count = stream_guest_frames.fetch_add(1, std::memory_order_relaxed);
    if (count < 8 || (count & (count - 1)) == 0) {
        LOG_INFO(Core, "Ocarina zstd stream: the game decodes a frame ({}), {} so far", reason,
                 count + 1);
    }
}

/// The guest bytes in front of dst that are history at the start of a frame (its dictionary).
/// continuity_done: the game already moved the window to dst (the block path); otherwise
/// ZSTD_checkContinuity is still to run (the continue path).
bool FrameDictionary(Core::Memory::Memory& memory, u64 dctx, u64 dst, bool continuity_done,
                     u64& start, u64& size) {
    const u64 previous_end = memory.Read64(dctx + PreviousDstEnd);
    const u64 prefix_start = memory.Read64(dctx + PrefixStart);
    const u64 virtual_start = memory.Read64(dctx + VirtualStart);
    const u64 dict_end = memory.Read64(dctx + DictEnd);
    const u64 ext_size = prefix_start - virtual_start;
    if (!continuity_done && dst != previous_end) {
        // ZSTD_checkContinuity will make [prefixStart, previousDstEnd) the window's old segment.
        if (ext_size != 0) {
            return false;
        }
        start = prefix_start;
        size = previous_end - prefix_start;
        return size <= MaxDictionarySize;
    }
    const u64 prefix_size = dst - prefix_start;
    if (ext_size != 0 && prefix_size != 0) {
        return false;
    }
    start = ext_size != 0 ? dict_end - ext_size : prefix_start;
    size = ext_size != 0 ? ext_size : prefix_size;
    return size <= MaxDictionarySize;
}

/// The frame's header (size and bytes as the game keeps them in the dctx): it identifies the frame.
std::array<u8, HeaderBufferSize + 8> FrameHeader(Core::Memory::Memory& memory, u64 dctx) {
    std::array<u8, HeaderBufferSize + 8> header{};
    const u64 size = memory.Read64(dctx + HeaderSize);
    std::memcpy(header.data(), &size, sizeof(size));
    memory.ReadBlock(dctx + HeaderBuffer, header.data() + 8, HeaderBufferSize);
    return header;
}

/// The last `size` bytes of the game's window that ends at history_end: the prefix
/// [prefixStart, history_end), then the old segment ending at dictEnd.
bool ReadHistoryTail(Core::Memory::Memory& memory, u64 dctx, u64 history_end, u64 size, u8* out) {
    const u64 prefix_start = memory.Read64(dctx + PrefixStart);
    const u64 virtual_start = memory.Read64(dctx + VirtualStart);
    const u64 dict_end = memory.Read64(dctx + DictEnd);
    if (history_end < prefix_start) {
        return false;
    }
    const u64 prefix_size = history_end - prefix_start;
    const u64 ext_size = prefix_start - virtual_start;
    const u64 from_prefix = std::min(size, prefix_size);
    const u64 from_ext = size - from_prefix;
    if (from_ext > ext_size) {
        return false;
    }
    if (from_ext != 0 && !memory.ReadBlock(dict_end - from_ext, out, from_ext)) {
        return false;
    }
    return from_prefix == 0 ||
           memory.ReadBlock(history_end - from_prefix, out + from_ext, from_prefix);
}

bool StartFrame(Core::Memory::Memory& memory, Stream& stream, u64 dctx, u64 dst,
                bool continuity_done) {
    const u64 window = memory.Read64(dctx + WindowSize);
    if (window == 0 || window > MaxWindowSize) {
        GuestFrame(stream, "window size");
        return false;
    }
    u64 dict_start{};
    u64 dict_size{};
    if (!FrameDictionary(memory, dctx, dst, continuity_done, dict_start, dict_size)) {
        GuestFrame(stream, "window layout");
        return false;
    }
    std::shared_ptr<const Dictionary> dictionary;
    if (dict_size != 0) {
        dictionary = GetDictionary(memory, 0, dict_start, dict_size);
        if (!dictionary) {
            GuestFrame(stream, "dictionary");
            return false;
        }
        const u32 frame_dict = memory.Read32(dctx + FrameDictId);
        if (frame_dict != 0 && frame_dict != ZSTD_getDictID_fromDDict(dictionary->ddict.get())) {
            GuestFrame(stream, "dictionary id");
            return false;
        }
    }
    if (!stream.context) {
        stream.context.reset(ZSTD_createDCtx());
        if (!stream.context) {
            GuestFrame(stream, "no context");
            return false;
        }
    }
    const size_t result = dictionary ? ZSTD_decompressBegin_usingDDict(stream.context.get(),
                                                                       dictionary->ddict.get())
                                     : ZSTD_decompressBegin(stream.context.get());
    if (ZSTD_isError(result)) {
        GuestFrame(stream, ZSTD_getErrorName(result));
        return false;
    }
    const size_t half = static_cast<size_t>(std::max<u64>(window, 1ULL << 20) + BlockSizeMax);
    if (stream.mirror.size() < half * 2) {
        stream.mirror.clear();
        stream.mirror.resize(half * 2);
    }
    stream.half = stream.mirror.size() / 2;
    stream.dictionary = std::move(dictionary);
    stream.current = 0;
    stream.position = 0;
    stream.produced = 0;
    stream.guest_end = dst;
    stream.header = FrameHeader(memory, dctx);
    stream.mode = Stream::Mode::Host;
    return true;
}

u8* Space(Stream& stream, size_t size) {
    if (stream.position + size > stream.half) {
        stream.current ^= 1;
        stream.position = 0;
    }
    return stream.mirror.data() + stream.current * stream.half + stream.position;
}

/// Takes the bytes the game wrote itself (raw and RLE blocks) since the host's last block into the
/// host's history, from the game's window ending at history_end.
bool CatchUp(Core::Memory::Memory& memory, Stream& stream, u64 dctx, u64 history_end, u64 gap) {
    thread_local std::vector<u8> bytes;
    bytes.resize(gap);
    if (!ReadHistoryTail(memory, dctx, history_end, gap, bytes.data())) {
        return false;
    }
    for (u64 done = 0; done < gap;) {
        const size_t chunk = static_cast<size_t>(std::min<u64>(gap - done, BlockSizeMax));
        u8* const out = Space(stream, chunk);
        std::memcpy(out, bytes.data() + done, chunk);
        if (ZSTD_isError(ZSTD_insertBlock(stream.context.get(), out, chunk))) {
            return false;
        }
        stream.position += chunk;
        stream.produced += chunk;
        done += chunk;
    }
    stream.guest_end = history_end;
    return true;
}

/// Whether the game's dctx still decodes the frame the host twin has: same header, and the bytes
/// the host wrote last are still where it wrote them.
bool SameFrame(Core::Memory::Memory& memory, Stream& stream, u64 dctx) {
    if (FrameHeader(memory, dctx) != stream.header) {
        return false;
    }
    const size_t tail = static_cast<size_t>(std::min<u64>(stream.position, TailCheck));
    if (tail == 0) {
        return true;
    }
    std::array<u8, TailCheck> guest{};
    if (!memory.ReadBlock(stream.guest_end - tail, guest.data(), tail)) {
        return false;
    }
    const u8* const host =
        stream.mirror.data() + stream.current * stream.half + stream.position - tail;
    return std::memcmp(guest.data(), host, tail) == 0;
}

enum class BlockResult : u8 { Guest, Done };

/// One compressed block of the game's frame at dctx. Guest: the game's code decodes it.
BlockResult DecodeBlock(Core::Memory::Memory& memory, u64 dctx, u64 dst, u64 capacity, u64 src,
                        u64 src_size, bool continuity_done, u64& result) {
    const std::shared_ptr<Stream> stream_owner = GetStream(dctx);
    Stream& stream = *stream_owner;
    const u64 decoded = memory.Read64(dctx + DecodedSize);
    const bool last_block = memory.Read32(dctx + Stage) == StageDecompressLastBlock;
    if (stream.mode == Stream::Mode::Host && decoded != 0 &&
        (decoded < stream.produced || !SameFrame(memory, stream, dctx))) {
        // Another frame began with stored blocks; its tables are still the game's own.
        stream.mode = Stream::Mode::Idle;
    }
    if (decoded == 0) {
        if (!StartFrame(memory, stream, dctx, dst, continuity_done)) {
            return BlockResult::Guest;
        }
    } else if (stream.mode != Stream::Mode::Host) {
        if (stream.mode == Stream::Mode::Idle) {
            GuestFrame(stream, "frame began before the hook saw it");
        }
        return BlockResult::Guest;
    } else if (decoded > stream.produced) {
        // The game's code wrote stored blocks since: they end where its window ends.
        const u64 history_end = continuity_done ? dst : memory.Read64(dctx + PreviousDstEnd);
        if (!CatchUp(memory, stream, dctx, history_end, decoded - stream.produced)) {
            // The game's tables are stale for this frame: it must not decode the rest itself.
            LOG_WARNING(Core, "Ocarina zstd stream: stored bytes of a host frame not found");
            stream.mode = Stream::Mode::Idle;
            result = ErrorCorruption;
            return BlockResult::Done;
        }
    }
    thread_local std::vector<u8> input;
    if (src_size >= BlockSizeMax) {
        stream.mode = Stream::Mode::Idle;
        result = ErrorCorruption;
        return BlockResult::Done;
    }
    input.resize(src_size);
    if (!memory.ReadBlock(src, input.data(), src_size)) {
        stream.mode = Stream::Mode::Idle;
        result = ErrorCorruption;
        return BlockResult::Done;
    }
    u8* const out = Space(stream, BlockSizeMax);
    const size_t size =
        ZSTD_decompressBlock(stream.context.get(), out, BlockSizeMax, input.data(), src_size);
    if (ZSTD_isError(size)) {
        LOG_WARNING(Core, "Ocarina zstd stream: block of {} bytes failed: {}", src_size,
                    ZSTD_getErrorName(size));
        stream.mode = Stream::Mode::Idle;
        result = ErrorCorruption;
        return BlockResult::Done;
    }
    if (size > capacity) {
        stream.mode = Stream::Mode::Idle;
        result = ErrorDstSizeTooSmall;
        return BlockResult::Done;
    }
    if (size != 0 && !memory.WriteBlock(dst, out, size)) {
        stream.mode = Stream::Mode::Idle;
        result = ErrorCorruption;
        return BlockResult::Done;
    }
    stream.position += size;
    stream.produced += size;
    stream.guest_end = dst + size;
    if (last_block) {
        stream.mode = Stream::Mode::Idle;
    }
    result = size;
    const u64 blocks = stream_blocks.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 bytes = stream_bytes.fetch_add(size, std::memory_order_relaxed) + size;
    if (blocks == 1 || (blocks & (blocks - 1)) == 0) {
        LOG_INFO(Core, "Ocarina zstd stream: {} blocks decoded on the host, {} MiB, {} frames "
                       "left to the game",
                 blocks, bytes >> 20, stream_guest_frames.load(std::memory_order_relaxed));
    }
    return BlockResult::Done;
}

/// ZSTD_decompressContinue for a compressed block, with the bookkeeping zstd 1.4 does after it.
BlockResult DecodeContinue(Core::Memory::Memory& memory, const Kernel::Svc::ThreadContext& ctx,
                           u64& result) {
    const u64 dctx = ctx.r[0];
    const u64 dst = ctx.r[1];
    const u64 capacity = ctx.r[2];
    const u64 src = ctx.r[3];
    const u64 src_size = ctx.r[4];
    const u32 stage = memory.Read32(dctx + Stage);
    if ((stage != StageDecompressBlock && stage != StageDecompressLastBlock) ||
        memory.Read32(dctx + BlockType) != BlockTypeCompressed ||
        src_size != memory.Read64(dctx + Expected)) {
        return BlockResult::Guest;
    }
    if (DecodeBlock(memory, dctx, dst, capacity, src, src_size, false, result) ==
        BlockResult::Guest) {
        return BlockResult::Guest;
    }
    // ZSTD_checkContinuity, as the game's code would have done before the block.
    const u64 previous_end = memory.Read64(dctx + PreviousDstEnd);
    if (capacity != 0 && dst != previous_end) {
        const u64 prefix_start = memory.Read64(dctx + PrefixStart);
        memory.Write64(dctx + DictEnd, previous_end);
        memory.Write64(dctx + VirtualStart, dst - (previous_end - prefix_start));
        memory.Write64(dctx + PrefixStart, dst);
        memory.Write64(dctx + PreviousDstEnd, dst);
    }
    if (ZSTD_isError(result)) {
        return BlockResult::Done;
    }
    const u64 size = result;
    const u64 decoded = memory.Read64(dctx + DecodedSize) + size;
    memory.Write64(dctx + DecodedSize, decoded);
    const bool checksum = memory.Read32(dctx + ChecksumFlag) != 0;
    if (checksum && size != 0) {
        XxhGuestState state{};
        memory.ReadBlock(dctx + XxhState, &state, sizeof(state));
        thread_local std::vector<u8> output;
        output.resize(size);
        memory.ReadBlock(dst, output.data(), size);
        XxhUpdate(state, output.data(), size);
        memory.WriteBlock(dctx + XxhState, &state, sizeof(state));
    }
    if (stage == StageDecompressLastBlock) {
        const u64 content_size = memory.Read64(dctx + FrameContentSize);
        if (content_size != ~0ULL && decoded != content_size) {
            result = ErrorCorruption;
            return BlockResult::Done;
        }
        memory.Write64(dctx + Expected, checksum ? 4 : 0);
        memory.Write32(dctx + Stage, checksum ? StageCheckChecksum : 0);
    } else {
        memory.Write32(dctx + Stage, StageDecodeBlockHeader);
        memory.Write64(dctx + Expected, 3);
        memory.Write64(dctx + PreviousDstEnd, dst + size);
    }
    return BlockResult::Done;
}

bool CodeMatches(std::span<const u8> image, u64 offset, std::span<const u32> code) {
    if (image.size() < offset + code.size_bytes()) {
        return false;
    }
    return std::memcmp(image.data() + offset, code.data(), code.size_bytes()) == 0;
}

} // Anonymous namespace

void PatchModule(u64 program_id, std::span<const u8> build_id, std::string_view name,
                 std::span<u8> image, u64 load_base) {
    if (program_id != ProgramId || name != "main") {
        return;
    }
    if (Common::HexToString(build_id.first(20)) != MainBuildId) {
        LOG_INFO(Core, "Ocarina zstd: other build of main, the game's decoder stays");
        return;
    }
    if (!CodeMatches(image, MultiFrameOffset, MultiFramePrologue)) {
        LOG_WARNING(Core, "Ocarina zstd: unexpected code at main+{:#x}, the game's decoder stays",
                    MultiFrameOffset);
        return;
    }
    const u32 svc = 0xd4000001 | (SvcNumber << 5);
    std::memcpy(image.data() + MultiFrameOffset, &svc, sizeof(svc));
    module_base.store(load_base, std::memory_order_relaxed);
    LOG_INFO(Core, "Ocarina zstd: ZSTD_decompressMultiFrame at {:#x} decompresses on the host",
             load_base + MultiFrameOffset);
    if (!CodeMatches(image, BlockOffset, BlockCode) ||
        !CodeMatches(image, ContinueOffset, ContinueCode)) {
        LOG_WARNING(Core, "Ocarina zstd: unexpected streaming code, the game decodes its streams");
        return;
    }
    std::memcpy(image.data() + BlockOffset, &svc, sizeof(svc));
    std::memcpy(image.data() + ContinueOffset, &svc, sizeof(svc));
    streaming_hooked.store(true, std::memory_order_relaxed);
    LOG_INFO(Core, "Ocarina zstd: streamed blocks decode on the host");
}

void Handle(Core::System& system) {
    auto& kernel = system.Kernel();
    auto* const process = Kernel::GetCurrentProcessPointer(kernel);
    if (process == nullptr) {
        return;
    }
    auto* const arm = process->GetArmInterface(kernel.CurrentPhysicalCoreIndex());
    if (arm == nullptr) {
        return;
    }
    auto& memory = process->GetMemory();
    Kernel::Svc::ThreadContext ctx{};
    arm->GetContext(ctx);
    // The JIT reports the instruction after the SVC.
    const u64 offset = ctx.pc - 4 - module_base.load(std::memory_order_relaxed);
    u64 result = 0;
    bool done = false;
    if (offset == MultiFrameOffset) {
        // The replaced pacibsp does nothing in the JIT: leaving needs no fix-up.
        done = DecompressFrames(memory, ctx, result);
    } else if (offset == BlockOffset && streaming_hooked.load(std::memory_order_relaxed)) {
        done = DecodeBlock(memory, ctx.r[0], ctx.r[1], ctx.r[2], ctx.r[3], ctx.r[4], true,
                           result) == BlockResult::Done;
        if (!done) {
            ctx.r[8] = ctx.r[4] >> 17; // lsr x8, x4, #17
            arm->SetContext(ctx);
            return;
        }
    } else if (offset == ContinueOffset && streaming_hooked.load(std::memory_order_relaxed)) {
        done = DecodeContinue(memory, ctx, result) == BlockResult::Done;
        if (!done) {
            ctx.r[8] = memory.Read64(ctx.r[0] + Expected); // ldr x8, [x0, #0x7068]
            arm->SetContext(ctx);
            return;
        }
    } else {
        LOG_ERROR(Core, "Ocarina zstd: SVC {:#x} from unexpected address {:#x}", SvcNumber,
                  ctx.pc - 4);
        return;
    }
    if (!done) {
        return; // the game's code continues after the SVC
    }
    ctx.r[0] = result;
    ctx.pc = ctx.lr;
    arm->SetContext(ctx);
}

} // namespace Core::OotZstd
