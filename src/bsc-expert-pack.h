#pragma once

// BSC thesis: Expert weight pack file format
//
// A sidecar file containing MoE expert weight slices laid out for direct
// io_uring + O_DIRECT reads. Eliminates the staging-buffer + memcpy overhead
// of reading unaligned slices from the original GGUF file.
//
// File layout:
//
//   [HEADER, 4096 bytes, zero-padded]
//     magic        = "BSCEXP01"
//     header_size  = 4096
//     n_layers, n_proj, n_expert
//     expert_bytes = exact size of one expert slice (e.g. 4,406,400 for MXFP4)
//     slot_bytes   = expert_bytes rounded up to 512 (e.g. 4,406,784)
//     ...
//
//   [DATA SECTION, starts at offset 4096]
//     slot 0: slot_bytes bytes
//     slot 1: slot_bytes bytes
//     ...
//     slot N-1: slot_bytes bytes
//
// Slot indexing:
//   slot_idx = layer * n_proj * n_expert + proj_idx * n_expert + expert_id
//   pack_off = header_size + slot_idx * slot_bytes
//
// Both header_size and slot_bytes are multiples of 512, so every pack_off
// is 512-aligned. This is the property that lets us read directly into a
// 512-aligned compact buffer with O_DIRECT — no staging buffer, no memcpy.
//
// Projection ordering convention (MUST match runtime loader):
//   With gate (n_proj == 3):  proj 0 = down, 1 = gate, 2 = up
//   Without gate (n_proj == 2): proj 0 = down, 1 = up
//
// The mul_mat_id operation reads (ne01 * nb01) bytes per expert, which equals
// expert_bytes — strictly less than slot_bytes. The trailing padding within
// each slot (slot_bytes - expert_bytes, at most 511 bytes) is never read.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Format constants ---

#define BSC_EXPERT_PACK_MAGIC        "BSCEXP01"   // 8 bytes, identifies format + version
#define BSC_EXPERT_PACK_MAGIC_LEN    8
#define BSC_EXPERT_PACK_HEADER_SIZE  4096         // multiple of 512 — keeps slots aligned
#define BSC_EXPERT_PACK_ALIGN        512          // O_DIRECT alignment requirement

// --- Header struct ---
//
// Stored at offset 0 of the pack file. Total in-file size is
// BSC_EXPERT_PACK_HEADER_SIZE bytes (zero-padded after the fields below).

struct bsc_expert_pack_header {
    char     magic[BSC_EXPERT_PACK_MAGIC_LEN];  // "BSCEXP01"
    uint32_t header_size;       // == BSC_EXPERT_PACK_HEADER_SIZE
    uint32_t n_layers;          // number of transformer layers
    uint32_t n_proj;            // 2 (down,up) or 3 (down,gate,up)
    uint32_t n_expert;          // total experts per layer (e.g. 32)
    uint64_t expert_bytes;      // exact size of one expert slice in bytes
    uint64_t slot_bytes;        // 512-aligned slot size = ceil(expert_bytes/512)*512
    uint64_t total_data_bytes;  // n_layers * n_proj * n_expert * slot_bytes
    uint64_t source_gguf_size;  // sanity check vs source file size
    // Reserved bytes follow, zero-filled, until BSC_EXPERT_PACK_HEADER_SIZE.
};

// --- Helpers (inline so both preprocessor and runtime use the same code) ---

// Round size up to the next multiple of BSC_EXPERT_PACK_ALIGN.
static inline uint64_t bsc_expert_pack_align_up(uint64_t v) {
    return (v + (BSC_EXPERT_PACK_ALIGN - 1)) & ~(uint64_t)(BSC_EXPERT_PACK_ALIGN - 1);
}

// Compute the slot index for a given (layer, proj_idx, expert_id).
// Both preprocessor and runtime MUST use this function — divergence here
// produces silent data corruption.
static inline uint64_t bsc_expert_pack_slot_index(
        uint32_t layer,
        uint32_t proj_idx,
        uint32_t expert_id,
        uint32_t n_proj,
        uint32_t n_expert) {
    return (uint64_t)layer * (uint64_t)n_proj * (uint64_t)n_expert
         + (uint64_t)proj_idx * (uint64_t)n_expert
         + (uint64_t)expert_id;
}

// Compute the file offset for a given slot index.
// Always 512-aligned because header_size and slot_bytes are both 512-aligned.
static inline uint64_t bsc_expert_pack_slot_offset(
        uint64_t slot_idx,
        uint64_t slot_bytes) {
    return (uint64_t)BSC_EXPERT_PACK_HEADER_SIZE + slot_idx * slot_bytes;
}

// Total number of slots given config.
static inline uint64_t bsc_expert_pack_total_slots(
        uint32_t n_layers,
        uint32_t n_proj,
        uint32_t n_expert) {
    return (uint64_t)n_layers * (uint64_t)n_proj * (uint64_t)n_expert;
}

// Compile-time invariants — catch alignment bugs at build time.
#ifdef __cplusplus
static_assert(BSC_EXPERT_PACK_HEADER_SIZE % BSC_EXPERT_PACK_ALIGN == 0,
              "header_size must be 512-aligned");
static_assert(sizeof(struct bsc_expert_pack_header) <= BSC_EXPERT_PACK_HEADER_SIZE,
              "header struct must fit within reserved header_size");
#else
_Static_assert(BSC_EXPERT_PACK_HEADER_SIZE % BSC_EXPERT_PACK_ALIGN == 0,
               "header_size must be 512-aligned");
_Static_assert(sizeof(struct bsc_expert_pack_header) <= BSC_EXPERT_PACK_HEADER_SIZE,
               "header struct must fit within reserved header_size");
#endif

#ifdef __cplusplus
}
#endif
