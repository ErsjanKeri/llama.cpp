#ifndef TENSOR_TRACE_H
#define TENSOR_TRACE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Phase enumeration for tracking prompt vs generation
enum TensorTracePhase {
    TRACE_PHASE_PROMPT = 0,
    TRACE_PHASE_GENERATE = 1,
};

// 64-byte fixed-size log entry (cache-line aligned)
struct TensorAccessLog {
    // === Timestamp (8 bytes) ===
    uint64_t timestamp_ns;        // Nanoseconds since trace start

    // === Execution Context (16 bytes) ===
    uint32_t token_id;            // Which token being processed
    uint16_t layer_id;            // Which transformer layer (0-based, 65535=N/A)
    uint16_t thread_id;           // CPU thread ID
    uint8_t  operation_type;      // Enum: MUL_MAT, ADD, ROPE, etc.
    uint8_t  phase;               // Enum: PROMPT, GENERATE
    uint16_t padding1;            // Alignment
    uint32_t padding1b;           // Alignment

    // === Tensor Identification (20 bytes) ===
    uint32_t tensor_idx;          // Index into tensor name table
    uint64_t tensor_ptr;          // Virtual address of tensor->data
    uint64_t file_offset;         // Offset in GGUF file (0 = not in file/intermediate)
    uint32_t size_bytes;          // Size of tensor in bytes

    // === Attention-Specific (4 bytes) ===
    uint8_t  attention_head;      // Which attention head (0-127, or 255=N/A)
    uint8_t  qkv_type;            // Enum: Q, K, V, O, or N/A
    uint16_t padding2;            // Alignment

    // === MoE-Specific (12 bytes) ===
    uint8_t  expert_id;           // Which expert (0-255, or 255=N/A)
    uint8_t  expert_rank;         // Routing rank (0=top, 1=second, etc.)
    uint16_t routing_score;       // Quantized routing score (0-65535)
    uint32_t padding3;            // Alignment
    uint32_t padding4;            // Final padding to reach 64 bytes

    // Total: 64 bytes (verified at compile time)
} __attribute__((packed));

// Static assertion to ensure struct is exactly 64 bytes
_Static_assert(sizeof(struct TensorAccessLog) == 64,
               "TensorAccessLog must be exactly 64 bytes");

// === API Functions ===

// Initialize tracing system
// log_path: Path to binary log file (e.g., "/dev/shm/tensor_trace.bin")
// capacity_bytes: Maximum log file size (e.g., 2GB)
void tensor_trace_init(const char* log_path, size_t capacity_bytes);

// Log a tensor access (fast path, no error checking)
// entry: Pre-filled log entry to write
void tensor_trace_log(const struct TensorAccessLog* entry);

// Flush all thread-local buffers and close log file
void tensor_trace_shutdown(void);

// Register tensor metadata during model load
// name: Tensor name (e.g., "blk.5.attn_q.weight")
// data_ptr: Virtual address of tensor->data
// file_offset: Offset in GGUF file (0 for intermediates)
// size_bytes: Size of tensor
void tensor_trace_register_tensor(
    const char* name,
    void* data_ptr,
    uint64_t file_offset,
    uint64_t size_bytes
);

// Get current timestamp in nanoseconds (for logging)
uint64_t tensor_trace_get_timestamp_ns(void);

// Get current thread ID (Linux-specific)
uint16_t tensor_trace_get_thread_id(void);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_TRACE_H
