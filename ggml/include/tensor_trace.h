#ifndef TENSOR_TRACE_H
#define TENSOR_TRACE_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Phase enumeration for tracking prompt vs generation
enum TensorTracePhase {
    TRACE_PHASE_PROMPT = 0,
    TRACE_PHASE_GENERATE = 1,
};

// Trace mode: controls which operations are logged
enum TensorTraceMode {
    TRACE_MODE_OFF     = 0,  // no tracing
    TRACE_MODE_EXPERTS = 1,  // only MoE ops (MUL_MAT_ID, ADD_ID with expert_ids)
    TRACE_MODE_ALL     = 2,  // all operations (original behavior)
};

// Memory source enumeration (disk-backed GGUF vs runtime buffers)
enum MemorySource {
    MEMORY_SOURCE_DISK = 0,   // GGUF file (mmap'd model parameters)
    MEMORY_SOURCE_BUFFER = 1  // Runtime buffers (KV cache, scratch, activations)
};

// === Source Tensor Information (160 bytes per source) ===
// Embedded in TensorAccessLog for each source tensor
struct SourceTensorInfo {
    char name[128];               // Tensor name - EXPANDED for full names + suffixes (e.g., "blk.5.attn_q.weight (view)")
    uint64_t tensor_ptr;          // Virtual address of tensor->data
    uint32_t size_bytes;          // Tensor size in bytes
    uint16_t layer_id;            // Extracted layer ID (65535=N/A)
    uint8_t  memory_source;       // MEMORY_SOURCE_DISK or MEMORY_SOURCE_BUFFER
    uint8_t  padding1;            // Alignment
    uint64_t disk_offset_or_buffer_id;  // If DISK: offset in GGUF, if BUFFER: buffer ID
    uint32_t tensor_idx;          // Path B: index in registry (UINT32_MAX if not found)
    uint8_t  padding2[4];         // Alignment to 160 bytes
} __attribute__((packed));

// Static assertion for source info size (compatible with C and C++)
#ifdef __cplusplus
static_assert(sizeof(struct SourceTensorInfo) == 160, "SourceTensorInfo must be exactly 160 bytes");
#else
_Static_assert(sizeof(struct SourceTensorInfo) == 160, "SourceTensorInfo must be exactly 160 bytes");
#endif

// === Main Log Entry: ONE entry per operation (1024 bytes, 16 cache lines) ===
// Contains operation metadata + destination + ALL source tensors
struct TensorAccessLog {
    // === Operation Metadata (24 bytes) ===
    uint64_t timestamp_ns;        // Nanoseconds since trace start
    uint32_t token_id;            // Which token being processed
    uint16_t layer_id;            // Operation's layer (extracted from dst or src, 65535=N/A)
    uint16_t thread_id;           // CPU thread ID
    uint8_t  operation_type;      // ggml_op enum value (GGML_OP_MUL_MAT, GGML_OP_ADD, etc.)
    uint8_t  phase;               // TRACE_PHASE_PROMPT or TRACE_PHASE_GENERATE
    uint8_t  num_sources;         // Number of source tensors (0-4, most ops have 1-3)
    uint8_t  padding1[5];         // Alignment

    // === Destination Tensor (128 bytes) ===
    char dst_name[128];           // What's being computed (output tensor name) - EXPANDED for full names

    // === Source Tensors (640 bytes = 4 × 160 bytes) ===
    // sources[0]: Primary source (usually weight matrix)
    // sources[1]: Secondary source (usually input activations)
    // sources[2]: Tertiary source (optional, e.g., position indices for ROPE)
    // sources[3]: Quaternary source (rarely used, e.g., frequency factors)
    struct SourceTensorInfo sources[4];

    // === MoE Expert IDs (65 bytes) ===
    // For MUL_MAT_ID operations: which expert IDs were selected (0-31)
    // For other operations: unused (all zeros)
    int32_t expert_ids[16];       // Up to 16 expert IDs (64 bytes)
    uint8_t num_experts;          // Number of valid expert IDs (1 byte)

    // === Padding (167 bytes) ===
    uint8_t  padding2[167];       // Padding to 1024 bytes for cache alignment

    // Total: 24 + 128 + 640 + 64 + 1 + 167 = 1024 bytes (verified at compile time)
} __attribute__((packed));

// Static assertion to ensure struct is exactly 1024 bytes (compatible with C and C++)
#ifdef __cplusplus
static_assert(sizeof(struct TensorAccessLog) == 1024, "TensorAccessLog must be exactly 1024 bytes");
#else
_Static_assert(sizeof(struct TensorAccessLog) == 1024, "TensorAccessLog must be exactly 1024 bytes");
#endif

// === Buffer Lifecycle Event (Phase 1.3) ===
// Logs buffer allocations/deallocations for memory occupancy analysis
// 128 bytes (2 cache lines) for efficient logging
struct BufferEvent {
    uint64_t timestamp_ns;        // Nanoseconds since trace start (for correlation with tensor ops)
    uint8_t  event_type;          // 0=ALLOC, 1=DEALLOC
    uint8_t  buffer_usage;        // GGML_BACKEND_BUFFER_USAGE_WEIGHTS, COMPUTE, or ANY
    uint16_t layer_id;            // Associated layer (65535=N/A for global buffers)
    uint64_t buffer_id;           // Unique buffer identifier (pointer value)
    uint64_t buffer_ptr;          // Virtual address
    uint64_t size_bytes;          // Buffer size in bytes
    char buffer_name[64];         // Name (e.g., "ModelWeights_file0", "KVCache_CPU")
    char backend_type[16];        // Backend type (e.g., "CPU", "Metal", "CUDA")
    uint8_t  _padding[12];        // Padding to reach 128 bytes (8+1+1+2+8+8+8+64+16+12=128)
} __attribute__((packed));

// Static assertion for BufferEvent size (compatible with C and C++)
#ifdef __cplusplus
static_assert(sizeof(struct BufferEvent) == 128, "BufferEvent must be exactly 128 bytes");
#else
_Static_assert(sizeof(struct BufferEvent) == 128, "BufferEvent must be exactly 128 bytes");
#endif

// Buffer event types
enum BufferEventType {
    BUFFER_EVENT_ALLOC = 0,
    BUFFER_EVENT_DEALLOC = 1
};

// === Helper Functions ===

// Extract layer ID from tensor name (e.g., "blk.5.attn_q.weight" → 5)
// Returns 65535 (UINT16_MAX) for non-layer tensors (embeddings, output, etc.)
static inline uint16_t tensor_trace_extract_layer_id(const char* name) {
    if (name && strncmp(name, "blk.", 4) == 0) {
        int layer = -1;
        if (sscanf(name + 4, "%d", &layer) == 1 && layer >= 0 && layer < 65535) {
            return (uint16_t)layer;
        }
    }
    return 65535;  // Not a layer tensor
}

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

// Lookup tensor index by data pointer (Path B - efficient lookup)
// Returns tensor_idx if found, UINT32_MAX if not found
uint32_t tensor_trace_lookup_idx(void* data_ptr);

// Dump registry table to file (for validation and analysis)
// Outputs CSV: tensor_idx,tensor_name,data_ptr,file_offset,size_bytes,layer_id
void tensor_trace_dump_registry(const char* output_path);

// === NEW: Generic Operation Logging (Phase 1.1) ===

// Forward declaration for ggml types
struct ggml_tensor;

// Log a complete operation (all source tensors)
// Called from ggml_compute_forward() dispatcher BEFORE switch statement
// ith: thread index (0 for first thread, 1+ for worker threads)
void tensor_trace_log_operation(
    const struct ggml_tensor * dst,
    int ith);

// Set trace mode (OFF, EXPERTS, ALL)
// Called before tensor_trace_init or at any time to change filtering
void tensor_trace_set_mode(uint8_t mode);

// Set current inference phase (PROMPT or GENERATE)
// Called from application code (e.g., llama-completion) to track phase
void tensor_trace_set_phase(uint8_t phase);

// Set current token ID being processed
// Called from application code to track which generated token is being processed
// During PROMPT: typically 0, During GENERATE: 0, 1, 2, ... for each output token
void tensor_trace_set_token_id(uint32_t token_id);

// Helper functions for memory source detection (implemented in Phase 1.2)
enum MemorySource tensor_trace_detect_memory_source(const struct ggml_tensor * tensor);
uint64_t tensor_trace_get_disk_offset(const struct ggml_tensor * tensor);
uint64_t tensor_trace_get_buffer_id(const struct ggml_tensor * tensor);

// Register tensor disk offset (Phase 1.2)
// Called from llama during model load to register GGUF file offsets
void tensor_trace_register_disk_offset(const char * name, uint64_t offset);

// === Buffer Tracking API (Phase 1.3) ===

// Log buffer allocation event
// buffer_id: Unique buffer identifier (typically buffer pointer)
// buffer_ptr: Virtual address of buffer
// size_bytes: Size of buffer in bytes
// buffer_name: Descriptive name (e.g., "ModelWeights_file0")
// backend_type: Backend type name (e.g., "CPU", "Metal")
// buffer_usage: GGML_BACKEND_BUFFER_USAGE enum value
// layer_id: Associated layer ID (65535 for global buffers)
void tensor_trace_log_buffer_alloc(
    uint64_t buffer_id,
    void * buffer_ptr,
    size_t size_bytes,
    const char * buffer_name,
    const char * backend_type,
    uint8_t buffer_usage,
    uint16_t layer_id);

// Log buffer deallocation event
// buffer_id: Unique buffer identifier (same as used in alloc)
void tensor_trace_log_buffer_dealloc(uint64_t buffer_id);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_TRACE_H
