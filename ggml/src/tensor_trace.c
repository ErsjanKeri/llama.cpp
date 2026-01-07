#include "tensor_trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>

#ifdef __linux__
#include <sys/syscall.h>
#endif

// Forward declarations for buffer tracking (Phase 1.3)
static void tensor_trace_init_buffer_stats(void);
static void tensor_trace_shutdown_buffer_stats(void);

// === Global State ===

// Memory-mapped log buffer
static void* g_log_buffer = NULL;
static size_t g_log_capacity = 0;      // Total capacity in bytes
static size_t g_log_offset = 0;        // Current write offset (not thread-safe, needs atomics later)
static int g_log_fd = -1;              // File descriptor for log file
static uint64_t g_trace_start_ns = 0;  // Trace start time (for relative timestamps)

// Thread-local buffer for batching writes (avoid contention)
#define THREAD_LOCAL_BUFFER_SIZE 512  // 512 entries = 128KB per thread (256 bytes each)
static __thread struct TensorAccessLog g_thread_local_buffer[THREAD_LOCAL_BUFFER_SIZE];
static __thread size_t g_thread_local_offset = 0;

// Global execution context (Phase 1.1+)
static int g_trace_enabled = 1;
static uint8_t g_current_phase = TRACE_PHASE_PROMPT;
static uint32_t g_current_token_id = 0;

// === Tensor Registration Table (Path B) ===

#define MAX_REGISTERED_TENSORS 1024  // TinyLlama has ~201 tensors, use 1024 for safety

struct TensorRegistryEntry {
    void* data_ptr;          // Memory address (key for lookup)
    char name[64];           // Tensor name (e.g., "blk.5.attn_q.weight")
    uint64_t file_offset;    // Offset in GGUF file
    uint64_t size_bytes;     // Size in bytes
    uint16_t layer_id;       // Extracted layer ID
    uint32_t tensor_idx;     // Index in registry (for trace logs)
};

static struct TensorRegistryEntry g_tensor_registry[MAX_REGISTERED_TENSORS];
static uint32_t g_registry_count = 0;  // Number of registered tensors

// === Helper Functions ===

// Get current timestamp in nanoseconds (relative to trace start)
uint64_t tensor_trace_get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    return now - g_trace_start_ns;  // Return relative time since trace start
}

// Get current thread ID (Linux-specific)
uint16_t tensor_trace_get_thread_id(void) {
#ifdef __linux__
    return (uint16_t)syscall(SYS_gettid);
#else
    // Fallback for non-Linux (not perfect, but works for debugging)
    return (uint16_t)((uintptr_t)pthread_self() & 0xFFFF);
#endif
}

// === Core API Implementation ===

void tensor_trace_init(const char* log_path, size_t capacity_bytes) {
    if (g_log_buffer != NULL) {
        fprintf(stderr, "[TENSOR_TRACE] Error: Already initialized\n");
        return;
    }

    // Record trace start time
    g_trace_start_ns = tensor_trace_get_timestamp_ns();

    // Open/create log file
    g_log_fd = open(log_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (g_log_fd < 0) {
        fprintf(stderr, "[TENSOR_TRACE] Error: Failed to open %s: %s\n",
                log_path, strerror(errno));
        return;
    }

    // Resize file to capacity
    if (ftruncate(g_log_fd, capacity_bytes) != 0) {
        fprintf(stderr, "[TENSOR_TRACE] Error: Failed to resize file: %s\n",
                strerror(errno));
        close(g_log_fd);
        g_log_fd = -1;
        return;
    }

    // Memory-map the file
    g_log_buffer = mmap(NULL, capacity_bytes, PROT_WRITE, MAP_SHARED, g_log_fd, 0);
    if (g_log_buffer == MAP_FAILED) {
        fprintf(stderr, "[TENSOR_TRACE] Error: Failed to mmap: %s\n",
                strerror(errno));
        close(g_log_fd);
        g_log_fd = -1;
        g_log_buffer = NULL;
        return;
    }

    g_log_capacity = capacity_bytes;
    g_log_offset = 0;

    // Initialize buffer stats tracking (Phase 1.3)
    tensor_trace_init_buffer_stats();

    printf("[TENSOR_TRACE] Initialized: %s (%.2f GB capacity)\n",
           log_path, capacity_bytes / (1024.0 * 1024.0 * 1024.0));
}

void tensor_trace_log(const struct TensorAccessLog* entry) {
    if (g_log_buffer == NULL) {
        return;  // Not initialized, silently skip
    }

    // Add to thread-local buffer
    g_thread_local_buffer[g_thread_local_offset++] = *entry;

    // Flush when buffer is full
    if (g_thread_local_offset >= THREAD_LOCAL_BUFFER_SIZE) {
        // TODO: Add atomic operation here for thread safety
        // For MVP, we'll use simple (non-thread-safe) flush

        size_t bytes_to_write = g_thread_local_offset * sizeof(struct TensorAccessLog);

        // Check capacity
        if (g_log_offset + bytes_to_write <= g_log_capacity) {
            // Copy thread-local buffer to global mmap'd buffer
            memcpy((char*)g_log_buffer + g_log_offset,
                   g_thread_local_buffer,
                   bytes_to_write);
            g_log_offset += bytes_to_write;
        } else {
            fprintf(stderr, "[TENSOR_TRACE] Warning: Log buffer full, dropping entries\n");
        }

        // Reset thread-local buffer
        g_thread_local_offset = 0;
    }
}

void tensor_trace_shutdown(void) {
    if (g_log_buffer == NULL) {
        return;  // Not initialized
    }

    // Flush remaining thread-local entries
    if (g_thread_local_offset > 0) {
        size_t bytes_to_write = g_thread_local_offset * sizeof(struct TensorAccessLog);

        if (g_log_offset + bytes_to_write <= g_log_capacity) {
            memcpy((char*)g_log_buffer + g_log_offset,
                   g_thread_local_buffer,
                   bytes_to_write);
            g_log_offset += bytes_to_write;
        }

        g_thread_local_offset = 0;
    }

    // Sync to disk
    msync(g_log_buffer, g_log_capacity, MS_SYNC);

    // Unmap and close
    munmap(g_log_buffer, g_log_capacity);
    close(g_log_fd);

    size_t num_entries = g_log_offset / sizeof(struct TensorAccessLog);
    printf("[TENSOR_TRACE] Shutdown: %zu entries logged (%.2f MB)\n",
           num_entries, g_log_offset / (1024.0 * 1024.0));

    // Shutdown buffer stats tracking (Phase 1.3)
    tensor_trace_shutdown_buffer_stats();

    // Reset state
    g_log_buffer = NULL;
    g_log_capacity = 0;
    g_log_offset = 0;
    g_log_fd = -1;
}

void tensor_trace_register_tensor(
    const char* name,
    void* data_ptr,
    uint64_t file_offset,
    uint64_t size_bytes)
{
    // Check capacity
    if (g_registry_count >= MAX_REGISTERED_TENSORS) {
        fprintf(stderr, "[TENSOR_TRACE] Warning: Registry full, cannot register '%s'\n", name);
        return;
    }

    // Store tensor metadata
    struct TensorRegistryEntry* entry = &g_tensor_registry[g_registry_count];
    entry->data_ptr = data_ptr;
    entry->file_offset = file_offset;
    entry->size_bytes = size_bytes;
    entry->tensor_idx = g_registry_count;

    // Copy name (safely)
    strncpy(entry->name, name ? name : "", sizeof(entry->name) - 1);
    entry->name[sizeof(entry->name) - 1] = '\0';

    // Extract layer ID from name
    entry->layer_id = tensor_trace_extract_layer_id(entry->name);

    g_registry_count++;

    // Debug: Log registration (can be disabled later)
    // printf("[TENSOR_TRACE] Registered tensor %u: %s (ptr=%p, offset=%llu, size=%llu, layer=%u)\n",
    //        entry->tensor_idx, entry->name, data_ptr, file_offset, size_bytes, entry->layer_id);
}

// Lookup tensor by data pointer (returns tensor_idx, or UINT32_MAX if not found)
uint32_t tensor_trace_lookup_idx(void* data_ptr) {
    // Simple linear search (fine for ~201 tensors)
    // Could optimize with hash table if needed for larger models
    for (uint32_t i = 0; i < g_registry_count; i++) {
        if (g_tensor_registry[i].data_ptr == data_ptr) {
            return i;  // Found!
        }
    }
    return UINT32_MAX;  // Not found
}

// Dump registry table to CSV file for validation
void tensor_trace_dump_registry(const char* output_path) {
    FILE* f = fopen(output_path, "w");
    if (!f) {
        fprintf(stderr, "[TENSOR_TRACE] Error: Failed to open %s for writing: %s\n",
                output_path, strerror(errno));
        return;
    }

    // Write CSV header
    fprintf(f, "tensor_idx,tensor_name,data_ptr,file_offset,size_bytes,layer_id\n");

    // Write all registered tensors
    for (uint32_t i = 0; i < g_registry_count; i++) {
        struct TensorRegistryEntry* entry = &g_tensor_registry[i];
        fprintf(f, "%u,%s,%p,%llu,%llu,%u\n",
                entry->tensor_idx,
                entry->name,
                entry->data_ptr,
                entry->file_offset,
                entry->size_bytes,
                entry->layer_id);
    }

    fclose(f);
    printf("[TENSOR_TRACE] Dumped %u tensors to %s\n", g_registry_count, output_path);
}

// === Buffer Tracking (Phase 1.3) ===

// Buffer stats output file (JSONL format for streaming)
static FILE * g_buffer_stats_file = NULL;

// Initialize buffer stats file
static void tensor_trace_init_buffer_stats(void) {
    g_buffer_stats_file = fopen("/tmp/buffer_stats.jsonl", "w");
    if (g_buffer_stats_file == NULL) {
        fprintf(stderr, "[TENSOR_TRACE] Warning: Failed to open buffer_stats.jsonl: %s\n",
                strerror(errno));
    } else {
        printf("[TENSOR_TRACE] Buffer stats logging to /tmp/buffer_stats.jsonl\n");
    }
}

// Log buffer allocation event
void tensor_trace_log_buffer_alloc(
    uint64_t buffer_id,
    void * buffer_ptr,
    size_t size_bytes,
    const char * buffer_name,
    const char * backend_type,
    uint8_t buffer_usage,
    uint16_t layer_id) {

    if (g_buffer_stats_file == NULL || !g_trace_enabled) {
        return;
    }

    // Get timestamp (relative to trace start, for correlation with tensor ops)
    uint64_t timestamp_ns = tensor_trace_get_timestamp_ns();

    // Write as JSON line for easy parsing
    fprintf(g_buffer_stats_file,
        "{\"timestamp_ms\":%.3f,\"event\":\"alloc\",\"buffer_id\":%llu,"
        "\"buffer_ptr\":%llu,\"size\":%zu,\"name\":\"%s\",\"backend\":\"%s\","
        "\"usage\":%u,\"layer\":%u}\n",
        timestamp_ns / 1e6,
        (unsigned long long)buffer_id,
        (unsigned long long)buffer_ptr,
        size_bytes,
        buffer_name ? buffer_name : "unnamed",
        backend_type ? backend_type : "unknown",
        buffer_usage,
        layer_id
    );
    fflush(g_buffer_stats_file);
}

// Log buffer deallocation event
void tensor_trace_log_buffer_dealloc(uint64_t buffer_id) {
    if (g_buffer_stats_file == NULL || !g_trace_enabled) {
        return;
    }

    // Get timestamp (for correlation)
    uint64_t timestamp_ns = tensor_trace_get_timestamp_ns();

    fprintf(g_buffer_stats_file,
        "{\"timestamp_ms\":%.3f,\"event\":\"dealloc\",\"buffer_id\":%llu}\n",
        timestamp_ns / 1e6,
        (unsigned long long)buffer_id
    );
    fflush(g_buffer_stats_file);
}

// Shutdown buffer stats
static void tensor_trace_shutdown_buffer_stats(void) {
    if (g_buffer_stats_file != NULL) {
        fclose(g_buffer_stats_file);
        g_buffer_stats_file = NULL;
        printf("[TENSOR_TRACE] Buffer stats closed\n");
    }
}

// === NEW: Generic Operation Logging (Phase 1.1) ===

// Include ggml headers for tensor types
#include "ggml.h"
#include "ggml-backend.h"

// Disk offset storage (Phase 1.2)
// Maps tensor name to GGUF file offset
#define MAX_OFFSET_MAP_SIZE 2048
static struct {
    char name[64];
    uint64_t offset;
} g_offset_map[MAX_OFFSET_MAP_SIZE];
static uint32_t g_offset_map_count = 0;

// Register tensor disk offset (called from llama during model load)
void tensor_trace_register_disk_offset(const char * name, uint64_t offset) {
    if (name == NULL || g_offset_map_count >= MAX_OFFSET_MAP_SIZE) {
        return;
    }

    strncpy(g_offset_map[g_offset_map_count].name, name, sizeof(g_offset_map[0].name) - 1);
    g_offset_map[g_offset_map_count].name[sizeof(g_offset_map[0].name) - 1] = '\0';
    g_offset_map[g_offset_map_count].offset = offset;
    g_offset_map_count++;
}

// Lookup disk offset by tensor name
static uint64_t tensor_trace_lookup_disk_offset(const char * name) {
    if (name == NULL) return 0;

    for (uint32_t i = 0; i < g_offset_map_count; i++) {
        if (strcmp(g_offset_map[i].name, name) == 0) {
            return g_offset_map[i].offset;
        }
    }
    return 0;
}

// === Memory Source Detection (Phase 1.2) ===
// Uses buffer API to determine if tensor is from GGUF file or runtime buffer

enum MemorySource tensor_trace_detect_memory_source(const struct ggml_tensor * tensor) {
    if (tensor == NULL || tensor->buffer == NULL) {
        return MEMORY_SOURCE_BUFFER;  // Default to buffer if unknown
    }

    // Query buffer usage type from ggml backend API
    ggml_backend_buffer_t buf = tensor->buffer;
    enum ggml_backend_buffer_usage usage = ggml_backend_buffer_get_usage(buf);

    if (usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return MEMORY_SOURCE_DISK;  // Model weights from GGUF file
    } else {
        // GGML_BACKEND_BUFFER_USAGE_COMPUTE or GGML_BACKEND_BUFFER_USAGE_ANY
        return MEMORY_SOURCE_BUFFER;  // Runtime buffers (KV cache, scratch, activations)
    }
}

uint64_t tensor_trace_get_disk_offset(const struct ggml_tensor * tensor) {
    if (tensor == NULL) {
        return 0;
    }

    // Get tensor name
    const char * name = ggml_get_name(tensor);
    if (name == NULL || name[0] == '\0') {
        return 0;  // Unnamed tensor (intermediate result)
    }

    // Lookup from registered offset map
    return tensor_trace_lookup_disk_offset(name);
}

uint64_t tensor_trace_get_buffer_id(const struct ggml_tensor * tensor) {
    // PLACEHOLDER: Use tensor pointer as buffer ID for now
    // This will be properly implemented in Phase 1.2
    if (tensor == NULL || tensor->buffer == NULL) {
        return 0;
    }
    return (uint64_t)tensor->buffer;
}

// Main generic operation logging function
// Called from ggml_compute_forward() dispatcher BEFORE switch statement
// Creates ONE entry per operation with ALL sources embedded
void tensor_trace_log_operation(
    const struct ggml_tensor * dst,
    int ith) {

    if (!g_trace_enabled || dst == NULL) {
        return;
    }

    // Only first thread logs (avoid duplicate entries)
    if (ith != 0) {
        return;
    }

    // Create ONE trace entry for this operation
    struct TensorAccessLog entry = {0};  // Zero-initialize all fields

    // === Fill Operation Metadata ===
    entry.timestamp_ns = tensor_trace_get_timestamp_ns();
    entry.thread_id = tensor_trace_get_thread_id();
    entry.operation_type = (uint8_t)dst->op;  // Use ggml_op enum directly
    entry.phase = g_current_phase;
    entry.token_id = g_current_token_id;

    // === Fill Destination Tensor ===
    const char * dst_name = ggml_get_name(dst);
    if (dst_name != NULL) {
        strncpy(entry.dst_name, dst_name, sizeof(entry.dst_name) - 1);
        entry.dst_name[sizeof(entry.dst_name) - 1] = '\0';
    }

    // Extract layer ID from destination or first source
    entry.layer_id = tensor_trace_extract_layer_id(dst_name);

    // === Fill ALL Source Tensors ===
    entry.num_sources = 0;

    for (int i = 0; i < GGML_MAX_SRC && i < 4; i++) {  // Max 4 sources in our struct
        const struct ggml_tensor * src = dst->src[i];

        // Stop when we hit NULL (no more sources)
        if (src == NULL) {
            break;
        }

        // Skip if tensor has no data (shouldn't happen, but safety check)
        if (src->data == NULL) {
            continue;
        }

        // Fill this source's information
        struct SourceTensorInfo * src_info = &entry.sources[entry.num_sources];

        // Name
        const char * src_name = ggml_get_name(src);
        if (src_name != NULL) {
            strncpy(src_info->name, src_name, sizeof(src_info->name) - 1);
            src_info->name[sizeof(src_info->name) - 1] = '\0';
        }

        // Basic info
        src_info->tensor_ptr = (uint64_t)src->data;
        src_info->size_bytes = (uint32_t)ggml_nbytes(src);
        src_info->layer_id = tensor_trace_extract_layer_id(src_name);
        src_info->tensor_idx = tensor_trace_lookup_idx(src->data);

        // Memory source detection (placeholder for now)
        src_info->memory_source = tensor_trace_detect_memory_source(src);
        if (src_info->memory_source == MEMORY_SOURCE_DISK) {
            src_info->disk_offset_or_buffer_id = tensor_trace_get_disk_offset(src);
        } else {
            src_info->disk_offset_or_buffer_id = tensor_trace_get_buffer_id(src);
        }

        // If layer_id not set at operation level, try to get it from first source
        if (entry.layer_id == 65535 && src_info->layer_id != 65535) {
            entry.layer_id = src_info->layer_id;
        }

        entry.num_sources++;
    }

    // Log the single entry (contains operation + destination + all sources)
    tensor_trace_log(&entry);
}
