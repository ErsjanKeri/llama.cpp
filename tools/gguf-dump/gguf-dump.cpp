// GGUF Structure Dumper
// Extracts tensor metadata from GGUF files for tensor access tracking
// Usage: ./gguf_offset_dump <model.gguf> > gguf_structure.csv

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Minimal GGUF parsing (self-contained, no llama.cpp dependencies for MVP)
// Based on GGUF specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

#define GGUF_MAGIC 0x46554747  // "GGUF" in little-endian
#define GGUF_VERSION 3

// GGML type information for correct size calculation
// Extracted from ggml/src/ggml.c type_traits array (lines 607-897)
struct ggml_type_traits {
    const char* type_name;
    uint32_t blck_size;    // Block size (number of elements per block)
    uint32_t type_size;    // Block size in bytes
};

// Type information table for all GGML quantization formats
static const ggml_type_traits GGML_TYPE_TRAITS[] = {
    // type_id: {name, blck_size, type_size_bytes}
    {/*  0 */ "F32",      1,   4},   // float32
    {/*  1 */ "F16",      1,   2},   // float16
    {/*  2 */ "Q4_0",     32,  18},  // 32 elements in 18 bytes
    {/*  3 */ "Q4_1",     32,  20},  // 32 elements in 20 bytes
    {/*  4 */ "Q5_0",     32,  22},  // Q5_0 (not used in modern models)
    {/*  5 */ "Q5_1",     32,  24},  // Q5_1 (not used in modern models)
    {/*  6 */ "Q8_0",     32,  34},  // 32 elements in 34 bytes
    {/*  7 */ "Q8_1",     32,  36},  // 32 elements in 36 bytes
    {/*  8 */ "Q2_K",     256, 82},  // 256 elements in 82 bytes (QK_K)
    {/*  9 */ "Q3_K_S",   256, 110}, // 256 elements in 110 bytes
    {/* 10 */ "Q3_K_M",   256, 110}, // 256 elements in 110 bytes
    {/* 11 */ "Q3_K_L",   256, 110}, // 256 elements in 110 bytes
    {/* 12 */ "Q4_K",     256, 144}, // 256 elements in 144 bytes
    {/* 13 */ "Q5_K",     256, 176}, // 256 elements in 176 bytes
    {/* 14 */ "Q6_K",     256, 210}, // 256 elements in 210 bytes
    {/* 15 */ "Q8_K",     256, 292}, // 256 elements in 292 bytes
    {/* 16 */ "IQ2_XXS",  256, 66},  // 256 elements in 66 bytes
    {/* 17 */ "IQ2_XS",   256, 74},  // 256 elements in 74 bytes
    {/* 18 */ "IQ3_XXS",  256, 98},  // 256 elements in 98 bytes
    {/* 19 */ "IQ1_S",    256, 50},  // 256 elements in 50 bytes
    {/* 20 */ "IQ4_NL",   32,  18},  // 32 elements in 18 bytes
    {/* 21 */ "IQ3_S",    256, 110}, // 256 elements in 110 bytes
    {/* 22 */ "IQ2_S",    256, 82},  // 256 elements in 82 bytes
    {/* 23 */ "IQ4_XS",   256, 136}, // 256 elements in 136 bytes
    {/* 24 */ "I8",       1,   1},   // int8
    {/* 25 */ "I16",      1,   2},   // int16
    {/* 26 */ "I32",      1,   4},   // int32
    {/* 27 */ "I64",      1,   8},   // int64
    {/* 28 */ "F64",      1,   8},   // float64
    {/* 29 */ "IQ1_M",    256, 56},  // 256 elements in 56 bytes
    {/* 30 */ "BF16",     1,   2},   // bfloat16
    {/* 31 */ "Q4_0_4_4", 32,  18},  // Q4_0 with 4x4 blocking
    {/* 32 */ "Q4_0_4_8", 32,  18},  // Q4_0 with 4x8 blocking
    {/* 33 */ "Q4_0_8_8", 32,  18},  // Q4_0 with 8x8 blocking
    {/* 34 */ "TQ1_0",    256, 50},  // Ternary quantization
    {/* 35 */ "TQ2_0",    256, 66},  // Ternary quantization
};

#define GGML_TYPE_COUNT 36

// Type 39 (MXFP4) - handled separately as it comes after the main array
#define GGML_TYPE_MXFP4 39
static const ggml_type_traits GGML_TYPE_MXFP4_TRAITS = {"MXFP4", 32, 17};

/**
 * Calculate correct tensor size accounting for quantization.
 *
 * @param tensor_type GGML type ID (0=F32, 1=F16, 12=Q4_K, 39=MXFP4, etc.)
 * @param n_dims Number of dimensions
 * @param ne Array of dimension sizes
 * @return Size in bytes
 */
static uint64_t calculate_tensor_size(uint32_t tensor_type, uint32_t n_dims, const uint64_t* ne) {
    // Get type traits
    const ggml_type_traits* traits = nullptr;

    if (tensor_type == GGML_TYPE_MXFP4) {
        // Special case for MXFP4 (type 39)
        traits = &GGML_TYPE_MXFP4_TRAITS;
    } else if (tensor_type < GGML_TYPE_COUNT) {
        // Standard types (0-35)
        traits = &GGML_TYPE_TRAITS[tensor_type];
    } else {
        // Unknown type - fallback to F32 assumption (4 bytes)
        fprintf(stderr, "Warning: Unknown tensor type %u, assuming F32\n", tensor_type);
        traits = &GGML_TYPE_TRAITS[0];  // F32
    }

    // Calculate total number of elements
    uint64_t n_elements = 1;
    for (uint32_t d = 0; d < n_dims; d++) {
        n_elements *= ne[d];
    }

    // Calculate size based on quantization
    if (traits->blck_size == 1) {
        // Non-quantized: simple multiplication
        return n_elements * traits->type_size;
    } else {
        // Quantized: calculate number of blocks (round up)
        uint64_t n_blocks = (n_elements + traits->blck_size - 1) / traits->blck_size;
        return n_blocks * traits->type_size;
    }
}

struct gguf_header {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

struct tensor_info {
    std::string name;
    uint64_t offset;
    uint64_t size_bytes;
    int layer_id;
    std::string component_type;
    uint32_t n_dims;
    uint64_t ne[4];  // dimensions
    uint32_t tensor_type;  // GGML type (for size calculation)
};

// Extract layer ID from tensor name (e.g., "blk.5.attn_q.weight" → 5)
int extract_layer_id(const char* name) {
    // Pattern: "blk.N.*"
    if (strncmp(name, "blk.", 4) == 0) {
        int layer = -1;
        if (sscanf(name + 4, "%d", &layer) == 1) {
            return layer;
        }
    }
    return -1;  // Not a layer tensor
}

// Determine component type from tensor name
std::string determine_component_type(const char* name) {
    std::string n(name);

    // Token embeddings
    if (n.find("token_embd") != std::string::npos) return "Token Embeddings";

    // Output projection
    if (n.find("output") != std::string::npos) return "Output Projection";

    // Attention components
    if (n.find("attn_q") != std::string::npos) return "Attention Q";
    if (n.find("attn_k") != std::string::npos) return "Attention K";
    if (n.find("attn_v") != std::string::npos) return "Attention V";
    if (n.find("attn_output") != std::string::npos) return "Attention Output";
    if (n.find("attn_norm") != std::string::npos) return "Attention Norm";

    // FFN components
    if (n.find("ffn_up") != std::string::npos) return "FFN Up";
    if (n.find("ffn_down") != std::string::npos) return "FFN Down";
    if (n.find("ffn_gate") != std::string::npos) return "FFN Gate";
    if (n.find("ffn_norm") != std::string::npos) return "FFN Norm";

    // MoE experts
    if (n.find("expert") != std::string::npos) {
        // Extract expert ID
        size_t expert_pos = n.find("expert_");
        if (expert_pos != std::string::npos) {
            int expert_id = -1;
            if (sscanf(n.c_str() + expert_pos + 7, "%d", &expert_id) == 1) {
                char buf[64];
                snprintf(buf, sizeof(buf), "MoE Expert %d", expert_id);

                // Determine expert component
                if (n.find("ffn_up") != std::string::npos) return std::string(buf) + " Up";
                if (n.find("ffn_down") != std::string::npos) return std::string(buf) + " Down";
                if (n.find("ffn_gate") != std::string::npos) return std::string(buf) + " Gate";
            }
        }
        return "MoE Expert";
    }

    return "Other";
}

// Detect if tensor is a MoE expert tensor (3D with _exps in name)
bool is_moe_expert_tensor(const char* name, uint32_t n_dims) {
    if (n_dims != 3) return false;

    return (strstr(name, "ffn_down_exps") != NULL ||
            strstr(name, "ffn_gate_exps") != NULL ||
            strstr(name, "ffn_up_exps") != NULL);
}

// Read GGUF string (length-prefixed)
bool read_gguf_string(FILE* f, std::string& out) {
    uint64_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) return false;

    if (len > 1024 * 1024) {  // Sanity check: max 1MB string
        fprintf(stderr, "Error: String too long (%llu bytes)\n", len);
        return false;
    }

    std::vector<char> buf(len + 1);
    if (fread(buf.data(), 1, len, f) != len) return false;
    buf[len] = '\0';

    out = buf.data();
    return true;
}

// Skip GGUF value based on type
bool skip_gguf_value(FILE* f, uint32_t type) {
    switch (type) {
        case 0:  // UINT8
        case 1:  // INT8
            fseek(f, 1, SEEK_CUR);
            return true;
        case 2:  // UINT16
        case 3:  // INT16
            fseek(f, 2, SEEK_CUR);
            return true;
        case 4:  // UINT32
        case 5:  // INT32
        case 6:  // FLOAT32
            fseek(f, 4, SEEK_CUR);
            return true;
        case 7:  // BOOL
            fseek(f, 1, SEEK_CUR);
            return true;
        case 8:  // STRING
            {
                std::string dummy;
                return read_gguf_string(f, dummy);
            }
        case 9:  // ARRAY
            {
                uint32_t array_type;
                uint64_t array_len;
                if (fread(&array_type, sizeof(array_type), 1, f) != 1) return false;
                if (fread(&array_len, sizeof(array_len), 1, f) != 1) return false;

                // Skip array elements
                for (uint64_t i = 0; i < array_len; i++) {
                    if (!skip_gguf_value(f, array_type)) return false;
                }
                return true;
            }
        case 10: // UINT64
        case 11: // INT64
        case 12: // FLOAT64
            fseek(f, 8, SEEK_CUR);
            return true;
        default:
            fprintf(stderr, "Error: Unknown value type %u\n", type);
            return false;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        fprintf(stderr, "Output: CSV with tensor metadata\n");
        return 1;
    }

    const char* filename = argv[1];
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Failed to open %s\n", filename);
        return 1;
    }

    // Read header
    gguf_header header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read header\n");
        fclose(f);
        return 1;
    }

    // Verify magic
    if (header.magic != GGUF_MAGIC) {
        fprintf(stderr, "Error: Invalid GGUF file (bad magic: 0x%08x)\n", header.magic);
        fclose(f);
        return 1;
    }

    if (header.version != GGUF_VERSION) {
        fprintf(stderr, "Warning: GGUF version %u (expected %u)\n", header.version, GGUF_VERSION);
    }

    fprintf(stderr, "GGUF file: %s\n", filename);
    fprintf(stderr, "Tensors: %llu\n", header.n_tensors);
    fprintf(stderr, "Metadata KV pairs: %llu\n", header.n_kv);

    // Skip metadata key-value pairs
    for (uint64_t i = 0; i < header.n_kv; i++) {
        std::string key;
        if (!read_gguf_string(f, key)) {
            fprintf(stderr, "Error: Failed to read KV key %llu\n", i);
            fclose(f);
            return 1;
        }

        uint32_t value_type;
        if (fread(&value_type, sizeof(value_type), 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read KV value type\n");
            fclose(f);
            return 1;
        }

        if (!skip_gguf_value(f, value_type)) {
            fprintf(stderr, "Error: Failed to skip KV value\n");
            fclose(f);
            return 1;
        }
    }

    // Read tensor info
    std::vector<tensor_info> tensors;
    tensors.reserve(header.n_tensors);

    for (uint64_t i = 0; i < header.n_tensors; i++) {
        tensor_info info;

        // Read tensor name
        if (!read_gguf_string(f, info.name)) {
            fprintf(stderr, "Error: Failed to read tensor name %llu\n", i);
            fclose(f);
            return 1;
        }

        // Read n_dims
        if (fread(&info.n_dims, sizeof(info.n_dims), 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read n_dims\n");
            fclose(f);
            return 1;
        }

        if (info.n_dims > 4) {
            fprintf(stderr, "Error: Invalid n_dims %u\n", info.n_dims);
            fclose(f);
            return 1;
        }

        // Read dimensions
        for (uint32_t d = 0; d < info.n_dims; d++) {
            if (fread(&info.ne[d], sizeof(uint64_t), 1, f) != 1) {
                fprintf(stderr, "Error: Failed to read dimension %u\n", d);
                fclose(f);
                return 1;
            }
        }

        // Read tensor type
        uint32_t tensor_type;
        if (fread(&tensor_type, sizeof(tensor_type), 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read tensor type\n");
            fclose(f);
            return 1;
        }

        // Read tensor offset
        if (fread(&info.offset, sizeof(info.offset), 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read tensor offset\n");
            fclose(f);
            return 1;
        }

        // Store tensor type for later use
        info.tensor_type = tensor_type;

        // Calculate tensor size using correct quantization-aware function
        info.size_bytes = calculate_tensor_size(tensor_type, info.n_dims, info.ne);

        // Extract layer ID and component type
        info.layer_id = extract_layer_id(info.name.c_str());
        info.component_type = determine_component_type(info.name.c_str());

        tensors.push_back(info);
    }

    // Calculate data section offset
    // After reading all tensor info, we're at the end of the tensor metadata
    long tensor_info_end = ftell(f);
    fclose(f);

    // Data section is aligned to 32 bytes (GGUF_DEFAULT_ALIGNMENT)
    const uint32_t alignment = 32;
    uint64_t data_section_offset = ((tensor_info_end + alignment - 1) / alignment) * alignment;

    fprintf(stderr, "Tensor info ends at: %ld bytes\n", tensor_info_end);
    fprintf(stderr, "Data section starts at: %llu bytes (aligned to %u bytes)\n",
            data_section_offset, alignment);
    fprintf(stderr, "Adding %llu bytes to all tensor offsets\n", data_section_offset);

    // Output CSV
    printf("tensor_name,file_offset,size_bytes,layer_id,component_type,n_dims,dim0,dim1,dim2,dim3\n");

    for (const auto& t : tensors) {
        // Check if this is a MoE expert tensor that needs splitting
        if (is_moe_expert_tensor(t.name.c_str(), t.n_dims)) {
            // Split into individual expert entries
            uint64_t n_experts = t.ne[2];  // Third dimension = number of experts

            // Calculate size of one expert slice [dim0, dim1] using correct quantization
            uint64_t expert_dims[2] = {t.ne[0], t.ne[1]};
            uint64_t expert_size = calculate_tensor_size(t.tensor_type, 2, expert_dims);

            for (uint64_t exp_id = 0; exp_id < n_experts; exp_id++) {
                // Calculate absolute offset (relative offset + data section offset)
                uint64_t expert_relative_offset = t.offset + (exp_id * expert_size);
                uint64_t expert_absolute_offset = data_section_offset + expert_relative_offset;

                // Output format: tensor_name[expert_id]
                printf("%s[%llu],%llu,%llu,%d,%s Expert %llu,%u,%llu,%llu,0,0\n",
                       t.name.c_str(),
                       exp_id,
                       expert_absolute_offset,
                       expert_size,
                       t.layer_id,
                       t.component_type.c_str(),
                       exp_id,
                       2,  // n_dims = 2 for individual expert (not 3)
                       t.ne[0],
                       t.ne[1]);
            }
        } else {
            // Normal tensor - add data section offset to get absolute position
            uint64_t absolute_offset = data_section_offset + t.offset;

            printf("%s,%llu,%llu,%d,%s,%u,%llu,%llu,%llu,%llu\n",
                   t.name.c_str(),
                   absolute_offset,
                   t.size_bytes,
                   t.layer_id,
                   t.component_type.c_str(),
                   t.n_dims,
                   t.n_dims > 0 ? t.ne[0] : 0,
                   t.n_dims > 1 ? t.ne[1] : 0,
                   t.n_dims > 2 ? t.ne[2] : 0,
                   t.n_dims > 3 ? t.ne[3] : 0);
        }
    }

    fprintf(stderr, "\nDumped %zu tensors\n", tensors.size());

    return 0;
}
