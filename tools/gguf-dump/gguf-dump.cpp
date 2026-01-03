// GGUF Structure Dumper
// Extracts tensor metadata from GGUF files for tensor access tracking
// Usage: ./gguf_offset_dump <model.gguf> > gguf_structure.csv

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Minimal GGUF parsing (self-contained, no llama.cpp dependencies for MVP)
// Based on GGUF specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

#define GGUF_MAGIC 0x46554747  // "GGUF" in little-endian
#define GGUF_VERSION 3

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

        // Calculate tensor size (simplified - assumes element size based on type)
        // Type 0 = F32 (4 bytes), Type 1 = F16 (2 bytes), etc.
        size_t element_size = (tensor_type == 1) ? 2 : 4;  // Simplified
        info.size_bytes = element_size;
        for (uint32_t d = 0; d < info.n_dims; d++) {
            info.size_bytes *= info.ne[d];
        }

        // Extract layer ID and component type
        info.layer_id = extract_layer_id(info.name.c_str());
        info.component_type = determine_component_type(info.name.c_str());

        tensors.push_back(info);
    }

    fclose(f);

    // Output CSV
    printf("tensor_name,file_offset,size_bytes,layer_id,component_type,n_dims,dim0,dim1,dim2,dim3\n");

    for (const auto& t : tensors) {
        printf("%s,%llu,%llu,%d,%s,%u,%llu,%llu,%llu,%llu\n",
               t.name.c_str(),
               t.offset,
               t.size_bytes,
               t.layer_id,
               t.component_type.c_str(),
               t.n_dims,
               t.n_dims > 0 ? t.ne[0] : 0,
               t.n_dims > 1 ? t.ne[1] : 0,
               t.n_dims > 2 ? t.ne[2] : 0,
               t.n_dims > 3 ? t.ne[3] : 0);
    }

    fprintf(stderr, "\nDumped %zu tensors\n", tensors.size());

    return 0;
}
