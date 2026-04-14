// BSC thesis: Expert weight pack file builder
//
// Reads a GGUF file and writes a sidecar .bscexp file containing only the
// MoE expert weight slices, laid out in 512-aligned slots for direct
// io_uring + O_DIRECT reads at runtime (no staging buffer, no memcpy).
//
// Format definition: src/bsc-expert-pack.h
//
// Usage:
//   llama-bsc-pack-experts <model.gguf> [-o output.bscexp] [--verify]
//
// Default output path: <model.gguf>.bscexp (sidecar next to source).
// --verify: read back the pack file and compare every slice to the source
//           byte-for-byte. Roughly doubles runtime; recommended for first run.
//
// GGUF parsing is adapted from tools/gguf-dump/gguf-dump.cpp.

#include "../../src/bsc-expert-pack.h"

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal GGUF parsing — adapted from tools/gguf-dump/gguf-dump.cpp.
// We keep this self-contained to match gguf-dump's standalone style.
// ---------------------------------------------------------------------------

#define GGUF_MAGIC   0x46554747  // "GGUF" little-endian
#define GGUF_VERSION 3

// Subset of GGML type traits we need to compute expert slice byte sizes.
// (Mirrors tools/gguf-dump/gguf-dump.cpp; only the entries used by current
// MoE models are required, but we include the full table for safety.)
struct ggml_type_traits {
    const char * type_name;
    uint32_t     blck_size;
    uint32_t     type_size;
};

static const ggml_type_traits GGML_TYPE_TRAITS[] = {
    {"F32",      1,   4},   {"F16",      1,   2},
    {"Q4_0",     32,  18},  {"Q4_1",     32,  20},
    {"Q5_0",     32,  22},  {"Q5_1",     32,  24},
    {"Q8_0",     32,  34},  {"Q8_1",     32,  36},
    {"Q2_K",     256, 82},  {"Q3_K_S",   256, 110},
    {"Q3_K_M",   256, 110}, {"Q3_K_L",   256, 110},
    {"Q4_K",     256, 144}, {"Q5_K",     256, 176},
    {"Q6_K",     256, 210}, {"Q8_K",     256, 292},
    {"IQ2_XXS",  256, 66},  {"IQ2_XS",   256, 74},
    {"IQ3_XXS",  256, 98},  {"IQ1_S",    256, 50},
    {"IQ4_NL",   32,  18},  {"IQ3_S",    256, 110},
    {"IQ2_S",    256, 82},  {"IQ4_XS",   256, 136},
    {"I8",       1,   1},   {"I16",      1,   2},
    {"I32",      1,   4},   {"I64",      1,   8},
    {"F64",      1,   8},   {"IQ1_M",    256, 56},
    {"BF16",     1,   2},   {"Q4_0_4_4", 32,  18},
    {"Q4_0_4_8", 32,  18},  {"Q4_0_8_8", 32,  18},
    {"TQ1_0",    256, 50},  {"TQ2_0",    256, 66},
};
static const uint32_t GGML_TYPE_COUNT = sizeof(GGML_TYPE_TRAITS) / sizeof(GGML_TYPE_TRAITS[0]);
static const ggml_type_traits GGML_TYPE_MXFP4 = {"MXFP4", 32, 17};
#define GGML_TYPE_MXFP4_ID 39

static const ggml_type_traits * lookup_type_traits(uint32_t type_id) {
    if (type_id < GGML_TYPE_COUNT) return &GGML_TYPE_TRAITS[type_id];
    if (type_id == GGML_TYPE_MXFP4_ID) return &GGML_TYPE_MXFP4;
    return nullptr;
}

// Compute the size in bytes of a (sub)tensor with the given dims using the
// quantization-aware block math.
static uint64_t tensor_size_bytes(uint32_t type_id, int n_dims, const uint64_t * ne) {
    const auto * t = lookup_type_traits(type_id);
    if (!t) return 0;

    uint64_t n_elements = 1;
    for (int d = 0; d < n_dims; d++) {
        n_elements *= ne[d];
    }
    if (t->blck_size == 1) {
        return n_elements * t->type_size;
    }
    uint64_t n_blocks = (n_elements + t->blck_size - 1) / t->blck_size;
    return n_blocks * t->type_size;
}

struct gguf_header {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

struct tensor_info {
    std::string name;
    uint32_t    n_dims;
    uint64_t    ne[4];
    uint32_t    type_id;
    uint64_t    rel_offset;   // offset from start of data section (raw GGUF value)
    uint64_t    abs_offset;   // absolute file offset (rel_offset + data_section_offset)
    uint64_t    size_bytes;
};

static bool read_gguf_string(FILE * f, std::string & out) {
    uint64_t len = 0;
    if (fread(&len, sizeof(len), 1, f) != 1) return false;
    if (len > 16ULL * 1024 * 1024) {  // 16 MB sanity bound
        fprintf(stderr, "bsc-pack-experts: GGUF string too long (%" PRIu64 " bytes)\n", len);
        return false;
    }
    std::vector<char> buf(len + 1);
    if (len > 0 && fread(buf.data(), 1, len, f) != len) return false;
    buf[len] = '\0';
    out = buf.data();
    return true;
}

static bool skip_gguf_value(FILE * f, uint32_t type) {
    switch (type) {
        case 0: case 1:                  return fseek(f, 1, SEEK_CUR) == 0; // u8/i8
        case 2: case 3:                  return fseek(f, 2, SEEK_CUR) == 0; // u16/i16
        case 4: case 5: case 6:          return fseek(f, 4, SEEK_CUR) == 0; // u32/i32/f32
        case 7:                          return fseek(f, 1, SEEK_CUR) == 0; // bool
        case 8: { std::string s; return read_gguf_string(f, s); }           // string
        case 9: {                                                            // array
            uint32_t array_type;
            uint64_t array_len;
            if (fread(&array_type, sizeof(array_type), 1, f) != 1) return false;
            if (fread(&array_len,  sizeof(array_len),  1, f) != 1) return false;
            for (uint64_t i = 0; i < array_len; i++) {
                if (!skip_gguf_value(f, array_type)) return false;
            }
            return true;
        }
        case 10: case 11: case 12:       return fseek(f, 8, SEEK_CUR) == 0; // u64/i64/f64
        default:
            fprintf(stderr, "bsc-pack-experts: unknown GGUF value type %u\n", type);
            return false;
    }
}

// Parse the GGUF file and return all tensor metadata, with absolute offsets.
// Returns true on success.
static bool parse_gguf(const char * path, std::vector<tensor_info> & out_tensors) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "bsc-pack-experts: cannot open %s: %s\n", path, strerror(errno));
        return false;
    }

    gguf_header h;
    if (fread(&h, sizeof(h), 1, f) != 1) {
        fprintf(stderr, "bsc-pack-experts: cannot read GGUF header\n");
        fclose(f);
        return false;
    }
    if (h.magic != GGUF_MAGIC) {
        fprintf(stderr, "bsc-pack-experts: bad magic 0x%08x (expected 'GGUF')\n", h.magic);
        fclose(f);
        return false;
    }
    if (h.version != GGUF_VERSION) {
        fprintf(stderr, "bsc-pack-experts: WARNING — GGUF version %u (expected %u)\n",
                h.version, GGUF_VERSION);
    }

    fprintf(stderr, "bsc-pack-experts: parsing GGUF (%" PRIu64 " tensors, %" PRIu64 " KV)\n",
            h.n_tensors, h.n_kv);

    // Skip metadata KV pairs
    for (uint64_t i = 0; i < h.n_kv; i++) {
        std::string key;
        if (!read_gguf_string(f, key)) {
            fprintf(stderr, "bsc-pack-experts: failed to read KV key %" PRIu64 "\n", i);
            fclose(f);
            return false;
        }
        uint32_t value_type;
        if (fread(&value_type, sizeof(value_type), 1, f) != 1) {
            fprintf(stderr, "bsc-pack-experts: failed to read KV type for '%s'\n", key.c_str());
            fclose(f);
            return false;
        }
        if (!skip_gguf_value(f, value_type)) {
            fprintf(stderr, "bsc-pack-experts: failed to skip KV value '%s'\n", key.c_str());
            fclose(f);
            return false;
        }
    }

    // Read tensor info entries
    out_tensors.clear();
    out_tensors.reserve(h.n_tensors);
    for (uint64_t i = 0; i < h.n_tensors; i++) {
        tensor_info ti{};
        if (!read_gguf_string(f, ti.name)) {
            fprintf(stderr, "bsc-pack-experts: failed to read tensor name %" PRIu64 "\n", i);
            fclose(f);
            return false;
        }
        if (fread(&ti.n_dims, sizeof(ti.n_dims), 1, f) != 1) { fclose(f); return false; }
        if (ti.n_dims > 4) {
            fprintf(stderr, "bsc-pack-experts: invalid n_dims %u for '%s'\n",
                    ti.n_dims, ti.name.c_str());
            fclose(f);
            return false;
        }
        for (uint32_t d = 0; d < ti.n_dims; d++) {
            if (fread(&ti.ne[d], sizeof(uint64_t), 1, f) != 1) { fclose(f); return false; }
        }
        if (fread(&ti.type_id,    sizeof(ti.type_id),    1, f) != 1) { fclose(f); return false; }
        if (fread(&ti.rel_offset, sizeof(ti.rel_offset), 1, f) != 1) { fclose(f); return false; }
        ti.size_bytes = tensor_size_bytes(ti.type_id, (int)ti.n_dims, ti.ne);
        out_tensors.push_back(ti);
    }

    // Compute the data section start offset (32-byte aligned per GGUF spec)
    long info_end = ftell(f);
    fclose(f);
    if (info_end < 0) {
        fprintf(stderr, "bsc-pack-experts: ftell failed\n");
        return false;
    }
    const uint32_t GGUF_DATA_ALIGN = 32;
    uint64_t data_section_offset =
        ((uint64_t)info_end + GGUF_DATA_ALIGN - 1) & ~((uint64_t)GGUF_DATA_ALIGN - 1);

    fprintf(stderr, "bsc-pack-experts: data section starts at offset %" PRIu64 "\n",
            data_section_offset);

    // Convert relative tensor offsets to absolute file offsets
    for (auto & ti : out_tensors) {
        ti.abs_offset = data_section_offset + ti.rel_offset;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Expert tensor discovery
// ---------------------------------------------------------------------------

// Identify whether a tensor name is one of the three expert weight tensors,
// and return the projection index (0=down, 1=gate, 2=up) and layer number.
// Returns true on match.
static bool match_expert_tensor(const std::string & name,
                                int & out_layer, int & out_proj_with_gate) {
    // Names look like: "blk.<L>.ffn_down_exps.weight"
    // We only handle the .weight variants (not .bias — those stay in the GGUF).
    if (name.compare(0, 4, "blk.") != 0) return false;

    size_t dot = name.find('.', 4);
    if (dot == std::string::npos) return false;
    std::string layer_str = name.substr(4, dot - 4);
    char * end = nullptr;
    long layer = strtol(layer_str.c_str(), &end, 10);
    if (end == layer_str.c_str() || *end != '\0') return false;

    std::string suffix = name.substr(dot + 1);
    if      (suffix == "ffn_down_exps.weight") out_proj_with_gate = 0;
    else if (suffix == "ffn_gate_exps.weight") out_proj_with_gate = 1;
    else if (suffix == "ffn_up_exps.weight")   out_proj_with_gate = 2;
    else return false;

    out_layer = (int)layer;
    return true;
}

// ---------------------------------------------------------------------------
// I/O helpers — pread/pwrite full
// ---------------------------------------------------------------------------

static bool pread_full(int fd, void * buf, size_t size, off_t offset) {
    size_t total = 0;
    char * p = (char *)buf;
    while (total < size) {
        ssize_t r = pread(fd, p + total, size - total, offset + (off_t)total);
        if (r < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "pread failed at offset %lld: %s\n",
                    (long long)(offset + (off_t)total), strerror(errno));
            return false;
        }
        if (r == 0) {
            fprintf(stderr, "pread short read at offset %lld (got %zu of %zu)\n",
                    (long long)(offset + (off_t)total), total, size);
            return false;
        }
        total += (size_t)r;
    }
    return true;
}

static bool pwrite_full(int fd, const void * buf, size_t size, off_t offset) {
    size_t total = 0;
    const char * p = (const char *)buf;
    while (total < size) {
        ssize_t w = pwrite(fd, p + total, size - total, offset + (off_t)total);
        if (w < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "pwrite failed at offset %lld: %s\n",
                    (long long)(offset + (off_t)total), strerror(errno));
            return false;
        }
        total += (size_t)w;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s <model.gguf> [-o output.bscexp] [--verify]\n"
        "\n"
        "Reads a GGUF file and writes a sidecar .bscexp pack file containing\n"
        "MoE expert weight slices in 512-aligned slots, suitable for direct\n"
        "io_uring + O_DIRECT reads at runtime.\n"
        "\n"
        "Options:\n"
        "  -o <path>   Output pack file path (default: <input>.bscexp)\n"
        "  --verify    After writing, read back every slot and compare to\n"
        "              the source byte-for-byte. Recommended for first run.\n",
        prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char * src_path = nullptr;
    std::string  out_path;
    bool verify = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--verify") {
            verify = true;
        } else if (a == "-o") {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: -o requires an argument\n", argv[0]);
                return 1;
            }
            out_path = argv[++i];
        } else if (a.size() > 0 && a[0] == '-') {
            fprintf(stderr, "%s: unknown option '%s'\n", argv[0], a.c_str());
            return 1;
        } else if (!src_path) {
            src_path = argv[i];
        } else {
            fprintf(stderr, "%s: unexpected positional argument '%s'\n", argv[0], a.c_str());
            return 1;
        }
    }

    if (!src_path) {
        print_usage(argv[0]);
        return 1;
    }
    if (out_path.empty()) {
        out_path = std::string(src_path) + ".bscexp";
    }

    // 1. Parse GGUF
    std::vector<tensor_info> tensors;
    if (!parse_gguf(src_path, tensors)) {
        return 1;
    }

    struct stat src_stat{};
    if (stat(src_path, &src_stat) != 0) {
        fprintf(stderr, "bsc-pack-experts: stat(%s) failed: %s\n", src_path, strerror(errno));
        return 1;
    }
    uint64_t src_size = (uint64_t)src_stat.st_size;

    // 2. Discover expert tensors
    // Map: (layer, proj_with_gate) -> tensor_info
    // proj_with_gate uses {0=down, 1=gate, 2=up}; we collapse to runtime ordering later.
    std::map<std::pair<int,int>, const tensor_info *> expert_map;
    int max_layer = -1;
    int n_expert  = -1;
    bool has_gate = false;
    uint64_t expert_bytes_consensus = 0;

    for (const auto & t : tensors) {
        int layer = -1;
        int pwg   = -1;
        if (!match_expert_tensor(t.name, layer, pwg)) continue;

        // Sanity: must be 3D [d0, d1, n_expert]
        if (t.n_dims != 3) {
            fprintf(stderr, "bsc-pack-experts: expert tensor '%s' has n_dims=%u (expected 3)\n",
                    t.name.c_str(), t.n_dims);
            return 1;
        }
        int e_count = (int)t.ne[2];
        if (n_expert < 0)       n_expert = e_count;
        else if (n_expert != e_count) {
            fprintf(stderr, "bsc-pack-experts: expert tensor '%s' has n_expert=%d, expected %d\n",
                    t.name.c_str(), e_count, n_expert);
            return 1;
        }

        // Per-expert byte size = size of the 2D slice [d0, d1]
        uint64_t slice_dims[2] = {t.ne[0], t.ne[1]};
        uint64_t slice_bytes = tensor_size_bytes(t.type_id, 2, slice_dims);
        if (slice_bytes == 0) {
            fprintf(stderr, "bsc-pack-experts: cannot compute slice size for '%s' (type %u)\n",
                    t.name.c_str(), t.type_id);
            return 1;
        }
        if (expert_bytes_consensus == 0) expert_bytes_consensus = slice_bytes;
        else if (expert_bytes_consensus != slice_bytes) {
            fprintf(stderr, "bsc-pack-experts: '%s' has slice_bytes=%" PRIu64
                    ", expected %" PRIu64 " — mixed sizes not supported\n",
                    t.name.c_str(), slice_bytes, expert_bytes_consensus);
            return 1;
        }

        if (pwg == 1) has_gate = true;
        if (layer > max_layer) max_layer = layer;
        expert_map[{layer, pwg}] = &t;
    }

    if (expert_map.empty()) {
        fprintf(stderr, "bsc-pack-experts: no expert weight tensors found\n");
        return 1;
    }

    int n_layers = max_layer + 1;

    // Figure out runtime projection ordering:
    //   has_gate=true  -> n_proj=3, runtime order: 0=down, 1=gate, 2=up
    //   has_gate=false -> n_proj=2, runtime order: 0=down, 1=up
    int n_proj = has_gate ? 3 : 2;

    // Verify completeness: every (layer, proj) slot must exist.
    for (int l = 0; l < n_layers; l++) {
        if (!expert_map.count({l, 0})) {
            fprintf(stderr, "bsc-pack-experts: missing ffn_down_exps for layer %d\n", l);
            return 1;
        }
        if (has_gate && !expert_map.count({l, 1})) {
            fprintf(stderr, "bsc-pack-experts: missing ffn_gate_exps for layer %d\n", l);
            return 1;
        }
        if (!expert_map.count({l, 2})) {
            fprintf(stderr, "bsc-pack-experts: missing ffn_up_exps for layer %d\n", l);
            return 1;
        }
    }

    // 3. Build header
    bsc_expert_pack_header hdr{};
    memcpy(hdr.magic, BSC_EXPERT_PACK_MAGIC, BSC_EXPERT_PACK_MAGIC_LEN);
    hdr.header_size      = BSC_EXPERT_PACK_HEADER_SIZE;
    hdr.n_layers         = (uint32_t)n_layers;
    hdr.n_proj           = (uint32_t)n_proj;
    hdr.n_expert         = (uint32_t)n_expert;
    hdr.expert_bytes     = expert_bytes_consensus;
    hdr.slot_bytes       = bsc_expert_pack_align_up(expert_bytes_consensus);
    hdr.total_data_bytes = bsc_expert_pack_total_slots(hdr.n_layers, hdr.n_proj, hdr.n_expert)
                         * hdr.slot_bytes;
    hdr.source_gguf_size = src_size;

    fprintf(stderr,
        "bsc-pack-experts: model layout\n"
        "  source        = %s (%.2f GiB)\n"
        "  output        = %s\n"
        "  n_layers      = %u\n"
        "  n_proj        = %u %s\n"
        "  n_expert      = %u\n"
        "  expert_bytes  = %" PRIu64 "\n"
        "  slot_bytes    = %" PRIu64 " (padding %" PRIu64 " bytes/slot)\n"
        "  total slots   = %" PRIu64 "\n"
        "  pack data     = %" PRIu64 " bytes (%.2f GiB)\n",
        src_path, src_size / (1024.0*1024.0*1024.0),
        out_path.c_str(),
        hdr.n_layers, hdr.n_proj,
        has_gate ? "(down,gate,up)" : "(down,up)",
        hdr.n_expert, hdr.expert_bytes,
        hdr.slot_bytes, hdr.slot_bytes - hdr.expert_bytes,
        bsc_expert_pack_total_slots(hdr.n_layers, hdr.n_proj, hdr.n_expert),
        hdr.total_data_bytes,
        hdr.total_data_bytes / (1024.0*1024.0*1024.0));

    // 4. Open source + destination fds
    int src_fd = open(src_path, O_RDONLY);
    if (src_fd < 0) {
        fprintf(stderr, "bsc-pack-experts: open(%s) failed: %s\n", src_path, strerror(errno));
        return 1;
    }
    // Open RDWR (not WRONLY) so the optional verification pass can pread the
    // slots back through the same fd.
    int dst_fd = open(out_path.c_str(),
                      O_RDWR | O_CREAT | O_TRUNC,
                      S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (dst_fd < 0) {
        fprintf(stderr, "bsc-pack-experts: open(%s) failed: %s\n",
                out_path.c_str(), strerror(errno));
        close(src_fd);
        return 1;
    }

    // 5. Reserve full pack file size up-front (allocates extents, faster writes)
    uint64_t total_file_size = (uint64_t)BSC_EXPERT_PACK_HEADER_SIZE + hdr.total_data_bytes;
    if (ftruncate(dst_fd, (off_t)total_file_size) != 0) {
        fprintf(stderr, "bsc-pack-experts: ftruncate failed: %s\n", strerror(errno));
        close(src_fd); close(dst_fd);
        return 1;
    }

    // 6. Write header (zero-padded to BSC_EXPERT_PACK_HEADER_SIZE)
    {
        std::vector<uint8_t> hdr_buf(BSC_EXPERT_PACK_HEADER_SIZE, 0);
        memcpy(hdr_buf.data(), &hdr, sizeof(hdr));
        if (!pwrite_full(dst_fd, hdr_buf.data(), hdr_buf.size(), 0)) {
            close(src_fd); close(dst_fd);
            return 1;
        }
    }

    // 7. Copy expert slices into aligned slots
    //
    // Mapping from runtime proj_idx to "with-gate" proj index:
    //   has_gate: runtime 0=down, 1=gate, 2=up <-> pwg 0,1,2
    //   no gate:  runtime 0=down, 1=up         <-> pwg 0,2
    //
    // The runtime loader iterates proj_idx in [0, n_proj). We must use the
    // same order so slot indices match.
    auto runtime_proj_to_pwg = [has_gate](int runtime_p) -> int {
        if (has_gate) return runtime_p;       // 0,1,2 -> 0,1,2
        return (runtime_p == 0) ? 0 : 2;      // 0,1   -> 0,2
    };

    std::vector<uint8_t> slice_buf(hdr.expert_bytes);
    uint64_t slots_written = 0;
    uint64_t total_slots   = bsc_expert_pack_total_slots(hdr.n_layers, hdr.n_proj, hdr.n_expert);

    for (uint32_t l = 0; l < hdr.n_layers; l++) {
        for (uint32_t rp = 0; rp < hdr.n_proj; rp++) {
            int pwg = runtime_proj_to_pwg((int)rp);
            const tensor_info * t = expert_map[{(int)l, pwg}];

            for (uint32_t e = 0; e < hdr.n_expert; e++) {
                uint64_t src_off = t->abs_offset + (uint64_t)e * hdr.expert_bytes;

                if (!pread_full(src_fd, slice_buf.data(), hdr.expert_bytes, (off_t)src_off)) {
                    fprintf(stderr,
                        "bsc-pack-experts: failed reading layer=%u proj=%u expert=%u from %s\n",
                        l, rp, e, t->name.c_str());
                    close(src_fd); close(dst_fd);
                    return 1;
                }

                uint64_t slot_idx = bsc_expert_pack_slot_index(l, rp, e, hdr.n_proj, hdr.n_expert);
                uint64_t pack_off = bsc_expert_pack_slot_offset(slot_idx, hdr.slot_bytes);

                if (!pwrite_full(dst_fd, slice_buf.data(), hdr.expert_bytes, (off_t)pack_off)) {
                    fprintf(stderr,
                        "bsc-pack-experts: failed writing slot %" PRIu64 " (layer=%u proj=%u expert=%u)\n",
                        slot_idx, l, rp, e);
                    close(src_fd); close(dst_fd);
                    return 1;
                }

                slots_written++;
                if (slots_written % 256 == 0 || slots_written == total_slots) {
                    fprintf(stderr,
                        "\rbsc-pack-experts: wrote %" PRIu64 "/%" PRIu64 " slots (%.1f%%)",
                        slots_written, total_slots, 100.0 * slots_written / total_slots);
                    fflush(stderr);
                }
            }
        }
    }
    fprintf(stderr, "\n");

    // 8. fsync to make sure data is flushed before verification
    if (fsync(dst_fd) != 0) {
        fprintf(stderr, "bsc-pack-experts: fsync failed: %s\n", strerror(errno));
        close(src_fd); close(dst_fd);
        return 1;
    }

    // 9. Optional verification — read every slot back and compare to source.
    if (verify) {
        fprintf(stderr, "bsc-pack-experts: verifying %" PRIu64 " slots...\n", total_slots);
        std::vector<uint8_t> a(hdr.expert_bytes), b(hdr.expert_bytes);
        uint64_t slots_verified = 0;
        for (uint32_t l = 0; l < hdr.n_layers; l++) {
            for (uint32_t rp = 0; rp < hdr.n_proj; rp++) {
                int pwg = runtime_proj_to_pwg((int)rp);
                const tensor_info * t = expert_map[{(int)l, pwg}];
                for (uint32_t e = 0; e < hdr.n_expert; e++) {
                    uint64_t src_off  = t->abs_offset + (uint64_t)e * hdr.expert_bytes;
                    uint64_t slot_idx = bsc_expert_pack_slot_index(l, rp, e, hdr.n_proj, hdr.n_expert);
                    uint64_t pack_off = bsc_expert_pack_slot_offset(slot_idx, hdr.slot_bytes);

                    if (!pread_full(src_fd, a.data(), hdr.expert_bytes, (off_t)src_off) ||
                        !pread_full(dst_fd, b.data(), hdr.expert_bytes, (off_t)pack_off)) {
                        close(src_fd); close(dst_fd);
                        return 1;
                    }
                    if (memcmp(a.data(), b.data(), hdr.expert_bytes) != 0) {
                        fprintf(stderr,
                            "\nbsc-pack-experts: VERIFICATION MISMATCH at layer=%u proj=%u expert=%u\n",
                            l, rp, e);
                        close(src_fd); close(dst_fd);
                        return 1;
                    }
                    slots_verified++;
                    if (slots_verified % 256 == 0 || slots_verified == total_slots) {
                        fprintf(stderr,
                            "\rbsc-pack-experts: verified %" PRIu64 "/%" PRIu64 " slots (%.1f%%)",
                            slots_verified, total_slots, 100.0 * slots_verified / total_slots);
                        fflush(stderr);
                    }
                }
            }
        }
        fprintf(stderr, "\nbsc-pack-experts: verification PASSED\n");
    }

    close(src_fd);
    close(dst_fd);

    fprintf(stderr, "bsc-pack-experts: done — %s (%.2f GiB)\n",
            out_path.c_str(),
            total_file_size / (1024.0*1024.0*1024.0));
    return 0;
}
