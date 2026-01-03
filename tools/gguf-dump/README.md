# llama-gguf-dump

Extract tensor metadata from GGUF model files for analysis and profiling.

## Purpose

This tool dumps the complete tensor structure of a GGUF model file to CSV format, including:
- Tensor names (e.g., `blk.5.attn_q.weight`)
- File offsets (byte positions in GGUF file)
- Tensor sizes (in bytes)
- Layer IDs (extracted from tensor names)
- Component types (Attention, FFN, Embeddings, etc.)
- Tensor dimensions

## Use Cases

- **Memory Access Analysis**: Correlate tensor accesses with file I/O patterns
- **Model Structure Inspection**: Understand model architecture and parameter layout
- **Optimization**: Identify hot/cold parameters for intelligent caching
- **Research**: Analyze parameter distribution across layers and components

## Usage

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make llama-gguf-dump

# Run
./bin/llama-gguf-dump path/to/model.gguf > model_structure.csv

# Example output (CSV format):
# tensor_name,file_offset,size_bytes,layer_id,component_type,n_dims,dim0,dim1,dim2,dim3
# token_embd.weight,524288,524288000,-1,Token Embeddings,2,32000,4096,0,0
# blk.0.attn_q.weight,524812288,67108864,0,Attention Q,2,4096,4096,0,0
# blk.0.attn_k.weight,591921152,67108864,0,Attention K,2,4096,4096,0,0
# ...
```

## Output Format

CSV with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `tensor_name` | Full tensor name | `blk.5.attn_q.weight` |
| `file_offset` | Byte offset in GGUF file | `524812288` |
| `size_bytes` | Tensor size in bytes | `67108864` |
| `layer_id` | Transformer layer (0-based, -1 for non-layer tensors) | `5` |
| `component_type` | Semantic component type | `Attention Q` |
| `n_dims` | Number of dimensions | `2` |
| `dim0, dim1, dim2, dim3` | Dimension sizes | `4096, 4096, 0, 0` |

## Component Types

The tool automatically categorizes tensors:

- **Token Embeddings**: Input token embedding table
- **Output Projection**: Final output layer
- **Attention Q/K/V**: Query, Key, Value projection matrices
- **Attention Output**: Attention output projection
- **Attention Norm**: Attention normalization weights
- **FFN Up/Down/Gate**: Feed-forward network components
- **FFN Norm**: FFN normalization weights
- **MoE Expert N**: Mixture-of-Experts expert-specific weights (if applicable)

## Integration with Tensor Access Tracing

This tool is designed to work with the tensor access tracing system:

1. **Extract structure**: `llama-gguf-dump model.gguf > structure.csv`
2. **Run traced inference**: `llama-cli -m model.gguf ... > trace.bin`
3. **Correlate accesses**: Join trace logs with structure.csv to map memory accesses to semantic tensor names

## Technical Details

- **Self-contained**: Uses minimal GGUF parsing (no heavy dependencies)
- **Format support**: GGUF version 3 (with backward compatibility warnings)
- **Output**: CSV to stdout, metadata to stderr
- **Performance**: Fast, single-pass file reading

## Example Analysis Workflow

```bash
# 1. Dump model structure
./llama-gguf-dump ~/models/llama-2-7b.gguf > llama2_structure.csv

# 2. Analyze layer distribution
cat llama2_structure.csv | awk -F',' '{print $4}' | sort | uniq -c

# 3. Identify largest tensors
cat llama2_structure.csv | sort -t',' -k3 -n -r | head -10

# 4. Calculate total model size
cat llama2_structure.csv | awk -F',' '{sum+=$3} END {print sum/1024/1024/1024 " GB"}'
```

## See Also

- [Tensor Access Tracing Documentation](../../ggml/include/tensor_trace.h)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
