# Quantized Package Tensor-Data Writer

## Overview

The tensor-data writer extends the metadata-only quantized package format with optional `.npy` tensor payloads. This enables deterministic, local packaging of q4/q8 weights, scales, zeros, and norm weights for use in generation pipelines.

## Package directory layout

```
package_dir/
  package.json         # metadata with tensor file references and checksums
  tensors/
    layers.0.qkv.weight.npy
    layers.0.qkv.scales.npy
    layers.0.input_layernorm.weight.npy
    layers.0.o_proj.weight.npy
    layers.0.o_proj.scales.npy
    ...
    embedding.weight.npy        (optional)
    norm.weight.npy             (optional)
    lm_head.weight.npy          (optional)
```

## Writing tensor data

### From the converter

```python
from models.checkpoint_adapter import CheckpointAdapterConfig, adapter_from_in_memory_tensors
from models.checkpoint_converter import CheckpointConverter, CheckpointConverterConfig
from models.llama_config import tiny_gqa_debug_config

config = tiny_gqa_debug_config()
adapter = adapter_from_in_memory_tensors(config, tensors, ...)
converter = CheckpointConverter(
    adapter,
    CheckpointConverterConfig(bits=4, group_size=32, save_tensor_data=True),
)
package, report = converter.convert("path/to/output/")
```

When `save_tensor_data=True`:
- `output_path` is treated as a directory
- Tensors are written to `output_path/tensors/`
- `package.json` is updated with relative file paths and checksums

### Using QuantizedPackageWriter directly

```python
from models.quantized_package_writer import PackageWriterConfig, QuantizedPackageWriter

writer = QuantizedPackageWriter(PackageWriterConfig(tensor_subdir="tensors"))
report = writer.write_tensors(
    package,                 # QuantizedCheckpointPackage
    layer_packages,          # list[QuantizedLlamaLayerPackage]
    "path/to/output/dir",
    global_tensors={         # optional
        "embedding": embedding_weight,
        "norm": norm_weight,
    },
)
```

### Via CLI

```bash
# Dry-run (count tensors)
python scripts/write_quantized_package.py path/to/package.json --dry-run

# Write synthetic tensor data
python scripts/write_quantized_package.py path/to/package.json --synthetic --seed 42

# Custom output directory
python scripts/write_quantized_package.py path/to/package.json --synthetic --output-dir ./my_package
```

## Reading tensor data

```python
from models.tensor_data_io import load_tensor_npy

weight = load_tensor_npy("path/to/layers.0.qkv.weight.npy")
```

### Checking package tensor data

```python
from models.quantized_package_io import QuantizedCheckpointPackage

package = QuantizedCheckpointPackage.load_json("path/to/package.json")

# Quick check if all tensors have data_file references
has_data = package.has_tensor_data()

# List all referenced tensor files
files = package.tensor_files(base_dir="path/to")

# Validate files exist (and optionally checksums)
issues = package.validate_tensor_files("path/to", check_checksums=True)
```

### With the inspect CLI

```bash
# Check tensor file existence
python scripts/inspect_quantized_package.py path/to/package.json --check-tensor-files

# Also validate checksums
python scripts/inspect_quantized_package.py path/to/package.json --check-tensor-files --check-checksums

# Specify package root for relative paths
python scripts/inspect_quantized_package.py path/to/package.json --check-tensor-files --package-root path/to
```

## API reference

### `models/tensor_data_io.py`

| Function | Description |
|---|---|
| `save_tensor_npy(tensor, path)` | Save tensor as `.npy`, returns `TensorDataInfo` |
| `load_tensor_npy(path)` | Load tensor from `.npy` file |
| `tensor_shape(path)` | Read shape without loading full tensor |
| `tensor_dtype(path)` | Read dtype without loading full tensor |
| `tensor_nbytes(path)` | Read nbytes without loading full tensor |
| `compute_file_checksum(path, algorithm)` | Compute SHA256 (default) checksum |
| `validate_tensor_file(path, ...)` | Validate shape, dtype, and/or checksum |

### `models/quantized_package_writer.py`

| Class | Description |
|---|---|
| `PackageWriterConfig` | Configuration: `tensor_subdir`, `checksum_algorithm` |
| `PackageWriterReport` | Result: files written, errors, bytes, tensor count |
| `QuantizedPackageWriter` | Main writer: `write_tensors()`, `write_package()` |

### Methods on `QuantizedCheckpointPackage`

| Method | Description |
|---|---|
| `has_tensor_data()` | Check if all tensor data_file fields are set |
| `tensor_files(base_dir)` | Return dict of all referenced tensor file paths |
| `validate_tensor_files(base_dir, check_checksums)` | Validate existence (and optionally checksums) |

## Design notes

- Only `.npy` is guaranteed (numpy is already a runtime dependency).
- Checksums are SHA256 by default.
- Tensor file paths in metadata are relative (portable across machines).
- Metadata-only packages remain the default; tensor-data writing is opt-in via `save_tensor_data=True`.
- Deterministic output: same config + same seed = identical package.
