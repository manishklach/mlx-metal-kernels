from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .quantized_layer_package import QuantizedLlamaLayerPackage
from .quantized_package_io import QuantizedCheckpointPackage, QuantizedTensorMetadata
from .tensor_data_io import compute_file_checksum, save_tensor_npy


@dataclass
class PackageWriterConfig:
    tensor_subdir: str = "tensors"
    checksum_algorithm: str = "sha256"

    def validate(self) -> PackageWriterConfig:
        if not self.tensor_subdir:
            raise ValueError("tensor_subdir must be non-empty")
        if self.checksum_algorithm not in ("sha256", "sha1", "md5"):
            raise ValueError(
                f"checksum_algorithm must be one of sha256, sha1, md5, got {self.checksum_algorithm!r}"
            )
        return self


@dataclass
class PackageWriterReport:
    ok: bool
    files_written: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tensor_count: int = 0
    total_bytes: int = 0
    package_path: str | None = None

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


class QuantizedPackageWriter:
    def __init__(self, config: PackageWriterConfig | None = None):
        self._config = (config or PackageWriterConfig()).validate()

    def config(self) -> PackageWriterConfig:
        return self._config

    def _tensor_rel_path(self, layer_idx: int | None, name: str, suffix: str) -> str:
        if layer_idx is not None:
            return f"layers.{layer_idx}.{name}{suffix}"
        return f"{name}{suffix}"

    def write_tensors(
        self,
        package: QuantizedCheckpointPackage,
        layer_packages: list[QuantizedLlamaLayerPackage],
        output_dir: str | Path,
        *,
        global_tensors: dict[str, Any] | None = None,
    ) -> PackageWriterReport:
        output_path = Path(output_dir)
        tensor_dir = output_path / self._config.tensor_subdir
        files_written: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []
        total_bytes = 0
        tensor_count = 0

        layer_map = {pkg.layer_idx: pkg for pkg in layer_packages}

        for layer_meta in package.layers:
            layer_pkg = layer_map.get(layer_meta.layer_idx)
            if layer_pkg is None:
                errors.append(
                    f"layer {layer_meta.layer_idx}: no QuantizedLlamaLayerPackage provided"
                )
                continue
            for key, tensor_meta in layer_meta.tensors.items():
                try:
                    info = self._write_layer_tensor(
                        tensor_dir,
                        layer_meta.layer_idx,
                        key,
                        tensor_meta,
                        layer_pkg,
                    )
                    files_written.append(info.file_path)
                    total_bytes += info.nbytes
                    tensor_count += 1
                except Exception as exc:
                    errors.append(
                        f"layer {layer_meta.layer_idx}, tensor {key}: {exc}"
                    )

        if global_tensors:
            for name, tensor_data in global_tensors.items():
                try:
                    tensor_meta = package.global_tensors.get(name)
                    suffix = ".weight.npy" if name != "norm" else ".weight.npy"
                    rel_path = self._tensor_rel_path(None, name, suffix)
                    full_path = tensor_dir / rel_path
                    parent = Path(full_path).parent
                    info = save_tensor_npy(tensor_data, full_path)
                    if tensor_meta:
                        tensor_meta.data_file = str(rel_path)
                        tensor_meta.checksum = info.checksum
                    files_written.append(info.file_path)
                    total_bytes += info.nbytes
                    tensor_count += 1
                except Exception as exc:
                    errors.append(f"global tensor {name}: {exc}")

        if not errors:
            package.save_json(output_path / "package.json")
            package_path = str(output_path / "package.json")
        else:
            package_path = str(output_path / "package.json") if files_written else None

        return PackageWriterReport(
            ok=not errors,
            files_written=files_written,
            errors=errors,
            warnings=warnings,
            tensor_count=tensor_count,
            total_bytes=total_bytes,
            package_path=package_path,
        )

    def _write_layer_tensor(
        self,
        tensor_dir: Path,
        layer_idx: int,
        key: str,
        tensor_meta: QuantizedTensorMetadata,
        layer_pkg: QuantizedLlamaLayerPackage,
    ):
        if tensor_meta.role in ("norm", "final_norm"):
            return self._write_norm_tensor(tensor_dir, layer_idx, key, tensor_meta, layer_pkg)

        linear_pkg = getattr(layer_pkg, key, None)
        if linear_pkg is None:
            raise ValueError(f"layer {layer_idx}: no attribute {key} on QuantizedLlamaLayerPackage")

        rel_weight = self._tensor_rel_path(layer_idx, key, ".weight.npy")
        rel_scales = self._tensor_rel_path(layer_idx, key, ".scales.npy")
        full_weight = tensor_dir / rel_weight
        full_scales = tensor_dir / rel_scales

        weight_info = save_tensor_npy(linear_pkg.weight, full_weight)
        scales_info = save_tensor_npy(linear_pkg.scales, full_scales)

        tensor_meta.data_file = str(rel_weight)
        tensor_meta.scales_file = str(rel_scales)
        tensor_meta.checksum = weight_info.checksum

        if linear_pkg.zeros is not None:
            rel_zeros = self._tensor_rel_path(layer_idx, key, ".zeros.npy")
            full_zeros = tensor_dir / rel_zeros
            zeros_info = save_tensor_npy(linear_pkg.zeros, full_zeros)
            tensor_meta.zeros_file = str(rel_zeros)

        return TensorWriteInfo(
            file_path=str(full_weight),
            nbytes=weight_info.nbytes,
        )

    def _write_norm_tensor(
        self,
        tensor_dir: Path,
        layer_idx: int,
        key: str,
        tensor_meta: QuantizedTensorMetadata,
        layer_pkg: QuantizedLlamaLayerPackage,
    ):
        norm_attr = f"{key}_weight" if not key.endswith("_weight") else key
        weight = getattr(layer_pkg, norm_attr, None)
        if weight is None:
            raise ValueError(f"layer {layer_idx}: no weight attribute for norm {key}")
        rel_path = self._tensor_rel_path(layer_idx, key, ".weight.npy")
        full_path = tensor_dir / rel_path
        info = save_tensor_npy(weight, full_path)
        tensor_meta.data_file = str(rel_path)
        tensor_meta.checksum = info.checksum
        return TensorWriteInfo(file_path=str(full_path), nbytes=info.nbytes)

    def write_package(
        self,
        package: QuantizedCheckpointPackage,
        layer_packages: list[QuantizedLlamaLayerPackage],
        output_dir: str | Path,
        *,
        global_tensors: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> PackageWriterReport:
        if dry_run:
            tensor_count = sum(len(layer.tensors) for layer in package.layers)
            if global_tensors:
                tensor_count += len(global_tensors)
            return PackageWriterReport(
                ok=True,
                tensor_count=tensor_count,
                package_path=str(Path(output_dir) / "package.json"),
            )
        return self.write_tensors(
            package, layer_packages, output_dir, global_tensors=global_tensors
        )


@dataclass
class TensorWriteInfo:
    file_path: str
    nbytes: int
