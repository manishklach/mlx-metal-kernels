from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checkpoint_adapter import CheckpointAdapter, CheckpointAdapterConfig
from .checkpoint_quantizer import CheckpointQuantizer
from .llama_config import LlamaLikeConfig
from .quantize_weights import QuantizationConfig
from .quantized_layer_package import QuantizedLlamaLayerPackage
from .quantized_package_io import QuantizedCheckpointPackage, package_from_quantized_layers
from .tensor_store import ManifestTensorStore


@dataclass
class CheckpointConverterConfig:
    bits: int = 4
    group_size: int = 32
    layers: list[int] | None = None
    fuse_qkv: bool = True
    save_tensor_data: bool = False
    output_format: str = "json_metadata"
    allow_partial: bool = False
    symmetric: bool = True
    with_zeros: bool = False

    def validate(self) -> CheckpointConverterConfig:
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        valid_formats = ("json_metadata",)
        if self.output_format not in valid_formats:
            raise ValueError(
                f"output_format must be one of {valid_formats}, got {self.output_format!r}"
            )
        return self


@dataclass
class CheckpointConverterReport:
    ok: bool
    layers_converted: list[int] = field(default_factory=list)
    output_path: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tensor_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


class CheckpointConverter:
    def __init__(
        self,
        checkpoint_adapter: CheckpointAdapter,
        converter_config: CheckpointConverterConfig | None = None,
    ):
        self.checkpoint_adapter = checkpoint_adapter
        self.converter_config = (converter_config or CheckpointConverterConfig()).validate()
        self._quant_config = QuantizationConfig(
            bits=self.converter_config.bits,
            group_size=self.converter_config.group_size,
            symmetric=self.converter_config.symmetric,
            with_zeros=self.converter_config.with_zeros,
        )
        self._quantizer = CheckpointQuantizer(self.checkpoint_adapter, self._quant_config)
        self._warnings: list[str] = []

    def config(self) -> LlamaLikeConfig:
        return self.checkpoint_adapter.config

    def _resolve_layer_indices(self, layer_indices: list[int] | None = None) -> list[int]:
        config = self.config()
        if layer_indices is not None:
            for idx in layer_indices:
                if idx < 0 or idx >= config.num_hidden_layers:
                    raise ValueError(
                        f"layer index {idx} out of range [0, {config.num_hidden_layers})"
                    )
            return sorted(set(layer_indices))
        if self.converter_config.layers is not None:
            for idx in self.converter_config.layers:
                if idx < 0 or idx >= config.num_hidden_layers:
                    raise ValueError(
                        f"configured layer index {idx} out of range [0, {config.num_hidden_layers})"
                    )
            return sorted(set(self.converter_config.layers))
        return list(range(config.num_hidden_layers))

    def convert_layer(self, layer_idx: int) -> QuantizedLlamaLayerPackage:
        return self._quantizer.quantize_layer(layer_idx)

    def convert_layers(
        self, layer_indices: list[int] | None = None
    ) -> list[QuantizedLlamaLayerPackage]:
        indices = self._resolve_layer_indices(layer_indices)
        packages: list[QuantizedLlamaLayerPackage] = []
        for idx in indices:
            try:
                packages.append(self.convert_layer(idx))
            except NotImplementedError:
                self._warnings.append(
                    f"layer {idx}: tensor data not available (manifest-only store); skipping"
                )
        return packages

    def build_package(
        self, quantized_layers: list[QuantizedLlamaLayerPackage]
    ) -> QuantizedCheckpointPackage:
        config = self.config()
        return package_from_quantized_layers(
            config,
            quantized_layers,
            bits=self.converter_config.bits,
            group_size=self.converter_config.group_size,
            model_type=config.model_type,
            symmetric=self.converter_config.symmetric,
            with_zeros=self.converter_config.with_zeros,
            metadata={"converter_config": self.converter_config.__dict__},
        )

    def save_package(
        self,
        package: QuantizedCheckpointPackage,
        output_path: str | Path,
        quantized_layers: list[QuantizedLlamaLayerPackage] | None = None,
        *,
        global_tensors: dict[str, Any] | None = None,
    ) -> str:
        if self.converter_config.save_tensor_data:
            from .quantized_package_writer import PackageWriterConfig, QuantizedPackageWriter

            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            writer = QuantizedPackageWriter(
                PackageWriterConfig(tensor_subdir="tensors")
            )
            writer.write_tensors(
                package,
                quantized_layers or [],
                output_dir,
                global_tensors=global_tensors,
            )
            return str(output_dir / "package.json")
        package.save_json(str(output_path))
        return str(output_path)

    def convert(
        self, output_path: str | Path | None = None
    ) -> tuple[QuantizedCheckpointPackage | None, CheckpointConverterReport]:
        config = self.config()
        all_indices = self._resolve_layer_indices()
        errors: list[str] = []
        layers_converted: list[int] = []
        package: QuantizedCheckpointPackage | None = None
        tensor_count = 0

        is_manifest_only = isinstance(self.checkpoint_adapter.tensor_store, ManifestTensorStore)

        if is_manifest_only:
            if self.converter_config.layers is not None:
                all_indices = self.converter_config.layers
            plan = {
                "model_type": config.model_type,
                "layers_available": config.num_hidden_layers,
                "layers_selected": all_indices,
                "bits": self.converter_config.bits,
                "group_size": self.converter_config.group_size,
                "fuse_qkv": self.converter_config.fuse_qkv,
                "adapter_description": self.checkpoint_adapter.describe(),
                "note": "dry-run plan only; manifest store cannot quantize tensor data",
            }
            report = CheckpointConverterReport(
                ok=True,
                layers_converted=[],
                output_path=str(output_path) if output_path else None,
                tensor_count=0,
                metadata={"dry_run_plan": plan},
            )
            if output_path:
                self._write_plan_json(plan, output_path)
            if self._warnings:
                report.warnings = list(self._warnings)
            return None, report

        quantized_packages = self.convert_layers(all_indices)
        if not quantized_packages:
            errors.append("no layers were converted; check tensor store and adapter configuration")
            report = CheckpointConverterReport(ok=False, errors=errors, warnings=list(self._warnings))
            return None, report

        package = self.build_package(quantized_packages)
        package.validate(allow_partial=self.converter_config.allow_partial)
        tensor_count = package.tensor_count()
        layers_converted = [pkg.layer_idx for pkg in quantized_packages]

        if output_path:
            actual_path = self.save_package(
                package, output_path, quantized_layers=quantized_packages
            )

        report = CheckpointConverterReport(
            ok=True,
            layers_converted=layers_converted,
            output_path=str(output_path) if output_path else None,
            tensor_count=tensor_count,
            metadata={"config_keys": list(config.to_dict().keys())},
            warnings=list(self._warnings),
        )
        return package, report

    def _write_plan_json(self, plan: dict[str, Any], output_path: str | Path) -> None:
        import json

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(plan, indent=2), encoding="utf-8")
