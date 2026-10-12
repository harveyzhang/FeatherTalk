#!/usr/bin/env python3
"""Export FeatherHuBERT and the released 144x144 model to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_utils.feather_hubert.feather_hubert import load_feather_hubert
from model import (
    AUDIO_FEATURE_DIM, AUDIO_TOKENS, IMAGE_SIZE, MODEL_TYPES,
    load_visual_model, visual_export_model,
)


def export_feather(checkpoint: Path, output: Path, opset: int) -> None:
    model = load_feather_hubert(checkpoint, device=torch.device("cpu")).eval()
    waveform = torch.zeros((1, 16000), dtype=torch.float32)
    torch.onnx.export(
        model,
        waveform,
        output,
        input_names=["waveform"],
        output_names=["hidden"],
        dynamic_axes={"waveform": {1: "samples"}, "hidden": {1: "tokens"}},
        opset_version=opset,
        export_params=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(output))
    print(f"[export] FeatherHuBERT -> {output}")


def export_visual(
    checkpoint: Path, output: Path, opset: int, model_type: str = "auto"
) -> None:
    native_model = load_visual_model(str(checkpoint), model_type=model_type, reparameterize=True)
    if native_model.image_size != IMAGE_SIZE:
        raise ValueError(f"The C++ export interface requires {IMAGE_SIZE}x{IMAGE_SIZE} images")
    model = visual_export_model(native_model)
    image = torch.zeros((1, 6, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    audio = torch.zeros((1, AUDIO_TOKENS, AUDIO_FEATURE_DIM), dtype=torch.float32)
    torch.onnx.export(
        model,
        (image, audio),
        output,
        input_names=["input", "audio"],
        output_names=["output"],
        opset_version=opset,
        export_params=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(output))
    print(f"[export] {native_model.model_type} visual model -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feather-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=("auto", *MODEL_TYPES), default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "FeatherTalk-CPP" / "models"
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_feather(
        args.feather_checkpoint, args.output_dir / "feather_hubert.onnx", args.opset
    )
    export_visual(
        args.visual_checkpoint, args.output_dir / "feathertalk_144.onnx", args.opset, args.model
    )


if __name__ == "__main__":
    main()
