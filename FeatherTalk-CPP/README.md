## 以下是一个大致的使用说明，借助AI，你一定可以快速把代码迁移至任何平台,ENJOY 🎉

# FeatherTalk C++ / MNN

This directory contains standalone, offline C++ inference for FeatherTalk and
MobileOne visual models. Runtime inference depends on MNN and FFmpeg, but not Python,
PyTorch, OpenCV or ONNX Runtime.

## Fixed model interface

```text
FeatherHuBERT
  waveform: [1, samples]
  hidden:   [1, tokens, 1024]

Visual model (FeatherTalk or exported MobileOne)
  input:  [1, 6, 144, 144]
  audio:  [1, 40, 1024]
  output: [1, 3, 144, 144]
```

The runner extracts audio features once for the complete WAV. Each video frame
uses the corresponding 40-token slice. The generated 144x144 patch directly
replaces the inner face region.

Both FeatherTalk and MobileOne export this same interface. MobileOne is fused
automatically during export; its input adapter selects the central 8 video-frame
steps (16 tokens) from the common 20-step window and reshapes them for its legacy
audio branch. The C++ runner therefore needs no architecture-specific changes.
Native Python MobileOne uses `[B,16,32,32]` audio, but exported ONNX/MNN models
always use the public `[1,40,1024]` interface shown above.

## Export and convert models

From the repository root:

```bash
python FeatherTalk-CPP/tools/export_models.py \
  --feather-checkpoint ./feather_hubert.pth \
  --visual-checkpoint /path/to/visual_model.pth \
  --model auto

cd FeatherTalk-CPP
./setup_mnn_macos.sh
./convert_mnn_models.sh
```

This produces:

```text
models/feather_hubert.mnn
models/feathertalk_144_fp16.mnn
```

ONNX is used only as an offline conversion format.
The visual architecture is detected from checkpoint metadata or legacy weights
for these two HuBERT-compatible architectures.
Optionally pass `--model feathertalk` or `--model mobileone` to the exporter to
verify the selected architecture; a mismatch is an error. MobileOne is
reparameterized automatically, without modifying the source checkpoint.
Continue training from that original, unfused checkpoint.

The output filenames remain the same for both architectures. Exporting another
model into the same directory replaces the previous exported files; keep separate
copies when comparing models.

## Build on Apple Silicon

```bash
brew install ffmpeg
./setup_macos.sh
./setup_mnn_macos.sh
./build_macos.sh
```

Generated binaries:

```text
bin/feathertalk_mnn
bin/benchmark_mnn
```

For integration into another C++ project, link the `feathertalk_runtime` CMake
target and include `include/feathertalk_api.h`:

```cpp
feathertalk::OfflineOptions options;
options.feather_model = "/models/feather_hubert.mnn";
options.visual_model = "/models/feathertalk_144_fp16.mnn";
options.dataset = "/resources/person";
options.audio_wav = "/audio/input.wav";
options.output_mp4 = "/output/result.mp4";

std::string error;
if (!feathertalk::RunOffline(options, &error)) {
  // Handle error.
}
```

## Run

The resource directory must contain matching numeric image and landmark files:

```text
person_resource/
  full_body_img/0.jpg
  full_body_img/1.jpg
  landmarks/0.lms
  landmarks/1.lms
```

```bash
./bin/feathertalk_mnn \
  --feather-model models/feather_hubert.mnn \
  --unet-model models/feathertalk_144_fp16.mnn \
  --dataset /path/to/person_resource \
  --audio /path/to/audio_16k_mono.wav \
  --output result.mp4 \
  --backend cpu \
  --precision low \
  --threads 1
```

Useful optional flags:

```text
--max-frames N
--frames-dir PATH
--video-crf N
--profile
```

## Benchmark

```bash
./bin/benchmark_mnn \
  --feather-model models/feather_hubert.mnn \
  --unet-model models/feathertalk_144_fp16.mnn \
  --backend cpu --precision low --threads 1 \
  --offline-seconds 20 --audio-iterations 5 --audio-warmup 1
```

The benchmark uses synthetic inputs and reports UNet time plus two separate,
model-only audio cost scenarios:

- Offline: extract a complete 20-second waveform, then divide audio time by the
  number of usable video frames. `--offline-seconds N` changes the duration;
  `--audio-iterations N` / `--audio-warmup N` control this longer measurement.
- Window recompute: extract an 800 ms audio context again for every 40 ms video
  frame. An extra 80 samples (5 ms) are needed by the valid-convolution frontend
  to produce all 40 tokens. This measures recompute cost, not a complete streaming
  implementation, and does not claim feature equivalence to offline extraction.

Offline total = whole-waveform audio time / usable video frames + UNet time.
Window-recompute total = one 800 ms window extraction + one UNet inference.
`--iterations` / `--warmup` control the UNet and window measurements; the separate
`--audio-iterations` / `--audio-warmup` options control whole-waveform measurements.

Times include input/output tensor transfers but exclude normalization, file I/O,
compositing, video encoding and look-ahead latency. The former 720-sample timing
is no longer used as a per-video-frame audio estimate.
The supplied `RunOffline` API still accepts a complete WAV, not live PCM chunks.

Generated models, MNN source/build trees and binaries are intentionally ignored
by Git.
