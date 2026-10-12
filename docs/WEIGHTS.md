# Model weights

The preprocessing path uses these small local models:

```text
data_utils/scrfd_2.5g_kps.onnx
data_utils/checkpoint_epoch_335.pth.tar
```

FeatherHuBERT is the project's independently trained lightweight audio encoder,
trained through HuBERT feature distillation. Its generic weights are included at
the repository root and can be shared across identities; deployment does not
require the large HuBERT teacher model:

```text
feather_hubert.pth       generic audio encoder, 3.36M parameters
```

The included model has 3,364,096 parameters (256 channels, 8 temporal blocks).
It takes 16 kHz waveform input and produces 1024-dimensional features at a 20 ms
stride. HuBERT compatibility refers to this interface, not identical feature
values. Training a new identity only requires training its visual model, not
retraining this audio encoder.

The currently published weights were trained only on Chinese speech. Training
remains incomplete due to the author's disk-capacity limits. Developers interested in more complete
training can contact the author through the community group linked at the end
of the main README to request the FeatherHuBERT training code.

Its SHA-256 checksum is:

```text
3b147e66502132bac3ff76b4730c2a83c341c75251d22a28182a5a1b689cfc46
```

Training produces a personalized 144x144 visual checkpoint for each identity.
For C++ inference, convert that checkpoint and the included audio encoder with
`FeatherTalk-CPP/tools/export_models.py` and `convert_mnn_models.sh`.

Do not commit personalized checkpoints, generated ONNX files or MNN files.
Publish approved deployment artifacts as versioned release assets.
