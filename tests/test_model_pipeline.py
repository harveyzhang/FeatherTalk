import importlib.util
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import onnx
import torch

from dataset_mouth_roi import TemporalMouthRoiDataset
from face_utils import gather_audio_window, reshape_audio_feat
from inference import load_model, run
from model import (
    MobileOneBlock,
    create_model,
    load_visual_model,
    reparameterize_model,
    visual_export_model,
    visual_model_config,
)
from train import resume_if_any, save_checkpoint


def _dataset(root: Path) -> Path:
    root.mkdir(exist_ok=True)
    (root / "full_body_img").mkdir()
    (root / "landmarks").mkdir()
    landmarks = np.full((110, 2), 80, dtype=np.float32)
    landmarks[1] = (20, 60)
    landmarks[31] = (172, 60)
    landmarks[52] = (90, 20)
    landmarks[90:100] = (75, 105)
    landmarks[100:110] = (115, 130)
    for index in range(3):
        image = np.full((192, 192, 3), 80 + index * 10, dtype=np.uint8)
        assert cv2.imwrite(str(root / "full_body_img" / f"{index}.jpg"), image)
        np.savetxt(root / "landmarks" / f"{index}.lms", landmarks)
    rng = np.random.default_rng(12)
    np.save(root / "aud_hu.npy", rng.standard_normal((3, 2, 1024)).astype(np.float32))
    return root


def test_both_models_train_from_the_common_dataset(tmp_path) -> None:
    root = _dataset(tmp_path / "person")
    for model_type, audio_shape in (("feathertalk", (40, 1024)), ("mobileone", (16, 32, 32))):
        model = create_model(model_type).train()
        dataset = TemporalMouthRoiDataset(
            str(root), audio_half_window=model.audio_half_window,
            raw_hubert_audio=model.raw_hubert_audio,
        )
        images, targets, audio, masks = dataset[0]
        assert images.shape == (2, 6, 144, 144)
        assert audio.shape == (2, *audio_shape)
        prediction = model(images, audio)
        loss = (prediction - targets).abs().mean()
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
        assert masks.shape == (2, 1, 144, 144)


def test_training_entry_selects_and_resumes_both_models(tmp_path) -> None:
    import train

    class TestPerceptualLoss:
        # Exercise CLI wiring without downloading VGG weights for a unit test.
        def __init__(self, criterion, device):
            self.criterion = criterion

        def __call__(self, generated, target):
            return self.criterion(generated, target)

    root = _dataset(tmp_path / "person")
    with patch.object(train, "PerceptualLoss", TestPerceptualLoss), patch.object(
        train, "get_training_device", return_value=torch.device("cpu")
    ):
        for model_type in ("feathertalk", "mobileone"):
            save_dir = tmp_path / model_type
            common_args = [
                "train.py", "--dataset_dir", str(root), "--save_dir", str(save_dir),
                "--batchsize", "1", "--num_workers", "0", "--save_every", "1",
            ]
            with patch("sys.argv", common_args + ["--epochs", "1", "--model", model_type]):
                train.main()
            checkpoint = torch.load(save_dir / "last.pth", map_location="cpu")
            assert checkpoint["epoch"] == 0
            assert checkpoint["model_config"]["model_type"] == model_type
            # No --model on resume: the saved architecture must be selected automatically.
            with patch("sys.argv", common_args + ["--epochs", "2", "--resume", str(save_dir / "last.pth")]):
                train.main()
            checkpoint = torch.load(save_dir / "last.pth", map_location="cpu")
            assert checkpoint["epoch"] == 1
            assert checkpoint["model_config"]["model_type"] == model_type


def test_checkpoint_metadata_resume_and_legacy_detection(tmp_path) -> None:
    for model_type in ("feathertalk", "mobileone"):
        model = create_model(model_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        path = tmp_path / f"{model_type}.pth"
        save_checkpoint(str(path), model, optimizer, epoch=7)
        checkpoint = torch.load(path, map_location="cpu")
        assert checkpoint["model_config"] == visual_model_config(model)
        restored = load_visual_model(str(path))
        assert restored.model_type == model_type
        assert restored.image_size == 144
        assert resume_if_any(str(path), restored, torch.optim.Adam(restored.parameters()), torch.device("cpu")) == 8
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], tensor)
        legacy_path = tmp_path / f"{model_type}_legacy.pth"
        torch.save(model.state_dict(), legacy_path)
        assert load_visual_model(str(legacy_path)).model_type == model_type
        try:
            load_visual_model(str(path), model_type="mobileone" if model_type == "feathertalk" else "feathertalk")
        except ValueError as error:
            assert "--model" in str(error)
        else:
            raise AssertionError("conflicting model selection was accepted")


def test_mobileone_inference_fuses_and_loads_deployment_weights(tmp_path) -> None:
    path = tmp_path / "mobileone.pth"
    model = create_model("mobileone").eval()
    torch.save(model.state_dict(), path)
    restored = load_model(str(path), torch.device("cpu"))
    assert all(module.inference_mode for module in restored.modules() if isinstance(module, MobileOneBlock))
    fused_path = tmp_path / "fused.pth"
    torch.save(restored.state_dict(), fused_path)
    deployed = load_model(str(fused_path), torch.device("cpu"))
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, deployed.state_dict()[name])


def test_mobileone_export_audio_matches_training_at_clip_edges() -> None:
    model = reparameterize_model(create_model("mobileone").eval())
    exported = visual_export_model(model)
    features = np.random.default_rng(42).standard_normal((25, 2, 1024)).astype(np.float32)
    image = torch.randn(1, 6, 144, 144)
    for index in (0, 10, 24):
        native = reshape_audio_feat(gather_audio_window(features, index, half_window=4), raw_hubert=False)[None]
        common = reshape_audio_feat(gather_audio_window(features, index))[None]
        with torch.no_grad():
            torch.testing.assert_close(exported(image, common), model(image, native))


def test_python_video_inference_both_architectures(tmp_path) -> None:
    from argparse import Namespace

    root = _dataset(tmp_path / "person")
    for model_type in ("feathertalk", "mobileone"):
        path = tmp_path / f"{model_type}.pth"
        torch.save(create_model(model_type).state_dict(), path)
        output = tmp_path / f"{model_type}.avi"
        run(Namespace(
            dataset=str(root), audio_feat=str(root / "aud_hu.npy"), checkpoint=str(path),
            save_path=str(output), audio_wav="", model="auto",
        ), device=torch.device("cpu"))
        capture = cv2.VideoCapture(str(output))
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        success, frame = capture.read()
        capture.release()
        assert success and frame.shape == (192, 192, 3)


def test_both_onnx_exports_keep_the_cpp_interface(tmp_path) -> None:
    path = Path(__file__).resolve().parents[1] / "FeatherTalk-CPP" / "tools" / "export_models.py"
    spec = importlib.util.spec_from_file_location("feathertalk_export_models", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    for model_type in ("feathertalk", "mobileone"):
        checkpoint = tmp_path / f"{model_type}.pth"
        torch.save(create_model(model_type).state_dict(), checkpoint)
        output = tmp_path / f"{model_type}.onnx"
        exporter.export_visual(checkpoint, output, 17)
        graph = onnx.load(output).graph
        shape = lambda value: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        assert [(value.name, shape(value)) for value in graph.input] == [
            ("input", [1, 6, 144, 144]), ("audio", [1, 40, 1024]),
        ]
        assert [(value.name, shape(value)) for value in graph.output] == [("output", [1, 3, 144, 144])]
        if model_type == "mobileone":
            # Deployment export must not retain training-time MobileOne branches.
            assert not any("rbr_conv" in initializer.name for initializer in graph.initializer)
