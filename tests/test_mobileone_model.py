import copy

import torch

from model import IMAGE_SIZE, MobileOneBlock, MobileOneModel, reparameterize_model


def test_mobileone_hubert_contract() -> None:
    model = MobileOneModel(6, "hubert").eval()
    image = torch.randn(1, 6, IMAGE_SIZE, IMAGE_SIZE)
    audio = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        output = model(image, audio)
    assert output.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_mobileone_wenet_contract() -> None:
    model = MobileOneModel(6, "wenet").eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 6, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(1, 128, 16, 32))
    assert output.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_mobileone_configurable_image_size() -> None:
    for mode, audio_shape in (("hubert", (1, 16, 32, 32)), ("wenet", (1, 128, 16, 32))):
        default = MobileOneModel(6, mode).eval()
        for size in (128, 144, 160, 192, 161):
            model = MobileOneModel(6, mode, image_size=size).eval()
            model.load_state_dict(default.state_dict(), strict=True)
            assert model.image_size == size
            with torch.no_grad():
                output = model(torch.zeros(1, 6, size, size), torch.zeros(*audio_shape))
            assert output.shape == (1, 3, size, size)


def test_mobileone_configured_size_reparameterization() -> None:
    model = MobileOneModel(6, "hubert", image_size=128).eval()
    image = torch.randn(1, 6, 128, 128)
    audio = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        expected = model(image, audio)
        fused = reparameterize_model(copy.deepcopy(model))
        torch.testing.assert_close(fused(image, audio), expected, rtol=1e-4, atol=1e-5)


def test_mobileone_reparameterization_preserves_output() -> None:
    torch.manual_seed(7)
    model = MobileOneModel(6, "hubert").eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-0.1, 0.1)
                module.running_var.uniform_(0.5, 1.5)
    image = torch.randn(1, 6, IMAGE_SIZE, IMAGE_SIZE)
    audio = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        expected = model(image, audio)
        fused = reparameterize_model(copy.deepcopy(model))
        actual = fused(image, audio)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert all(
        module.inference_mode
        for module in fused.modules()
        if isinstance(module, MobileOneBlock)
    )
    assert reparameterize_model(fused) is fused

    restored = reparameterize_model(MobileOneModel(6, "hubert").eval())
    restored.load_state_dict(fused.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(image, audio), actual)
