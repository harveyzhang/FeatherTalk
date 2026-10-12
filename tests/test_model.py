import torch

from model import AUDIO_FEATURE_DIM, AUDIO_TOKENS, IMAGE_SIZE, Model


def test_model_contract() -> None:
    model = Model().eval()
    image = torch.zeros(1, 6, IMAGE_SIZE, IMAGE_SIZE)
    audio = torch.zeros(1, AUDIO_TOKENS, AUDIO_FEATURE_DIM)
    with torch.no_grad():
        output = model(image, audio)
    assert output.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_model_rejects_wrong_audio_shape() -> None:
    model = Model().eval()
    image = torch.zeros(1, 6, IMAGE_SIZE, IMAGE_SIZE)
    audio = torch.zeros(1, AUDIO_TOKENS - 1, AUDIO_FEATURE_DIM)
    try:
        model(image, audio)
    except ValueError as error:
        assert "audio must have shape" in str(error)
    else:
        raise AssertionError("wrong audio shape was accepted")


def test_model_configurable_image_size() -> None:
    default = Model().eval()
    for size in (128, 160, 192, 161):
        model = Model(image_size=size).eval()
        model.load_state_dict(default.state_dict(), strict=True)
        assert model.image_size == size
        with torch.no_grad():
            output = model(
                torch.zeros(1, 6, size, size),
                torch.zeros(1, AUDIO_TOKENS, AUDIO_FEATURE_DIM),
            )
        assert output.shape == (1, 3, size, size)


def test_models_validate_image_configuration() -> None:
    from model import MobileOneModel

    for model_class in (Model, MobileOneModel):
        for size in (0, 15, 128.5, True):
            try:
                model_class(image_size=size)
            except ValueError as error:
                assert "image_size" in str(error)
            else:
                raise AssertionError(f"invalid image_size {size!r} was accepted")

        model = model_class(mode="hubert", image_size=128).eval()
        try:
            model(torch.zeros(1, 6, 144, 144), torch.zeros(1, 40, 1024))
        except ValueError as error:
            assert "image must have shape" in str(error)
        else:
            raise AssertionError("wrong image shape was accepted")


def test_model_defaults_match_preprocessing() -> None:
    from face_utils import FACE_BORDER, FACE_CROP_SIZE, FACE_INNER_SIZE
    from model import MobileOneModel

    assert IMAGE_SIZE == FACE_INNER_SIZE == 144
    assert FACE_CROP_SIZE == IMAGE_SIZE + 2 * FACE_BORDER
    assert Model().image_size == MobileOneModel().image_size == IMAGE_SIZE
