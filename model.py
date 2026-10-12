"""FeatherTalk UNets with configurable square image sizes."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.base_module import MobileOneBlock


IMAGE_SIZE = 144
AUDIO_TOKENS = 40
AUDIO_FEATURE_DIM = 1024


def _validate_image_size(image_size: int) -> int:
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 16:
        raise ValueError("image_size must be an integer of at least 16 pixels")
    return image_size


def _validate_image_input(
    image: torch.Tensor, n_channels: int, image_size: int
) -> None:
    if image.ndim != 4 or image.shape[1:] != (n_channels, image_size, image_size):
        raise ValueError(
            f"image must have shape [B, {n_channels}, {image_size}, {image_size}], "
            f"got {tuple(image.shape)}"
        )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        use_res_connect: bool,
        expand_ratio: int = 6,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        self.use_res_connect = use_res_connect
        hidden_dim = inp * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride,
                1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        return x + output if self.use_res_connect else output


class DoubleConvDW(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.double_conv = nn.Sequential(
            InvertedResidual(
                in_channels,
                out_channels,
                stride=stride,
                use_res_connect=False,
                expand_ratio=2,
            ),
            InvertedResidual(
                out_channels,
                out_channels,
                stride=1,
                use_res_connect=True,
                expand_ratio=2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class InConvDw(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.inconv = InvertedResidual(
            in_channels,
            out_channels,
            stride=1,
            use_res_connect=False,
            expand_ratio=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inconv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = DoubleConvDW(in_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.shape[2] - x1.shape[2]
        diff_x = x2.shape[3] - x1.shape[3]
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        return self.conv(torch.cat([x1, x2], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalConvBlock(nn.Module):
    """Residual depthwise-separable temporal block over ``[B, C, T]``."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.point_bn = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.bn(self.depthwise(x)))
        x = self.point_bn(self.pointwise(x))
        return self.activation(x + residual)


class AudioAdapter(nn.Module):
    """Convert 40 HuBERT-compatible tokens into 20 visual-frame steps."""

    input_tokens = AUDIO_TOKENS
    output_steps = 20
    input_dim = AUDIO_FEATURE_DIM
    output_dim = 512

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(self.input_dim * 2)
        self.input_projection = nn.Conv1d(
            self.input_dim * 2, hidden_dim, kernel_size=1
        )
        self.blocks = nn.Sequential(
            TemporalConvBlock(hidden_dim, dilation=1),
            TemporalConvBlock(hidden_dim, dilation=2),
            TemporalConvBlock(hidden_dim, dilation=4),
            TemporalConvBlock(hidden_dim, dilation=8),
        )
        self.output_projection = nn.Conv1d(
            hidden_dim, self.output_dim, kernel_size=1
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[1:] != (
            self.input_tokens,
            self.input_dim,
        ):
            raise ValueError(
                f"audio must have shape [B, {self.input_tokens}, {self.input_dim}], "
                f"got {tuple(audio.shape)}"
            )
        batch = audio.shape[0]
        x = audio.reshape(batch, self.output_steps, self.input_dim * 2)
        x = self.norm(x).transpose(1, 2)
        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.output_projection(x)
        return x.transpose(1, 2).contiguous()


class CenteredAudioContext(nn.Module):
    """Combine the current audio step with a learned local context."""

    def __init__(self, audio_dim: int, center_index: int = 10, radius: int = 2):
        super().__init__()
        self.audio_dim = audio_dim
        self.center_index = center_index
        self.radius = radius
        self.score = nn.Linear(audio_dim, 1)

    @property
    def output_dim(self) -> int:
        return self.audio_dim * 2

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[-1] != self.audio_dim:
            raise ValueError(
                f"expected audio [B, T, {self.audio_dim}], got {tuple(audio.shape)}"
            )
        center = min(self.center_index, audio.shape[1] - 1)
        start = max(0, center - self.radius)
        end = min(audio.shape[1], center + self.radius + 1)
        local = audio[:, start:end]
        weights = torch.softmax(self.score(local).squeeze(-1), dim=1)
        local_context = torch.sum(local * weights.unsqueeze(-1), dim=1)
        return torch.cat([audio[:, center], local_context], dim=-1)


class SpatialAudioFusion(nn.Module):
    """Use each visual location to attend over the aligned audio sequence."""

    def __init__(
        self,
        visual_channels: int,
        audio_dim: int = 512,
        attention_dim: int = 128,
        num_heads: int = 4,
        center_index: int = 10,
        max_audio_steps: int = 20,
    ):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.visual_norm = nn.LayerNorm(visual_channels)
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.query = nn.Linear(visual_channels, attention_dim)
        self.key = nn.Linear(audio_dim, attention_dim)
        self.value = nn.Linear(audio_dim, visual_channels)
        self.out = nn.Linear(visual_channels, visual_channels)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.center_index = center_index
        self.max_audio_steps = max_audio_steps
        self.audio_position = nn.Parameter(torch.zeros(max_audio_steps, audio_dim))
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, max_audio_steps))
        with torch.no_grad():
            positions = torch.arange(max_audio_steps, dtype=torch.float32)
            self.relative_bias.copy_(
                -0.12 * (positions - center_index).abs().unsqueeze(0)
            )
        self.residual_gate = nn.Parameter(torch.tensor(-1.5))

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if visual.ndim != 4 or audio.ndim != 3:
            raise ValueError("expected visual [B,C,H,W] and audio [B,T,D]")
        batch, channels, height, width = visual.shape
        if audio.shape[1] > self.max_audio_steps:
            raise ValueError("audio sequence is longer than the configured context")

        tokens = visual.flatten(2).transpose(1, 2)
        audio = audio + self.audio_position[: audio.shape[1]].unsqueeze(0)
        q = self.query(self.visual_norm(tokens))
        k = self.key(self.audio_norm(audio))
        v = self.value(self.audio_norm(audio))

        q = q.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(
            batch, -1, self.num_heads, channels // self.num_heads
        ).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        logits = logits + self.relative_bias[:, : audio.shape[1]].view(
            1, self.num_heads, 1, audio.shape[1]
        )
        weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(batch, -1, channels)
        context = self.out(context)
        fused = tokens + torch.sigmoid(self.residual_gate) * context
        return fused.transpose(1, 2).reshape(batch, channels, height, width)


class Model(nn.Module):
    """Current FeatherTalk visual model.

    Inputs:
        image: ``[B, n_channels, image_size, image_size]`` (default 144x144)
        audio: ``[B, 40, 1024]``
    Output:
        generated face patch ``[B, 3, image_size, image_size]``
    """

    model_type = "feathertalk"
    image_size = IMAGE_SIZE
    audio_half_window = 10
    raw_hubert_audio = True
    visual_channels = (24, 32, 64, 128, 256)
    audio_dim = 512
    center_audio_index = 10

    def __init__(
        self, n_channels: int = 6, mode: str = "hubert", image_size: int = IMAGE_SIZE
    ):
        super().__init__()
        if mode != "hubert":
            raise ValueError("FeatherTalk requires HuBERT-compatible audio features")

        x1_ch, x2_ch, x3_ch, x4_ch, x5_ch = self.visual_channels
        self.n_channels = n_channels
        self.mode = mode
        self.image_size = _validate_image_size(image_size)
        self.inc = InConvDw(n_channels, x1_ch)
        self.down1 = Down(x1_ch, x2_ch)
        self.down2 = Down(x2_ch, x3_ch)
        self.down3 = Down(x3_ch, x4_ch)
        self.down4 = Down(x4_ch, x5_ch)

        self.audio_model = AudioAdapter(hidden_dim=512)
        self.spatial_x3 = SpatialAudioFusion(x3_ch, self.audio_dim)
        self.spatial_x4 = SpatialAudioFusion(x4_ch, self.audio_dim)
        self.spatial_x5 = SpatialAudioFusion(x5_ch, self.audio_dim)
        self.audio_context = CenteredAudioContext(
            self.audio_dim, center_index=self.center_audio_index, radius=2
        )
        self.audio_to_x5 = nn.Linear(self.audio_context.output_dim, x5_ch)
        self.fuse_conv = DoubleConvDW(x5_ch * 2, x5_ch, stride=1)

        self.up1 = Up(x5_ch + x4_ch, x4_ch)
        self.up2 = Up(x4_ch + x3_ch, x3_ch)
        self.up3 = Up(x3_ch + x2_ch, x2_ch)
        self.up4 = Up(x2_ch + x1_ch, x1_ch)
        self.outc = OutConv(x1_ch, 3)

    def encode_image(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _validate_image_input(x, self.n_channels, self.image_size)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def decode_features(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        audio_sequence = self.audio_model(audio)
        x3 = self.spatial_x3(x3, audio_sequence)
        x4 = self.spatial_x4(x4, audio_sequence)
        x5 = self.spatial_x5(x5, audio_sequence)

        audio_context = self.audio_context(audio_sequence)
        audio_map = self.audio_to_x5(audio_context).unsqueeze(-1).unsqueeze(-1)
        audio_map = audio_map.expand(-1, -1, x5.shape[-2], x5.shape[-1])
        x5 = self.fuse_conv(torch.cat([x5, audio_map], dim=1))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.decode_features(*self.encode_image(image), audio)


def load_checkpoint_state(
    checkpoint_path: str, device: torch.device | str = "cpu"
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


_MOBILEONE_CHANNELS = (32, 64, 128, 256, 512)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int] = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileOneSeparableBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_conv_branches: int = 2,
        use_res_connect: bool = False,
    ):
        super().__init__()
        self.use_res_connect = use_res_connect and stride == 1 and in_channels == out_channels
        self.depthwise = MobileOneBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            num_conv_branches=num_conv_branches,
        )
        self.pointwise = MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            num_conv_branches=num_conv_branches,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.pointwise(self.depthwise(x))
        return x + output if self.use_res_connect else output


class DoubleConvMobileOne(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.double_conv = nn.Sequential(
            MobileOneSeparableBlock(in_channels, out_channels, stride=stride),
            MobileOneSeparableBlock(out_channels, out_channels, stride=1, use_res_connect=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class InConvMobileOne(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.inconv = MobileOneSeparableBlock(in_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inconv(x)


class MobileOneDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = DoubleConvMobileOne(in_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class MobileOneUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvMobileOne(in_channels, out_channels, stride=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.shape[2] - x1.shape[2]
        diff_x = x2.shape[3] - x1.shape[3]
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x1, x2], dim=1))


class MobileOneAudioConvWenet(nn.Module):
    def __init__(self, channels=_MOBILEONE_CHANNELS):
        super().__init__()
        ch = channels
        self.conv1 = MobileOneSeparableBlock(ch[2], ch[3], stride=1)
        self.conv2 = MobileOneSeparableBlock(ch[3], ch[3], stride=1, use_res_connect=True)
        self.conv3 = MobileOneBlock(ch[3], ch[3], kernel_size=3, padding=1, stride=(1, 2), num_conv_branches=2)
        self.conv4 = MobileOneSeparableBlock(ch[3], ch[3], stride=1, use_res_connect=True)
        self.conv5 = ConvBNAct(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.conv6 = MobileOneSeparableBlock(ch[4], ch[4], stride=1, use_res_connect=True)
        self.conv7 = MobileOneSeparableBlock(ch[4], ch[4], stride=1, use_res_connect=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.conv7(x)


class MobileOneAudioConvHubert(nn.Module):
    def __init__(self, channels=_MOBILEONE_CHANNELS):
        super().__init__()
        ch = channels
        self.conv1 = MobileOneSeparableBlock(16, ch[1], stride=1)
        self.conv2 = MobileOneSeparableBlock(ch[1], ch[2], stride=1)
        self.conv3 = MobileOneBlock(ch[2], ch[3], kernel_size=3, padding=1, stride=(2, 2), num_conv_branches=2)
        self.conv4 = MobileOneSeparableBlock(ch[3], ch[3], stride=1, use_res_connect=True)
        self.conv5 = ConvBNAct(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.conv6 = MobileOneSeparableBlock(ch[4], ch[4], stride=1, use_res_connect=True)
        self.conv7 = MobileOneSeparableBlock(ch[4], ch[4], stride=1, use_res_connect=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.conv7(x)


class MobileOneModel(nn.Module):
    """Original MobileOne UNet, compatible with its historical state dicts.

    Square image input and output sizes are configurable (default 144x144).
    HuBERT audio is reshaped to
    ``[B, 16, 32, 32]``; Wenet audio is ``[B, 128, 16, 32]``.
    Use ``reparameterize_model(model.eval())`` before inference/export.
    """

    model_type = "mobileone"
    image_size = IMAGE_SIZE
    audio_half_window = 4
    raw_hubert_audio = False
    visual_channels = _MOBILEONE_CHANNELS

    def __init__(
        self, n_channels: int = 6, mode: str = "hubert", image_size: int = IMAGE_SIZE
    ):
        super().__init__()
        audio_branches = {
            "wenet": MobileOneAudioConvWenet,
            "hubert": MobileOneAudioConvHubert,
        }
        if mode not in audio_branches:
            raise ValueError(f"Unknown asr mode: {mode}")

        self.n_channels = n_channels
        self.mode = mode
        self.image_size = _validate_image_size(image_size)
        ch = self.visual_channels
        self.audio_model = audio_branches[mode]()
        self.fuse_conv = nn.Sequential(
            DoubleConvMobileOne(ch[4] * 2, ch[4], stride=1),
            DoubleConvMobileOne(ch[4], ch[3], stride=1),
        )

        self.inc = InConvMobileOne(n_channels, ch[0])
        self.down1 = MobileOneDown(ch[0], ch[1])
        self.down2 = MobileOneDown(ch[1], ch[2])
        self.down3 = MobileOneDown(ch[2], ch[3])
        self.down4 = MobileOneDown(ch[3], ch[4])

        self.up1 = MobileOneUp(ch[4], ch[3] // 2)
        self.up2 = MobileOneUp(ch[3], ch[2] // 2)
        self.up3 = MobileOneUp(ch[2], ch[1] // 2)
        self.up4 = MobileOneUp(ch[1], ch[0])
        self.outc = OutConv(ch[0], 3)

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        _validate_image_input(image, self.n_channels, self.image_size)
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        audio = self.audio_model(audio)
        if audio.shape[-2:] != x5.shape[-2:]:
            audio = F.interpolate(
                audio, size=x5.shape[-2:], mode="bilinear", align_corners=False
            )
        x5 = self.fuse_conv(torch.cat([x5, audio], dim=1))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))


def reparameterize_model(model: nn.Module) -> nn.Module:
    """Fuse MobileOne branches in place; call after loading weights and eval()."""
    for module in list(model.modules()):
        if isinstance(module, MobileOneBlock):
            module.reparameterize()
    return model


MODEL_TYPES = ("feathertalk", "mobileone")


def create_model(model_type: str = "feathertalk", image_size: int = IMAGE_SIZE) -> nn.Module:
    """Create a visual model for the common FeatherHuBERT pipeline."""
    if model_type == "feathertalk":
        return Model(image_size=image_size)
    if model_type == "mobileone":
        return MobileOneModel(mode="hubert", image_size=image_size)
    raise ValueError(f"unknown model type {model_type!r}; expected one of {MODEL_TYPES}")


def visual_model_config(model: nn.Module) -> dict[str, Any]:
    return {
        "model_type": model.model_type,
        "image_size": model.image_size,
        "audio_mode": model.mode,
        "reparameterized": any(
            module.inference_mode
            for module in model.modules()
            if isinstance(module, MobileOneBlock)
        ),
    }


def load_visual_model(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    model_type: str = "auto",
    reparameterize: bool = False,
) -> nn.Module:
    """Load new metadata checkpoints or detect either architecture in old weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    config = checkpoint.get("model_config", {}) if "model" in checkpoint else {}
    detected_type = config.get("model_type")
    if detected_type is None:
        if any(key.startswith("spatial_x3.") for key in state):
            detected_type = "feathertalk"
        elif any(".rbr_conv." in key or ".reparam_conv." in key for key in state):
            detected_type = "mobileone"
        else:
            raise ValueError("Cannot identify visual model architecture in checkpoint")
    if model_type != "auto" and model_type != detected_type:
        raise ValueError(
            f"checkpoint contains {detected_type!r}, but --model selected {model_type!r}"
        )
    if config.get("audio_mode", "hubert") != "hubert":
        raise ValueError("The common pipeline only supports HuBERT visual checkpoints")
    model = create_model(detected_type, image_size=config.get("image_size", IMAGE_SIZE)).eval()
    fused = any(".reparam_conv." in key for key in state)
    if fused:
        reparameterize_model(model)
    model.load_state_dict(state, strict=True)
    if reparameterize:
        reparameterize_model(model)
    return model.to(device).eval()


class MobileOneExportAdapter(nn.Module):
    """Keep the C++ [B,40,1024] interface while using the native central 8 frames."""

    def __init__(self, model: MobileOneModel):
        super().__init__()
        if model.mode != "hubert":
            raise ValueError("MobileOne export adapter requires HuBERT mode")
        self.model = model
        center = AUDIO_TOKENS // 4
        self.token_start = (center - model.audio_half_window) * 2
        self.token_end = (center + model.audio_half_window) * 2

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[1:] != (AUDIO_TOKENS, AUDIO_FEATURE_DIM):
            raise ValueError("exported visual audio must have shape [B,40,1024]")
        native_audio = audio[:, self.token_start:self.token_end].reshape(-1, 16, 32, 32)
        return self.model(image, native_audio)


def visual_export_model(model: nn.Module) -> nn.Module:
    """Both architectures export the same image/audio interface for C++/MNN."""
    if isinstance(model, MobileOneModel):
        return MobileOneExportAdapter(model).eval()
    return model
