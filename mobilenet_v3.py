# MobileNet V3 implementation.
# Paper: https://arxiv.org/pdf/1905.02244.pdf

from typing import Tuple, Union
import collections

import torch
from torch import nn

import mobilenet_v3_configs as conf


def hard_sigmoid(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    return nn.functional.relu6(x + 3, inplace=inplace) / 6


def hard_swish(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    return hard_sigmoid(x, inplace=inplace) * x


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self._inplace = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_swish(x, inplace=self._inplace)


def _get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU
    elif activation == "relu6":
        return nn.ReLU6
    elif activation == "hardswish":
        return HardSwish
    else:
        raise ValueError(f"Unsupported activation: {activation}")


# SE and inverted residual are similar to MNASNet, but with MNV3 specific
# tweaks.
class _SqueezeAndExcitation(nn.Module):
    def __init__(self, channels: int, se_ratio: float):
        if se_ratio <= 0.0:
            raise ValueError("Squeeze and excitation depth ratio must be positive.")
        super().__init__()
        reduced_ch = _round_to_multiple_of(channels * se_ratio, 8)
        # Note: official implementation uses bias on SE.
        self.reduce = nn.Conv2d(channels, reduced_ch, 1, bias=False)
        self.expand = nn.Conv2d(reduced_ch, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean([2, 3], keepdim=True)
        y = nn.functional.relu(self.reduce(y), inplace=True)
        return hard_sigmoid(self.expand(y)) * x


class _ConvBnActivationBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int, int, int]],
        dilation: Union[int, Tuple[int, int]],
        activation: str = "relu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = _get_activation(activation)(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class _MobileNetV3Block(nn.Module):
    def __init__(
        self,
        in_ch,
        exp_ch,
        out_ch,
        kernel_size,
        stride,
        dilation=1,
        se_ratio=None,
        activation="relu",
        allow_residual=True,
    ):
        super().__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        activation = _get_activation(activation)
        self.apply_residual = allow_residual and (in_ch == out_ch and stride == 1)
        # Features are collected from pointwise immediately before the next
        # downsampling. If there's no downsampling, we don't keep the features.
        self.keep_features = stride > 1
        self.se_ratio = se_ratio

        if in_ch != exp_ch:
            # Pointwise expand.
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                nn.BatchNorm2d(exp_ch),
                activation(inplace=True),
            )
        else:
            self.expand = None

        effective_kernel_size = (kernel_size - 1) * dilation + 1
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                exp_ch,
                exp_ch,
                kernel_size,
                padding=effective_kernel_size // 2,
                stride=stride,
                dilation=dilation,
                groups=exp_ch,
                bias=False,
            ),
            nn.BatchNorm2d(exp_ch),
            activation(inplace=True),
        )

        if se_ratio is not None:
            self.se = _SqueezeAndExcitation(exp_ch, se_ratio)

        # Linear pointwise. Note that there's no activation afterwards.
        self.contract = nn.Sequential(
            nn.Conv2d(exp_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x) if self.expand is not None else x
        if self.keep_features:
            self.features = y
        y = self.dw_conv(y)
        if self.se_ratio is not None:
            y = self.se(y)
        y = self.contract(y)
        if self.apply_residual:
            y += x
        return y


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


class MobileNetV3(nn.Module):
    """ MobileNetV3 model: https://arxiv.org/pdf/1905.02244.pdf
    >>> model = MobileNetV3(alpha=1.0, model_type="small")
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model.forward(x)
    >>> list(y.shape)
    [1, 1000]
    >>> y.nelement()
    1000
    """

    def __init__(
        self,
        alpha: float = 1.0,
        in_ch: int = 3,
        num_classes: int = 1000,
        dropout: float = 0.2,  # Per paper.
        model_type: str = "large",
        has_classifier: bool = True,
    ):
        super().__init__()

        assert alpha > 0.0
        self.alpha = alpha
        assert in_ch > 0
        self.in_ch = in_ch
        assert num_classes > 1
        self.num_classes = num_classes
        assert model_type in conf.CONFIG
        self.model_type = model_type
        self.has_classifier = has_classifier

        config = conf.CONFIG[model_type]
        # Scale the channels, forcing them to be multiples of 8, biased towards
        # the higher number of channels.
        for c in config:
            c[0] = _round_to_multiple_of(c[0] * alpha, 8)
            c[1] = _round_to_multiple_of(c[1] * alpha, 8)
            c[2] = _round_to_multiple_of(c[2] * alpha, 8)

        # Build the first layer. It's the same for all networks.
        self.input_layer = _ConvBnActivationBlock(
            in_ch,
            config[0][0],
            3,  # kernel_size
            padding=1,
            stride=2,
            dilation=1,
            activation="hardswish",
        )

        # Build the bottleneck stack.
        body = collections.OrderedDict()
        for idx, c in enumerate(config):
            in_ch, exp_ch, out_ch, kernel_size, stride, dilation, se_ratio, activation = (
                c
            )
            body[f"bottleneck{idx}"] = _MobileNetV3Block(
                in_ch,
                exp_ch,
                out_ch,
                kernel_size,
                stride,
                dilation=dilation,
                se_ratio=se_ratio,
                activation=activation,
            )

        # Build the classifier.
        shallow_tail = any(x in model_type for x in ["_segmentation", "_detection"])
        if model_type == "large":
            last_conv_ch = 960 if not shallow_tail else 480
        elif model_type == "small":
            last_conv_ch = 576 if not shallow_tail else 288
        else:
            raise ValueError("Invalid model type")

        if alpha < 1.0:
            last_conv_ch = _round_to_multiple_of(last_conv_ch * alpha, 8)

        body["last_conv"] = _ConvBnActivationBlock(
            config[-1][2],
            last_conv_ch,
            1,
            padding=0,
            stride=1,
            dilation=1,
            activation="hardswish",
        )

        self.body = nn.Sequential(body)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_conv_ch, 1280),
            HardSwish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        if self.has_classifier:
            x = self.classifier(x)
        return x
