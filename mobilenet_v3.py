""" PyTorch implementation of MobileNet V3.
    Paper: https://arxiv.org/pdf/1905.02244.pdf."""

from typing import Tuple, Union
import collections

import torch
from torch import nn

CONFIG = {
    "large": [
        # in_ch, exp, out_ch, k, s, se, activation
        [16, 16, 16, 3, 1, None, "relu"],
        [16, 64, 24, 3, 2, None, "relu"],
        [24, 72, 24, 3, 1, None, "relu"],
        [24, 72, 40, 5, 2, 0.25, "relu"],
        [40, 120, 40, 5, 1, 0.25, "relu"],
        [40, 120, 40, 5, 1, 0.25, "relu"],
        [40, 240, 80, 3, 2, None, "hardswish"],
        [80, 200, 80, 3, 1, None, "hardswish"],
        [80, 184, 80, 3, 1, None, "hardswish"],
        [80, 184, 80, 3, 1, None, "hardswish"],
        [80, 480, 112, 3, 1, 0.25, "hardswish"],
        [112, 672, 112, 3, 1, 0.25, "hardswish"],
        [112, 672, 160, 5, 2, 0.25, "hardswish"],
        [160, 960, 160, 5, 1, 0.25, "hardswish"],
        [160, 960, 160, 5, 1, 0.25, "hardswish"],
    ],
    "small": [
        # in_ch, exp, out_ch, k, s, se, activation
        [16, 16, 16, 3, 2, 0.25, "relu"],
        [16, 72, 24, 3, 2, None, "relu"],
        [24, 88, 24, 3, 1, None, "relu"],
        [24, 96, 40, 5, 2, 0.25, "hardswish"],
        [40, 240, 40, 5, 1, 0.25, "hardswish"],
        [40, 240, 40, 5, 1, 0.25, "hardswish"],
        [40, 120, 48, 5, 1, 0.25, "hardswish"],
        [48, 144, 48, 5, 1, 0.25, "hardswish"],
        [48, 288, 96, 5, 2, 0.25, "hardswish"],
        [96, 576, 96, 5, 1, 0.25, "hardswish"],
        [96, 576, 96, 5, 1, 0.25, "hardswish"],
    ],
}


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
        # Note: some implementations do not use bias on SE. However, the
        # reference implementation of MNASNet (predecessor to MNv3 in some
        # ways) does use it, so we use it here.
        self.reduce = nn.Conv2d(channels, reduced_ch, 1, bias=True)
        self.expand = nn.Conv2d(reduced_ch, channels, 1, bias=True)

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
        se_ratio=None,
        activation="relu",
        allow_residual=True,
    ):
        super().__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        activation = _get_activation(activation)
        self.apply_residual = allow_residual and (in_ch == out_ch and stride == 1)

        layers = [
            # Pointwise
            nn.Conv2d(in_ch, exp_ch, 1, bias=False),
            nn.BatchNorm2d(exp_ch),
            activation(inplace=True),  # ?
            # Depthwise
            nn.Conv2d(
                exp_ch,
                exp_ch,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=exp_ch,
                bias=False,
            ),
            nn.BatchNorm2d(exp_ch),
            activation(inplace=True),
        ]
        # SE goes after activation. This is where the paper is unclear. In e.g.
        # MNASNet, for instance, SE goes after activation. I've done runs
        # with activation both before and after, and thus far, the results were
        # better with activation before SE. Still not as good as the paper
        # claims, but close enough for practical work.
        if se_ratio is not None:
            layers += [_SqueezeAndExcitation(exp_ch, se_ratio)]
        layers += [  # Linear pointwise. Note that there's no activation afterwards.
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_residual:
            return self.layers(x) + x
        else:
            return self.layers(x)


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
        model_type: str = "small",
    ):
        super().__init__()

        assert alpha > 0.0
        self.alpha = alpha
        assert in_ch > 0
        self.in_ch = in_ch
        assert num_classes > 1
        self.num_classes = num_classes
        assert model_type in CONFIG
        self.model_type = model_type

        config = CONFIG[model_type]
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
            in_ch, exp_ch, out_ch, kernel_size, stride, se_ratio, activation = c
            body[f"bottleneck{idx}"] = _MobileNetV3Block(
                in_ch,
                exp_ch,
                out_ch,
                kernel_size,
                stride,
                se_ratio=se_ratio,
                activation=activation,
            )
        self.body = nn.Sequential(body)

        # Build the classifier.
        if model_type == "large":
            classifier_inner_ch = 960
        elif model_type == "small":
            classifier_inner_ch = 576
        else:
            raise ValueError("Invalid model type")

        if alpha < 1.0:
            classifier_inner_ch = _round_to_multiple_of(classifier_inner_ch * alpha, 8)

        self.classifier = nn.Sequential(
            _ConvBnActivationBlock(
                config[-1][2],
                classifier_inner_ch,
                1,
                padding=0,
                stride=1,
                dilation=1,
                activation="hardswish",
            ),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(classifier_inner_ch, 1280),
            HardSwish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        return self.classifier(x)
