"""色データクラスと既定色の定義."""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
from typing_extensions import Self


def _validate_range(name: str, val: int | float, lower: int | float, upper: int | float) -> None:
    if val < lower or upper < val:
        raise ValueError(f"The value '{name}' must be in [0, 255]. : {val}")


class Color(ABC):
    """色クラスの既定."""

    @abstractmethod
    def to_bgr(self) -> Bgr:
        """BGRで表現されるデータに変換する.

        Returns:
            Bgr: BGRで表現される色.
        """

    @abstractmethod
    def to_hsv(self) -> Hsv:
        """HSVで表現されるデータに変換する.

        Returns:
            Hsv: HSVで表現される色.
        """

    @abstractmethod
    def to_gray(self) -> GrayScale:
        """グレースケールで表現されるデータに変換する.

        Returns:
            GrayScale: グレースケールで表現される色.
        """


class Bgr(tuple, Color):
    """BGR形式で表現される色.

    Attributes:
        b (int): Blue (0-255).
        g (int): Green (0-255).
        r (int): Red (0-255).
    """

    @property
    def b(self) -> int:
        """Blue (0-255)."""
        return self[0]

    @property
    def g(self) -> int:
        """Green (0-255)."""
        return self[1]

    @property
    def r(self) -> int:
        """Red (0-255)."""
        return self[2]

    def __new__(cls, b: int, g: int, r: int) -> Self:
        _validate_range("b", b, 0, 255)
        _validate_range("g", b, 0, 255)
        _validate_range("r", b, 0, 255)
        return tuple.__new__(cls, (b, g, r))

    @classmethod
    def from_bgr(cls, b: int, g: int, r: int) -> Self:
        """BGR値からインスタンスを生成する."""
        return cls(b, g, r)

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int) -> Self:
        """RGB値からインスタンスを生成する."""
        return cls(b, g, r)

    def to_bgr(self) -> Bgr:
        return self

    def to_hsv(self) -> Hsv:
        return bgr2hsv(self)

    def to_gray(self) -> GrayScale:
        return bgr2gray(self)


class Hsv(tuple, Color):
    """HSV形式で表現される色.

    Attributes:
        h (int): Hue (0-179).
        s (int): Saturation (0-255).
        v (int): Value (0-255).
    """

    @property
    def h(self) -> int:
        """Hue (0-179)."""
        return self[0]

    @property
    def s(self) -> int:
        """Saturation (0-255)."""
        return self[1]

    @property
    def v(self) -> int:
        """Value (0-255)."""
        return self[2]

    def __new__(cls, h: int, s: int, v: int) -> Self:
        _validate_range("h", h, 0, 179)
        _validate_range("s", s, 0, 255)
        _validate_range("v", v, 0, 255)
        return tuple.__new__(cls, (h, s, v))

    def to_bgr(self) -> Bgr:
        return hsv2bgr(self)

    def to_hsv(self) -> Hsv:
        return self

    def to_gray(self) -> GrayScale:
        return hsv2gray(self)


class GrayScale(int, Color):
    """HSV形式で表現される色."""

    def to_bgr(self) -> Bgr:
        return gray2bgr(self)

    def to_hsv(self) -> Hsv:
        return gray2hsv(self)

    def to_gray(self) -> GrayScale:
        return self


BLACK = Bgr(0, 0, 0)
GRAY = Bgr(127, 127, 127)
WHITE = Bgr(255, 255, 255)

BLUE = Bgr(255, 0, 0)
GREEN = Bgr(0, 255, 0)
RED = Bgr(0, 0, 255)
CYAN = Bgr(255, 255, 0)
MAGENTA = Bgr(255, 0, 255)
YELLOW = Bgr(0, 255, 255)

BgrLike = tuple[int, int, int] | Bgr
HsvLike = tuple[int, int, int] | Hsv
GrayLike = int | GrayScale

ColorLike = BgrLike | HsvLike | GrayLike


def bgr2hsv(bgr: BgrLike) -> Hsv:
    return Hsv(*cv2.cvtColor(np.array([[bgr]], np.uint8), cv2.COLOR_BGR2HSV)[0][0])


def bgr2gray(bgr: BgrLike) -> GrayScale:
    return GrayScale(cv2.cvtColor(np.array([[bgr]], np.uint8), cv2.COLOR_BGR2GRAY)[0][0])


def hsv2bgr(hsv: HsvLike) -> Bgr:
    return Bgr(*cv2.cvtColor(np.array([[hsv]], np.uint8), cv2.COLOR_HSV2BGR)[0][0])


def hsv2gray(hsv: HsvLike) -> GrayScale:
    return GrayScale(bgr2gray(hsv2bgr(hsv)))


def gray2bgr(gray: GrayLike) -> Bgr:
    return Bgr(*cv2.cvtColor(np.array([[gray]], np.uint8), cv2.COLOR_GRAY2BGR)[0][0])


def gray2hsv(gray: GrayLike) -> Hsv:
    return Hsv(*bgr2hsv(gray2bgr(gray)))
