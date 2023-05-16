from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
from typing_extensions import Self


class Color(ABC):
    @abstractmethod
    def to_bgr(self) -> Bgr:
        pass

    @abstractmethod
    def to_hsv(self) -> Hsv:
        pass

    @abstractmethod
    def to_gray(self) -> GrayScale:
        pass


class Bgr(tuple, Color):
    b: int  # 0-255
    g: int  # 0-255
    r: int  # 0-255

    def __new__(cls, b: int, g: int, r: int) -> Self:
        # TODO: validate range
        return tuple.__new__(cls, (b, g, r))

    @classmethod
    def from_bgr(cls, b: int, g: int, r: int) -> Self:
        return cls(b, g, r)

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int) -> Self:
        return cls(b, g, r)

    def to_bgr(self) -> Bgr:
        return self

    def to_hsv(self) -> Hsv:
        return bgr2hsv(self)

    def to_gray(self) -> GrayScale:
        return bgr2gray(self)


class Hsv(tuple, Color):
    h: int  # 0-179
    s: int  # 0-255
    v: int  # 0-255

    def __new__(cls, h: int, s: int, v: int) -> Hsv:
        # TODO: validate range
        return tuple.__new__(cls, (h, s, v))

    def to_bgr(self) -> Bgr:
        return hsv2bgr(self)

    def to_hsv(self) -> Hsv:
        return self

    def to_gray(self) -> GrayScale:
        return hsv2gray(self)


class GrayScale(int, Color):
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
