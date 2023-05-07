from __future__ import annotations

from typing import Union, cast

import cv2
import numpy as np
import numpy.typing as npt

# NOTE: 入出力形式を明示するために alias を作ったが, shape の厳密なチェックは掛からないので注意
BGRImage = npt.NDArray[np.uint8]
GrayImage = npt.NDArray[np.uint8]

Cv2Image = Union[BGRImage, GrayImage]


class BGR(tuple):
    b: int  # 0-255
    g: int  # 0-255
    r: int  # 0-255

    def __new__(cls, b: int, g: int, r: int) -> BGR:
        # TODO: validate range
        return tuple.__new__(cls, (b, g, r))

    def to_hsv(self) -> HSV:
        return bgr2hsv(self)


class HSV(tuple):
    h: int  # 0-179
    s: int  # 0-255
    v: int  # 0-255

    def __new__(cls, h: int, s: int, v: int) -> HSV:
        # TODO: validate range
        return tuple.__new__(cls, (h, s, v))

    def to_bgr(self) -> BGR:
        return hsv2bgr(self)


BLACK = BGR(0, 0, 0)
GRAY = BGR(127, 127, 127)
WHITE = BGR(255, 255, 255)

BLUE = BGR(255, 0, 0)
GREEN = BGR(0, 255, 0)
RED = BGR(0, 0, 255)
CYAN = BGR(255, 255, 0)
MAGENTA = BGR(255, 0, 255)
YELLOW = BGR(0, 255, 255)

BGRLike = Union[tuple[int, int, int], BGR]
HSVLike = Union[tuple[int, int, int], HSV]


def bgr2hsv(bgr: BGRLike) -> HSV:
    return HSV(*cv2.cvtColor(np.array([[bgr]], np.uint8), cv2.COLOR_BGR2HSV)[0][0])


def hsv2bgr(hsv: HSVLike) -> BGR:
    return BGR(*cv2.cvtColor(np.array([[hsv]], np.uint8), cv2.COLOR_HSV2BGR)[0][0])


def in_bgr_range(bgr_img: BGRImage, lower: HSVLike, upper: HSVLike) -> GrayImage:
    return cast(GrayImage, cv2.inRange(bgr_img, lower, upper))


def in_hsv_range(bgr_img: BGRImage, lower: HSVLike, upper: HSVLike) -> GrayImage:
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    return cast(GrayImage, cv2.inRange(hsv_img, lower, upper))


def in_bgr_range_pm(
    bgr_img: BGRImage, bgr: BGRLike, hsv_margins: tuple[int, int, int]
) -> GrayImage:
    lower = HSV(*map(lambda x: x[0] + x[1], zip(bgr, hsv_margins)))
    upper = HSV(*map(lambda x: x[0] - x[1], zip(bgr, hsv_margins)))
    print(lower, upper)
    return in_bgr_range(bgr_img, lower, upper)


def in_hsv_range_pm(
    bgr_img: BGRImage, hsv: HSVLike, hsv_margins: tuple[int, int, int]
) -> GrayImage:
    lower = HSV(*map(lambda x: x[0] - x[1], zip(hsv, hsv_margins)))
    upper = HSV(*map(lambda x: x[0] + x[1], zip(hsv, hsv_margins)))
    return in_hsv_range(bgr_img, lower, upper)


def calc_white_ratio(gray: GrayImage) -> float:
    white_pxs = cv2.countNonZero(gray)
    return white_pxs / gray.size
