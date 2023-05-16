from typing import Any

import cv2

from cv2wrap.color import BLACK, WHITE, BgrLike
from cv2wrap.image import BgrImage, GrayImage
from geometry.rect import Rect
from geometry.vector import Vector2d


def bgr2gray(bgr: BgrImage) -> GrayImage:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def gray2bgr(bgr: BgrImage) -> GrayImage:
    return cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)


def draw_rect(
    img: BgrImage,
    rect: Rect,
    color: BgrLike,
    thickness: int = 1,
    line_type: int = cv2.LINE_AA,
    inplace: bool = False,
) -> BgrImage:
    if not inplace:
        img = img.copy()
    if rect:
        cv2.rectangle(
            img,
            rect.topleft.to_int_tuple(),
            rect.botright.to_int_tuple(),
            color=color,
            thickness=thickness,
            lineType=line_type,
        )
    return img


def draw_text_with_stroke(
    img: BgrImage,
    text: Any,
    org: Vector2d,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 1.0,
    color: BgrLike = BLACK,
    stroke_color: BgrLike = WHITE,
    thickness: int = 1,
    stroke_thickness: int = 1,
    line_type: int = cv2.LINE_AA,
    inplace: bool = False,
) -> BgrImage:
    if not inplace:
        img = img.copy()
    text = str(text)
    if stroke_thickness > 0:
        cv2.putText(
            img,
            text,
            org.to_int_tuple(),
            font,
            scale,
            stroke_color,
            thickness=thickness + stroke_thickness,
            lineType=line_type,
        )
    cv2.putText(
        img,
        text,
        org.to_int_tuple(),
        font,
        scale,
        color,
        thickness=thickness,
        lineType=line_type,
    )
    return img
