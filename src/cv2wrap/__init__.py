"""画像ユーティリティモジュール."""
import errno
import os
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt

from cv2wrap.color import BLACK, WHITE, BGRImage, BGRLike, Cv2Image, GrayImage
from geometry import Rect, Vector2d


def read(
    filename: Union[str, Path],
    flags: int = cv2.IMREAD_COLOR,
    dtype: npt.DTypeLike = np.uint8,
) -> BGRImage:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)

    buf = np.fromfile(filename, dtype)
    img = cv2.imdecode(buf, flags)
    return img


def write(filename: Union[str, Path], img: BGRImage, *params: int) -> bool:
    path = Path(filename)
    if not path.parent.is_dir():
        raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), path.parent)
    ext = path.suffix
    result, buf = cv2.imencode(ext, img, params)
    if result:
        with open(filename, mode="w+b") as f:
            buf.tofile(f)
        return True
    else:
        return False


def bgr2gray(bgr: BGRImage) -> GrayImage:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def gray2bgr(bgr: BGRImage) -> GrayImage:
    return cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)


def ratio_rect_to_px(normed_rect: Rect, img: Cv2Image) -> Rect:
    """正規化された矩形領域から, 指定した画像上のピクセル座標における矩形領域を求める.

    結果が画像サイズからはみ出す場合, その部分は切り取られる.

    Args:
        normed_rect (Rect): 正規化 (0-1) された座標で表現される矩形領域.
        img (Cv2Image): H x W x C または H x W の配列で表される画像.

    Returns:
        Rect: ピクセル座標 (0-W, 0-H) で表される矩形領域.
    """
    h, w = img.shape[:2]
    return normed_rect.scale(w, h).intersection(Rect(0, 0, w, h)).to_int_rect()


def px_rect_to_ratio(rect: Rect, img: Cv2Image) -> Rect:
    """ピクセル座標における矩形領域から, 指定した画像上の正規化された矩形領域を求める.

    結果が画像サイズからはみ出す場合, その部分は切り取られる.

    Args:
        rect (Rect): ピクセル座標 (0-W, 0-H) で表される矩形領域.
        img (Cv2Image): H x W x C または H x W の配列で表される画像.

    Returns:
        Rect: 正規化 (0-1) された座標で表現される矩形領域.
    """
    h, w = img.shape[:2]
    return rect.scale(1 / w, 1 / h).intersection(Rect(0.0, 0.0, 1.0, 1.0))


def crop_img(img: Cv2Image, rect: Rect) -> Cv2Image:
    """画像配列を矩形領域で切り抜く.

    返り値は img のビューであり, メモリは共有されることに注意.

    Args:
        img (Cv2Image): 切り抜かれる画像.
        rect (Rect): 切り抜く領域. img のピクセル数によって指定された矩形.
    Returns:
        Cv2Image: img のスライス.
    """
    if not rect:
        return img
    x1, y1, x2, y2 = rect.to_xyxy_int_tuple()
    return img[y1:y2, x1:x2]


def crop_fixed_aspect_ratio(img: Cv2Image, aspect_ratio: float) -> Cv2Image:
    """画像配列を指定アスペクト比で切り抜く.

    img が aspect_ratio より縦長の場合は上下, 横長の場合は左右が切り取られる.
    返り値は img のビューであり, メモリは共有されることに注意.

    Args:
        img (Cv2Image): 切り抜かれる画像.
        aspect_ratio (float): アスペクト比 (横/縦). img のピクセル数によって指定された矩形.
    Returns:
        Cv2Image: img のスライス.
    """
    h, w = img.shape[:2]
    cropping_rect = Rect.from_xywh(0, 0, w, h).fit_inner(aspect_ratio)
    return crop_img(img, cropping_rect)


def put_img(img: BGRImage, img2: BGRImage, pos: Vector2d) -> BGRImage:
    src_h, src_w, _ = img.shape
    src2_h, src2_w, _ = img2.shape
    rect = Rect(0, 0, src_w, src_h)
    rect2 = Rect(0, 0, src2_w, src2_h).move(*pos)
    intersection = rect.intersection(rect2)

    w, h = intersection.size.to_int_tuple()
    x, y = pos.to_int_tuple()
    x2 = x + w
    y2 = y + h
    copy = img.copy()
    copy[y:y2, x:x2] = img2[0:h, 0:w]
    return copy


def draw_rect(
    img: BGRImage,
    rect: Rect,
    color: BGRLike,
    thickness: int = 1,
    line_type: int = cv2.LINE_AA,
    inplace: bool = False,
) -> BGRImage:
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
    img: BGRImage,
    text: Any,
    org: Vector2d,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 1.0,
    color: BGRLike = BLACK,
    stroke_color: BGRLike = WHITE,
    thickness: int = 1,
    stroke_thickness: int = 1,
    line_type: int = cv2.LINE_AA,
    inplace: bool = False,
) -> BGRImage:
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


def unmargin(img: BGRImage, scan_px: Optional[int] = None) -> BGRImage:
    rect = unmargin_rect(img, scan_px=scan_px)
    return crop_img(img, rect)


def unmargin_rect(img: BGRImage, scan_px: Optional[int] = None) -> Rect:
    h, w, _ = img.shape
    if scan_px is None or scan_px < 1:
        scan_x_h = h
        scan_y_w = w
    else:
        scan_x_h = scan_px
        scan_y_w = scan_px
    center = Vector2d(w / 2, h / 2)
    top = Vector2d(center.x, 0)
    bot = Vector2d(center.x, h - 1)
    left = Vector2d(0, center.y)
    right = Vector2d(w - 1, center.y - 1)

    top_color = img[int(top.y), int(top.x)]
    bot_color = img[int(bot.y), int(bot.x)]
    left_color = img[int(left.y), int(left.x)]
    right_color = img[int(right.y), int(right.x)]

    scan_rect_x = Rect.from_center(center.x, center.y, w, scan_x_h).to_int_rect()
    scan_rect_y = Rect.from_center(center.x, center.y, scan_y_w, h).to_int_rect()

    scan_img_x = crop_img(img, scan_rect_x)
    scan_img_y = crop_img(img, scan_rect_y)

    scan_x_from_left = np.where(np.any(scan_img_x != left_color, axis=2))
    scan_x_from_right = np.where(np.any(scan_img_x != right_color, axis=2))
    scan_y_from_top = np.where(np.any(scan_img_y != top_color, axis=2))
    scan_y_from_bot = np.where(np.any(scan_img_y != bot_color, axis=2))

    new_left = int(np.min(scan_x_from_left[1]))
    new_right = int(np.max(scan_x_from_right[1])) + 1
    new_top = int(np.min(scan_y_from_top[0]))
    new_bot = int(np.max(scan_y_from_bot[0])) + 1

    return Rect.from_xyxy(new_left, new_top, new_right, new_bot)


def morph_open(
    img: BGRImage, kernel=np.ones((3, 3), np.uint8), iterations=1
) -> BGRImage:
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morph_close(
    img: BGRImage, kernel=np.ones((3, 3), np.uint8), iterations=1
) -> BGRImage:
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
