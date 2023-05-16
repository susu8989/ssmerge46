from typing import Optional

import numpy as np

from cv2wrap import BgrImage
from geometry import Rect, Vector2d


def unmargin(img: BgrImage, scan_px: Optional[int] = None) -> BgrImage:
    rect = unmargin_rect(img, scan_px=scan_px)
    return img.crop(rect)


def unmargin_rect(img: BgrImage, scan_px: Optional[int] = None) -> Rect:
    """画像の上下左右の各辺において, 中央の色を基準にして同色部分の余白を取り除いた領域を取得する.

    各辺それぞれで画像端から中心に向けて画像を走査し, 異なる色が1つでも出現した行または列までを余白とする.

    Args:
        img (BgrImage): 余白を探す画像.
        scan_px (Optional[int], optional): 走査する幅 (px). デフォルトでは辺の全てを対象とする.

    Returns:
        Rect: 余白を取り除いた領域.
    """
    w, h = img.wh
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

    scan_img_x = img.crop(scan_rect_x)
    scan_img_y = img.crop(scan_rect_y)

    scan_x_from_left = np.where(np.any(scan_img_x != left_color, axis=2))
    scan_x_from_right = np.where(np.any(scan_img_x != right_color, axis=2))
    scan_y_from_top = np.where(np.any(scan_img_y != top_color, axis=2))
    scan_y_from_bot = np.where(np.any(scan_img_y != bot_color, axis=2))

    new_left = int(np.min(scan_x_from_left[1]))
    new_right = int(np.max(scan_x_from_right[1])) + 1
    new_top = int(np.min(scan_y_from_top[0]))
    new_bot = int(np.max(scan_y_from_bot[0])) + 1

    return Rect.from_xyxy(new_left, new_top, new_right, new_bot)
