from typing import List, Sequence

import cv2
import numpy as np

import cv2wrap as cv2wrap
from cv2wrap.color import BGRImage
from geometry import Rect
from recogn.matching import match_max
from recogn.template import Template

UMA_SCROLL_RATIO = Rect.from_xyxy(0.02, 0.465, 0.96, 0.865)
UMA_SCROLL_BAR_RATIO = Rect.from_xyxy(0.96, 0.465, 0.98, 0.865)


def stitch(
    imgs: Sequence[BGRImage],
    tgt_area_ratio: Rect = UMA_SCROLL_RATIO,
    bar_area_ratio: Rect = UMA_SCROLL_BAR_RATIO,
    overlap_ratio: float = 0.1,
    match_thresh: float = 0.8,
    include_footer: bool = False,
) -> BGRImage:
    """縦スクロール部分を結合して1枚のスクリーンショットを作成する.

    結合位置の特定に使用するため, 各画像はおよそ50px以上の重複した部分が必要です.
    また, 各画像の解像度は揃っている必要があります.

    Args:
        imgs (Sequence[BGRImage]): 結合するスクリーンショット群.
        tgt_area_ratio (Rect, optional): img 中でスクロールエリアを検索する領域 (比率).
            デフォルトはスキルまたは因子表示画面での位置.
        bar_area_ratio (Rect, optional): img 中でスクロールバーを検索する領域 (比率).
            デフォルトはスキルまたは因子表示画面での位置.
        overlap_ratio (float, optional): 各画像間で重複する部分の高さ (tgt_area の高さに対する比率).
            Defaults to 0.1.
        match_thresh (float, optional): テンプレートマッチングにおける類似度の閾値. Defaults to 0.8.
        include_footer (bool, optional): フッター部分を含めるか. Defaults to False.

    Raises:
        ValueError: imgs が空リストの時.
        ValueError: imgs の横解像度が一致しない時.
        Exception: 画像中のオーバーラップ部分が検知できない時.

    Returns:
        BGRImage: 結合後の画像
    """
    num = len(imgs)
    if num < 1:
        raise ValueError("'imgs' が空です。")
    if len(imgs) == 1:
        return imgs[0]

    # リサイズ
    resized_imgs = preproc_imgs(imgs)

    # スクロールバー位置でソート
    sorted_imgs = sorted(
        resized_imgs, key=lambda x: calc_scroll_bar_pos(x, bar_area_ratio)
    )

    base_h, base_w, _ = sorted_imgs[0].shape
    if any(base_w != img.shape[1] for img in sorted_imgs):
        raise ValueError("画像の幅が一致しません。")

    # 各領域の絶対座標
    tgt_rect = cv2wrap.ratio_rect_to_px(tgt_area_ratio, sorted_imgs[0])
    header_rect = Rect.from_xyxy(0, 0, base_w, tgt_rect.y)
    footer_rect = Rect.from_xyxy(0, tgt_rect.y2, base_w, base_h)

    query_h = int(tgt_rect.h * overlap_ratio)
    query_rect = Rect.from_xywh(tgt_rect.x, tgt_rect.y, tgt_rect.w, query_h)

    header = cv2wrap.crop_img(sorted_imgs[0], header_rect)
    footer = cv2wrap.crop_img(sorted_imgs[-1], footer_rect)

    # スクロールエリアのY座標
    tgt_top = int(tgt_rect.y)
    tgt_bot = int(tgt_rect.y2)

    # スクロール部分を結合していく
    # stitched: Optional[BGRImage] = None
    to_stack: List[BGRImage] = [sorted_imgs[0][tgt_top:tgt_bot]]

    for prev, cur in zip(sorted_imgs, sorted_imgs[1:]):
        query = cv2wrap.crop_img(cur, query_rect)

        matched_rect, score = match_max(
            prev,
            Template(query, 1, (base_w, base_h)),
            thresh=0.0,
            tgt=tgt_rect,
        )

        if matched_rect and score >= match_thresh:
            overlap_h = int(tgt_bot - matched_rect.y)
        else:
            # raise Exception("Failed to detect overlap in images.")
            print("Low matching score, simply stacked :", score)
            overlap_h = 0

        cropping_y = tgt_top + overlap_h
        to_stack.append(cur[cropping_y:tgt_bot])

    stacked = np.vstack(
        [header, *to_stack, footer] if include_footer else [header, *to_stack]
    )
    return stacked


def preproc_imgs(
    imgs: Sequence[BGRImage], limit_w: int = 720, aspect: float = 9 / 16
) -> List[BGRImage]:
    """解像度を調整し, ウマ表示部を切り抜く.

    Args:
        imgs (Sequence[BGRImage]): 結合するスクリーンショット群.
        limit_w (int, optional): 最大横解像度 (px). Defaults to 1080.
        aspect (float, optional): 切り抜くアスペクト比 (横/縦). Defaults to 9/16.

    Returns:
        List[BGRImage]: リサイズ済画像のリスト.
    """
    results = []
    for img in imgs:
        h, w, _ = img.shape
        cropping_rect = Rect.from_xywh(0, 0, w, h).fit_inner(aspect)
        cropped = cv2wrap.crop_img(img, cropping_rect)
        cropped_h, cropped_w, _ = cropped.shape
        if cropped_w > limit_w:
            dsize = (limit_w, round(cropped_h * limit_w / cropped_w))
            cropped = cv2.resize(cropped, dsize=dsize)
        results.append(cropped)
    return results


def calc_scroll_bar_pos(
    img: BGRImage, bar_area_ratio: Rect = UMA_SCROLL_BAR_RATIO, binary_thresh: int = 144
) -> int:
    """スクロールバーのY座標を検知する.

    Args:
        img (BGRImage): 被検索対象画像.
        bar_area_ratio (Rect, optional): img 中でスクロールバーを検索する領域(%).
            デフォルトはスキルまたは因子表示画面での位置.
        binary_thresh (int, optional): 2値化時の閾値. Defaults to 144.

    Raises:
        Exception: スクロールバーの検知に失敗した場合.

    Returns:
        int: スクロールバーの暗色部分の上端Y座標.
    """
    tgt_rect = cv2wrap.ratio_rect_to_px(bar_area_ratio, img)
    cropped = cv2wrap.crop_img(img, tgt_rect)
    gray_scaled = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray_scaled, binary_thresh, 255, cv2.THRESH_BINARY)
    horizontal_summed = np.sum(binarized, axis=1)

    # バーの暗色部分が存在するY座標のリスト
    black_ys = np.where(horizontal_summed < gray_scaled.shape[1] * 255)
    if len(black_ys) == 0 or black_ys[0].size == 0:
        raise Exception("スクロールバーの検知に失敗しました。")
    return int(tgt_rect.h) + int(np.min(black_ys))
