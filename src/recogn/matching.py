"""テンプレートマッチングモジュール."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Collection, Dict, List, Tuple, TypeVar

import cv2
import numpy as np
from numpy.typing import ArrayLike

from cv2wrap.image import BgrImage, GrayImage
from geometry import Rect
from recogn.template import Template

T = TypeVar("T")


@dataclass(frozen=True)
class MatchingResult:
    rect: Rect  # マッチした領域
    score: float  # スコア
    template: Template  # マッチしたテンプレート


def match_template(img: BgrImage, tmpl: Template, tgt: Rect | None = None) -> GrayImage:
    cropped = img.crop(tgt) if tgt else img
    w, h = cropped.wh
    if h == 0 or w == 0:
        print("Image size is 0.")
        return GrayImage(np.empty((0, 0), float))

    t_w, t_h = tmpl.img.wh
    if h < t_h or w < t_w:
        print(f"Image size ({w}, {h}) is smaller than template size ({t_w}, {t_h}).")
        return np.empty((0, 0), float).view(GrayImage)

    preproccessed = tmpl.preprocess(cropped)
    return cv2.matchTemplate(preproccessed, tmpl.img, cv2.TM_CCOEFF_NORMED)


def match_max(
    img: BgrImage,
    tmpl: Template[T],
    thresh: float | None = None,
    tgt: Rect | None = None,
) -> Tuple[Rect | None, float]:
    """テンプレートマッチングを実行し, 類似度が最大になる矩形領域を返す.

    Args:
        img (BgrImage): 被検索対象画像.
        tmpl (Template[T]): 検索するテンプレート画像.
        thresh (float | None, optional): 類似度の閾値 (0.0-1.0). Defaults to None.
        tgt (Rect | None, optional): img 中で検索対象とする領域. Defaults to None.

    Returns:
        Tuple[Rect | None, float]: (類似度が最大になる矩形領域, 類似度) の tuple.
        類似度が thresh 未満だった場合の1要素目は None になる.
    """
    result = match_template(img, tmpl, tgt=tgt)
    _, max_score, _, max_loc = cv2.minMaxLoc(result)

    tmpl_h, tmpl_w = tmpl.img.shape[:2]
    rect = Rect.from_xywh(max_loc[0], max_loc[1], tmpl_w, tmpl_h)
    if tgt:
        rect = rect.move(*tgt.topleft)

    if thresh is not None and max_score < thresh:
        # print(
        #     f"Not matched. : {tmpl.path.stem}, {max_score:.3f} (<{thresh}), {rect} in {tgt}"
        # )
        return None, max_score
    # print(
    #     f"Matched. : {tmpl.path.stem}, {max_score:.3f} (>={thresh}), {rect}, in {tgt}"
    # )
    return rect, max_score


def match_all(
    img: BgrImage,
    tmpls: Collection[Template[T]],
    thresh: float,
    tgt: Rect | None = None,
    nms_thresh: float = 0.6,
) -> Dict[T, List[Rect]]:
    """複数のテンプレート画像に対してテンプレートマッチングを実行し, 各テンプレート毎にマッチした領域のリストを返す.

    Args:
        img (BgrImage): 被検索対象画像.
        tmpls (Collection[Template[T]]): 検索するテンプレート画像群.
        thresh (float): 類似度の閾値 (0.0-1.0). Defaults to None.
        tgt (Rect | None, optional): img 中で検索対象とする領域. Defaults to None.
        nms_thresh (float, optional): マッチした領域に対して実施するNMS処理の閾値. Defaults to 0.6.

    Returns:
        Dict[T, List[Rect]]: {テンプレート: テンプレートに対してマッチした領域のリスト} で表される dict.
    """
    h, w = img.shape[:2]
    if tgt:
        cropped = img.crop(tgt)
    else:
        tgt = img.rect
        cropped = img

    bboxes = []
    scores = []
    keys = []
    for tmpl in tmpls:
        tmpl = tmpl.fit_resolution((w, h))
        result = match_template(cropped, tmpl)

        if result.size > 0:
            tgt_x, tgt_y = tgt.topleft
            tmpl_h, tmpl_w = tmpl.img.shape[:2]
            ys, xs = np.where(result >= thresh)
            bboxes.extend(
                (
                    [
                        [tgt_x + x, tgt_y + y, tgt_x + x + tmpl_w, tgt_y + y + tmpl_h]
                        for x, y in zip(xs, ys)
                    ]
                )
            )
            scores.extend(result[ys, xs])
            keys.extend([tmpl.value] * len(xs))

    if nms_thresh > 0:
        remains = apply_nms(bboxes, scores, thresh=nms_thresh)
        bboxes = [bboxes[i] for i in remains]
        keys = [keys[i] for i in remains]

    key_2_rects = defaultdict(list)
    for key, bbox in zip(keys, bboxes):
        key_2_rects[key].append(Rect.from_xyxy(*bbox))
    return key_2_rects


def match_best(
    img: BgrImage,
    tmpls: Collection[Template[T]],
    thresh: float,
    default: T | None = None,
    tgt: Rect | None = None,
) -> Tuple[T | None, Rect | None]:
    """複数のテンプレート画像に対してテンプレートマッチングを実行し, 類似度が最大となるテンプレートとその矩形領域を返す.

    Args:
        img (BgrImage): 被検索対象画像.
        tmpls (Collection[Template[T]]): 検索するテンプレート画像群.
        thresh (float): 類似度の閾値 (0.0-1.0). Defaults to None.
        default (T | None, optional): マッチするものが見つからなかった場合のデフォルト値. Defaults to None.
        tgt (Rect | None, optional): img 中で検索対象とする領域. Defaults to None.

    Returns:
        Tuple[T | None, Rect | None: (類似度が最大となるテンプレート, およびその矩形領域) で表される tuple.
            類似度が thresh 未満だった場合は (deafult, None) を返す.
    """
    key_rect_val = [
        (
            tmpl.value,
            *match_max(img, tmpl, thresh=thresh, tgt=tgt),
        )
        for tmpl in tmpls
    ]
    best = max(key_rect_val, key=lambda x: x[2])
    best_key, best_rect, _ = best
    if not best_rect:
        return default, None

    return best_key, best_rect


def apply_nms(
    boxes: ArrayLike, scores: ArrayLike, thresh: float = 0.0, top_k: int = 100
) -> List[int]:
    """Non-Maximum Suppression (NMS) を適用する.

    参考: https://www.sigfoss.com/developer_blog/detail?actual_object_id=379

    Args:
        boxes (ArrayLike): N x 4 の行列で表される矩形領域の候補.
        scores (ArrayLike): boxes それぞれに対応するスコア.
        thresh (float, optional): NMSの対象とするIOUの最低値. Defaults to 0.0.
        top_k (int, optional): スコア上位から何個で足切りするか. Defaults to 100.

    Returns:
        List[int]: NMSの処理結果. 残すべき boxes のインデックスのリスト (順序はスコアの降順).
    """
    boxes = np.array(boxes)
    scores = np.array(scores)
    keep: List[int] = []
    if len(boxes) == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idx = np.argsort(scores, axis=0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals

    while len(idx) > 0:
        last = len(idx) - 1
        i = idx[last]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[idx[:last]])
        yy1 = np.maximum(y1[i], y1[idx[:last]])
        xx2 = np.minimum(x2[i], x2[idx[:last]])
        yy2 = np.minimum(y2[i], y2[idx[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (area[idx[:last]] + area[i] - inter)
        idx = np.delete(idx, np.concatenate([[last], np.where(iou > thresh)[0]]))

    return keep
