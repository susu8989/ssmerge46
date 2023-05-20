from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from cv2wrap.color import Bgr
from cv2wrap.image import BgrImage, GrayImage
from geometry import Rect
from geometry.vector import Vector2d
from recogn.matching import match_max
from recogn.template import Template
from ssmerge46 import imgproc
from ssmerge46.exception import CroppingError, DetectionError

UMA_SCROLL_RATIO = Rect.from_xyxy(0.02, 0.465, 0.96, 0.865)
UMA_SCROLL_BAR_RATIO = Rect.from_xyxy(0.94, 0.465, 0.98, 0.865)


@dataclass(frozen=True)
class InputData:
    org: BgrImage
    cropped: BgrImage
    resized: BgrImage
    bar_pos: int  # in test


@dataclass(frozen=True)
class InspectionResult:
    inputs: List[InputData]
    bar_area: Rect  # in test
    scroll_area: Rect  # in test

    @property
    def header_area(self) -> Rect:
        return Rect.from_xyxy(0, 0, self.inputs[0].resized.w, self.scroll_area.y)

    @property
    def footer_area(self) -> Rect:
        return Rect.from_xyxy(
            0, self.scroll_area.y2, self.inputs[0].resized.w, self.inputs[0].resized.h
        )


class ScrollStitcher:
    test_wh = Vector2d(360, 640)
    overlap_ratio = 0.2
    match_thresh = 0.5

    def stitch(self, imgs: Sequence[BgrImage]):
        num = len(imgs)
        if num < 1:
            raise ValueError("'imgs' が空です。")
        if len(imgs) == 1:
            return imgs[0]

        # 方針

        # 接続位置を比率で確定
        # 出力解像度で出力

        inspection = self._pre_inspect(imgs)
        test_imgs = [input.resized for input in inspection.inputs]
        scale = inspection.inputs[0].cropped.h / inspection.inputs[0].resized.h
        scroll_area = inspection.scroll_area
        query_h = int(scroll_area.h * self.overlap_ratio)
        query_rect = Rect.from_xywh(scroll_area.x, scroll_area.y, scroll_area.w, query_h)

        # 結合位置を検知する
        base = test_imgs[0]
        matched_ys: List[float] = []
        for prev, cur in zip(test_imgs, test_imgs[1:]):
            # 次の画像の先頭部分を使って前の画像との類似点を探す
            query = cur.crop(query_rect)
            matched_rect, score = match_max(
                prev,
                Template(query, 1, (base.w, base.h)),
                thresh=0.0,
                tgt=Rect.from_xyxy(
                    scroll_area.x, scroll_area.y, scroll_area.x2, scroll_area.y2 + query_h
                ),
            )
            if not (matched_rect and score >= self.match_thresh):
                print("Low matching score :", score)
                raise DetectionError("類似部分の検知に失敗しました。")
            matched_ys.append(matched_rect.y)

        cropped_imgs = [input.cropped for input in inspection.inputs]
        header_y = int(inspection.header_area.y2 * scale)
        header = cropped_imgs[0][:header_y]
        to_stack: List[BgrImage] = [header]  # 結合画像リスト
        for i, y in enumerate(matched_ys):
            prev = cropped_imgs[i]
            crop_y = int(scroll_area.y * scale)
            crop_y2 = int(y * scale)
            to_stack.append(prev[crop_y:crop_y2])
        crop_y = int(scroll_area.y * scale)
        crop_y2 = int(scroll_area.y2 * scale)
        to_stack.append(cropped_imgs[-1][crop_y:crop_y2])

        stacked = np.vstack(to_stack).view(BgrImage)
        return stacked

    def _pre_inspect(self, imgs: Sequence[BgrImage]) -> InspectionResult:
        base = imgs[0]
        if any(base.wh != img.wh for img in imgs):
            raise ValueError("入力画像の解像度が一致しません。")

        main_aspect_ratio = self.test_wh.x / self.test_wh.y
        pre_cropping_rect = imgproc.unmargin_rect(base, 30).cut_with_fixed_aspect_ratio(
            main_aspect_ratio
        )
        unmargined_rect = imgproc.unmargin_rect(base.crop(pre_cropping_rect), 30).move(
            *pre_cropping_rect.topleft
        )

        cropped_imgs: List[BgrImage] = []
        resized_imgs: List[BgrImage] = []
        for img in imgs:
            cropped = img.crop(unmargined_rect)
            resized = cropped.resize_img(self.test_wh)
            if resized.wh != self.test_wh:
                raise CroppingError("余白切り抜きに失敗しました。")
            cropped_imgs.append(cropped)
            resized_imgs.append(resized)

        test_base = resized_imgs[0]
        bar_area = self._detect_scroll_bar_area(
            test_base, tgt=test_base.rect.crop_by_ratio(x1=0.5, x2=1.0)  # only right side
        )
        scroll_area = Rect.from_xyxy(16, bar_area.y, bar_area.x, bar_area.y2)
        sorted_inputs = sorted(
            [
                InputData(
                    imgs[i],
                    cropped_imgs[i],
                    resized_imgs[i],
                    self._detect_scroll_bar_pos(resized_imgs[i], bar_area),
                )
                for i in range(len(imgs))
            ],
            key=lambda x: x.bar_pos,
        )
        return InspectionResult(sorted_inputs, bar_area, scroll_area)

    def _detect_scroll_bar_area(
        self,
        img: BgrImage,
        tgt: Optional[Rect] = None,
        light_bgr: Bgr = Bgr(218, 210, 210),
        dark_bgr: Bgr = Bgr(138, 122, 122),
        bgr_tolerance: int = 8,
        expantion=(2, 6, 2, 6),
    ) -> Rect:
        """スクロールバー領域を検知する.

        Args:
            img (BgrImage): 被検索対象画像.
            tgt (Optional[Rect], optional) 検出対象領域. デフォルトはNone (画像全体を対象とする).
            light_bgr (Bgr, optional): 明るい部分の色. デフォルトは Bgr(218, 210, 210).
            dark_bgr (Bgr, optional): 暗い部分の色. デフォルトは Bgr(138, 122, 122).
            bgr_tolerance (int, optional): 許容する色の誤差範囲. デフォルトは 8.
            expantion (tuple, optional): 検出結果の左上右下への拡大範囲 (px). デフォルトは (2, 6, 2, 6).

        Returns:
            Rect: スクロールバーの領域 (px). 原点は img と共通.
        """
        if tgt:
            test_img = img.crop(tgt)
        else:
            tgt = img.rect
            test_img = img

        light = test_img.create_mask_by_bgr_pm(light_bgr, (bgr_tolerance,) * 3)
        dark = test_img.create_mask_by_bgr_pm(dark_bgr, (bgr_tolerance,) * 3)
        summed = (light + dark).view(GrayImage)

        m1 = summed.morph_close(np.ones((5, 1), np.uint8))
        m2 = m1.morph_open(np.ones((50, 1), np.uint8))

        contours, _ = cv2.findContours(m2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise DetectionError("スクロールバーが検知できませんでした。")
        contour = max(contours, key=cv2.contourArea)
        rect = Rect.from_xywh(*cv2.boundingRect(contour)).expand(*expantion)
        if rect.aspect_ratio > 0.2:
            raise DetectionError("スクロールバーの検知結果が異常です。")
        return rect.move(tgt.x, tgt.y)  # px in test_size

    def _detect_scroll_bar_pos(
        self, img: BgrImage, bar_area: Rect, binary_thresh: int = 144
    ) -> int:
        """スクロールバーのY座標を検知する.

        Args:
            img (BgrImage): 被検索対象画像.
            bar_area (Rect): img 中でスクロールバーを検索する領域 (px).
            binary_thresh (int, optional): 2値化時の閾値. デフォルトは 144.

        Raises:
            Exception: スクロールバーの検知に失敗した場合.

        Returns:
            int: スクロールバーの暗色部分の上端Y座標. (原点はimgと共通)
        """
        cropped = img.crop(bar_area)
        gray_scaled = cropped.to_gray()
        _, binarized = cv2.threshold(gray_scaled, binary_thresh, 255, cv2.THRESH_BINARY)
        horizontal_summed = np.sum(binarized, axis=1)

        # バーの暗色部分が存在するY座標のリスト
        black_ys = np.where(horizontal_summed < gray_scaled.shape[1] * 255)
        if len(black_ys) == 0 or black_ys[0].size == 0:
            raise DetectionError("スクロールバーが検知結果が異常です。")
        return int(bar_area.h) + int(np.min(black_ys))
