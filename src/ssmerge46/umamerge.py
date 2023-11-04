from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from cv2wrap.color import Bgr
from cv2wrap.image import BgrImage, GrayImage
from geometry import Rect
from geometry.vector import Vector2d
from recogn.matching import match_max
from recogn.template import Template
from ssmerge46 import imgproc
from ssmerge46.exception import (
    ImageProcessingError,
    InvalidInputImageError,
    InvalidSettingError,
    OverlapDetectionError,
    ScrollbarDetectionError,
)

UMA_SCROLL_RATIO = Rect.from_xyxy(0.02, 0.465, 0.96, 0.865)
UMA_SCROLL_BAR_RATIO = Rect.from_xyxy(0.94, 0.465, 0.98, 0.865)


@dataclass(frozen=True)
class _InputData:
    org: BgrImage
    cropped: BgrImage
    resized: BgrImage
    bar_pos: int  # in resized px


@dataclass(frozen=True)
class _PreProcessedData:
    inputs: List[_InputData]
    bar_area: Rect  # in resized px
    scroll_area: Rect  # in resized px

    @property
    def header_area(self) -> Rect:
        return Rect.from_xyxy(0, 0, self.inputs[0].resized.w, self.scroll_area.y)

    @property
    def footer_area(self) -> Rect:
        return Rect.from_xyxy(
            0, self.scroll_area.y2, self.inputs[0].resized.w, self.inputs[0].resized.h
        )


class ScrollStitcher:
    """ウマ娘スクロール画面の画像結合器.

    - スキル表示・因子表示画面の縦スクロールエリアを自動検出して縦に結合します.
    - その他の画面には基本的に使えません.
    - 画像の解像度は統一し、1～2割程度の重複部分（のりしろ）を作ってください.
    """

    test_wh = Vector2d(360, 640)

    def stitch_retryable(
        self,
        imgs: Sequence[BgrImage],
        overlap_ratios: Iterable[float] = (0.2,),
        match_threshs: Iterable[float] = (0.5,),
    ) -> BgrImage:
        """画像を結合する. 結合に失敗した場合はパラメータを順に変更しながらリトライする.

        Args:
            imgs (Sequence[BgrImage]): 入力とする画像の一覧.
            overlap_ratio (Iterable[float], optional): 結合位置検出に使うのりしろの標準比率 (0-1). Defaults to (0.2,).
                スクロールエリアの高さにこの数値をかけた高さをのりしろ領域とみなし, 1つ前の画像との共通部分を探します.
            match_thresh (Iterable[float], optional): テンプレートマッチングを行う際の閾値 (0-1). Defaults to (0.5,).

        Raises:
            InvalidInputImageError: 入力画像が異常な場合.
            InvalidSettingError: 結合処理用のパラメータの値が異常な場合.
            ImageProcessingError: 画像の余白落とし処理に失敗した場合.
            ScrollbarDetectionError: スクロールバーの検知に失敗した場合.
            OverlapDetectionError: 全てのパラメータで重複部分の検知に失敗した場合.

        Returns:
            BgrImage: 結合した画像.
        """
        if len(imgs) < 1:
            raise InvalidInputImageError("画像が空です。")
        if len(imgs) == 1:
            return imgs[0]
        preprocessed = self._pre_process(imgs)
        for ratio, thresh in zip(overlap_ratios, match_threshs):
            try:
                return self._process(preprocessed, ratio, thresh)
            except OverlapDetectionError as e:
                print(f"{e}, {ratio=}, {thresh=}")
        raise OverlapDetectionError("類似部分の検出に失敗しました。")

    def stitch(
        self, imgs: Sequence[BgrImage], overlap_ratio: float = 0.2, match_thresh: float = 0.5
    ) -> BgrImage:
        """画像を結合する.

        Args:
            imgs (Sequence[BgrImage]): 入力とする画像の一覧.
            overlap_ratio (float, optional): 結合位置検出に使うのりしろの標準比率 (0-1). Defaults to 0.2.
                スクロールエリアの高さにこの数値をかけた高さをのりしろ領域とみなし, 1つ前の画像との共通部分を探します.
            match_thresh (float, optional): テンプレートマッチングを行う際の閾値 (0-1). Defaults to 0.5.

        Raises:
            InvalidInputImageError: 入力画像が異常な場合.
            InvalidSettingError: 結合処理用のパラメータの値が異常な場合.
            ImageProcessingError: 画像の余白落とし処理に失敗した場合.
            ScrollbarDetectionError: スクロールバーの検知に失敗した場合.
            OverlapDetectionError: 重複部分の検知に失敗した場合.

        Returns:
            BgrImage: 結合した画像.
        """
        if len(imgs) < 1:
            raise InvalidInputImageError("画像が空です。")
        if len(imgs) == 1:
            return imgs[0]
        pre_processed = self._pre_process(imgs)
        return self._process(pre_processed, overlap_ratio, match_thresh)

    def _pre_process(self, imgs: Sequence[BgrImage]) -> _PreProcessedData:
        """入力画像の前処理 (切り抜き, リサイズ, スクロール検出, ソート) を行う.

        Args:
            imgs (Sequence[BgrImage]): 入力画像のシーケンス.

        Raises:
            InvalidInputImageError: _description_
            ImageProcessingError: _description_

        Returns:
            _PreProcessedData: _description_
        """
        base = imgs[0]
        for i, img in enumerate(imgs):
            img_num = i + 1
            if base.wh != img.wh:
                raise InvalidInputImageError(
                    f"入力画像の解像度が一致しません。 : 1枚目 = {base.wh}, {img_num}枚目 = {img.wh}"
                )

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
                raise ImageProcessingError("余白切り抜きに失敗しました。")
            cropped_imgs.append(cropped)
            resized_imgs.append(resized)

        test_base = resized_imgs[0]
        bar_area = self._detect_scroll_bar_area(
            test_base, tgt=test_base.rect.crop_by_ratio(x1=0.5, x2=1.0)  # only right side
        )
        scroll_area = Rect.from_xyxy(16, bar_area.y, bar_area.x, bar_area.y2)
        sorted_inputs = sorted(
            [
                _InputData(
                    imgs[i],
                    cropped_imgs[i],
                    resized_imgs[i],
                    self._detect_scroll_bar_pos(resized_imgs[i], bar_area),
                )
                for i in range(len(imgs))
            ],
            key=lambda x: x.bar_pos,
        )
        return _PreProcessedData(sorted_inputs, bar_area, scroll_area)

    def _process(
        self, preprocessed: _PreProcessedData, overlap_ratio: float, match_thresh: float
    ) -> BgrImage:
        if not (0.0 <= overlap_ratio < 1.0):
            raise InvalidSettingError(f"重複検索範囲比率が無効です。 : {overlap_ratio=}")
        if not (0.0 <= match_thresh < 1.0):
            raise InvalidSettingError(f"重複スコア閾値が無効です。 : {match_thresh=}")

        # 方針: リサイズした固定解像度上で結合位置を確定してから元の画像解像度で結合して出力
        test_imgs = [input.resized for input in preprocessed.inputs]
        scale = preprocessed.inputs[0].cropped.h / preprocessed.inputs[0].resized.h
        scroll_area = preprocessed.scroll_area
        query_h = int(scroll_area.h * overlap_ratio)
        query_rect = Rect.from_xywh(scroll_area.x, scroll_area.y, scroll_area.w, query_h)

        # 結合位置を検出する
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
            if not (matched_rect and score >= match_thresh):
                raise OverlapDetectionError(f"類似部分の検出に失敗しました。 : {score=}")
            print(f"重複部分検出成功。 : {score=}")
            matched_ys.append(int(matched_rect.y))
        matched_ys.append(-1)  # 最後の画像分

        cropped_imgs = [input.cropped for input in preprocessed.inputs]
        header_y = int(preprocessed.header_area.y2 * scale)
        header = cropped_imgs[0][:header_y]
        to_stack: List[BgrImage] = [header]  # 結合画像リスト
        for i, matched_y in enumerate(matched_ys):
            prev = cropped_imgs[i]
            y = int((scroll_area.y + (0 if i == 0 else query_h / 2)) * scale)
            y2 = int(((matched_y + (query_h / 2)) if matched_y > 0 else scroll_area.y2) * scale)
            to_stack.append(prev[y:y2])

        stacked = np.vstack(to_stack).view(BgrImage)
        return stacked

    def _detect_scroll_bar_area(
        self,
        img: BgrImage,
        tgt: Optional[Rect] = None,
        light_bgr: Bgr = Bgr(218, 210, 210),
        dark_bgr: Bgr = Bgr(138, 122, 122),
        bgr_tolerance: int = 8,
        expantion=(2, 6, 2, 6),
    ) -> Rect:
        """スクロールバー領域を検出する.

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
            raise ScrollbarDetectionError("スクロールバーが検出できませんでした。")
        contour = max(contours, key=cv2.contourArea)
        rect = Rect.from_xywh(*cv2.boundingRect(contour)).expand(*expantion)
        if rect.aspect_ratio > 0.2:
            raise ScrollbarDetectionError("スクロールバーの検出結果が異常です。")
        return rect.move(tgt.x, tgt.y)  # px in test_size

    def _detect_scroll_bar_pos(
        self, img: BgrImage, bar_area: Rect, binary_thresh: int = 144
    ) -> int:
        """スクロールバーのY座標を検出する.

        Args:
            img (BgrImage): 被検索対象画像.
            bar_area (Rect): img 中でスクロールバーを検索する領域 (px).
            binary_thresh (int, optional): 2値化時の閾値. デフォルトは 144.

        Raises:
            Exception: スクロールバーの検出に失敗した場合.

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
            raise ScrollbarDetectionError("スクロールバーが検出結果が異常です。")
        return int(bar_area.h) + int(np.min(black_ys))
