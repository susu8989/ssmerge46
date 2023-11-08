from dataclasses import dataclass
from pickle import TRUE
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from cv2wrap import draw_rect
from cv2wrap.color import BLACK, BLUE, GREEN, RED, Bgr
from cv2wrap.image import BgrImage, GrayImage
from geometry import Rect
from geometry.vector import Vector2d
from recogn.matching import match_template
from recogn.template import Template
from ssmerge46 import imgproc
from ssmerge46.exception import (
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
    bar_thumb_area: Rect


@dataclass(frozen=True)
class _PreProcessedData:
    inputs: list[_InputData]
    bar_area: Rect
    scroll_area: Rect

    @property
    def header_area(self) -> Rect:
        return Rect.from_xyxy(0, 0, self.inputs[0].cropped.w, self.scroll_area.y)

    @property
    def footer_area(self) -> Rect:
        return Rect.from_xyxy(
            0, self.scroll_area.y2, self.inputs[0].cropped.w, self.inputs[0].cropped.h
        )


@dataclass(frozen=True)
class _MatchingResult:
    max_loc_y: int
    max_score: float
    scores: GrayImage  # for debugging


@dataclass(frozen=True)
class StitchResult:
    preprocessed: _PreProcessedData
    query_rect: Rect
    match_thresh: float
    matching_results: list[_MatchingResult]

    @property
    def is_succeed(self) -> bool:
        return all(r.max_score >= self.match_thresh for r in self.matching_results)

    def create_result_img(self) -> BgrImage:
        header = self._crop_header()
        parts = self._crop_stitching_parts()
        return np.vstack([header, *parts]).view(BgrImage)

    def _crop_header(self) -> BgrImage:
        header_y = int(self.preprocessed.header_area.y2)
        header = self.preprocessed.inputs[0].cropped[:header_y].view(BgrImage)
        return header

    def _crop_stitching_parts(self) -> list[BgrImage]:
        cropped_imgs = [input.cropped for input in self.preprocessed.inputs]
        scroll_area = self.preprocessed.scroll_area

        parts: list[BgrImage] = []  # 結合画像リスト
        for i, img in enumerate(cropped_imgs):
            max_loc_y = self.matching_results[i].max_loc_y if i == len(cropped_imgs) else None
            query_h = self.query_rect.h / 2
            top_y = int(scroll_area.y + (0 if i == 0 else query_h / 2))
            bot_y = int((max_loc_y + (query_h / 2)) if max_loc_y else scroll_area.y2)
            parts.append(img[top_y:bot_y])
        return parts

    def create_debug_img(self) -> BgrImage:
        cropped_imgs = [input.cropped for input in self.preprocessed.inputs]
        scroll_area = self.preprocessed.scroll_area
        bar_area = self.preprocessed.bar_area
        header = self._crop_header()
        score_w = 100
        w = (header.w + score_w) * 2
        h = int(header.h + (sum(r.max_loc_y for r in self.matching_results) + scroll_area.h))

        dbg_img = BgrImage(np.zeros((h, w, 3), dtype=np.uint8))
        dbg_img.put_img(header, (score_w, 0), True)
        target_y = header.h
        for i, img in enumerate(cropped_imgs):
            max_loc_y = self.matching_results[i - 1].max_loc_y if i > 0 else 0
            if i > 0:
                scores = self.matching_results[i - 1].scores
                for loc_y, score in enumerate(scores):
                    score = min(max(score, 0), 1)
                    y = min(target_y + loc_y, dbg_img.h - 1)
                    if i % 2 == 0:
                        x1 = 0
                        x2 = int(score * score_w)
                    else:
                        x1 = int(dbg_img.w - score * score_w)
                        x2 = dbg_img.w
                    color = RED if loc_y == max_loc_y else GREEN
                    dbg_img[y, x1:x2] = color

            target_y += max_loc_y
            img = draw_rect(img, scroll_area, BLACK, inplace=True)
            img = draw_rect(img, bar_area, BLUE, inplace=True)
            img = draw_rect(img, self.preprocessed.inputs[i].bar_thumb_area, GREEN, inplace=True)
            if i > 0:
                img = draw_rect(img, self.query_rect, RED, inplace=True)
            img = img.crop(Rect.from_xywh(0, scroll_area.y, img.w, scroll_area.h))
            dbg_img.put_img(
                img,
                (score_w + (i % 2) * header.w, target_y),
                True,
            )

        return dbg_img


class ScrollStitcher:
    """ウマ娘スクロール画面の画像結合器.

    - スキル表示・因子表示画面の縦スクロールエリアを自動検出して縦に結合します.
    - その他の画面には基本的に使えません.
    - 画像の解像度は統一し、1～2割程度の重複部分（のりしろ）を作ってください.
    """

    _MIN_WH = Vector2d(360, 640)

    def stitch(
        self,
        imgs: Sequence[BgrImage],
        overlap_ratios: Iterable[float] = (0.2,),
        match_threshs: Iterable[float] = (0.5,),
        debug: bool = False,
    ) -> BgrImage:
        """画像を結合する. 結合に失敗した場合はパラメータを順に変更しながらリトライする.

        Args:
            imgs (Sequence[BgrImage]): 入力とする画像の一覧.
            overlap_ratio (Iterable[float], optional): 結合位置検出に使うのりしろの標準比率 (0-1). Defaults to (0.2,).
                スクロールエリアの高さにこの数値をかけた高さをのりしろ領域とみなし, 1つ前の画像との共通部分を探します.
            match_thresh (Iterable[float], optional): テンプレートマッチングを行う際の閾値 (0-1). Defaults to (0.5,).
            debug (bool, optional): デバッグ画像を出力するか. Defaults to False.

        Raises:
            InvalidInputImageError: 入力画像が異常な場合.
            InvalidSettingError: 結合処理用のパラメータの値が異常な場合.
            ImageProcessingError: 画像の余白落とし処理に失敗した場合.
            ScrollbarDetectionError: スクロールバーの検知に失敗した場合.
            OverlapDetectionError: 全てのパラメータで重複部分の検知に失敗した場合.

        Returns:
            BgrImage: 結合した画像.
        """
        if len(imgs) < 2:
            raise InvalidInputImageError("2枚以上の画像が必要です。")
        preprocessed = self._pre_process(imgs)
        for ratio, thresh in zip(overlap_ratios, match_threshs):
            try:
                result = self._process(preprocessed, ratio, thresh, ignore_failure=debug)
                if debug:
                    return result.create_debug_img()
                return result.create_result_img()
            except OverlapDetectionError as e:
                print(f"{e}, {ratio=}, {thresh=}")
        if debug:
            return result.create_debug_img()
        raise OverlapDetectionError("類似部分の検出に失敗しました。")

    def _pre_process(self, imgs: Sequence[BgrImage]) -> _PreProcessedData:
        base = imgs[0]
        for i, img in enumerate(imgs):
            img_num = i + 1
            if base.wh != img.wh:
                raise InvalidInputImageError(
                    f"入力画像の解像度が一致しません。 : 1枚目 = {base.wh}, {img_num}枚目 = {img.wh}"
                )

        tgt_aspect_ratio = self._MIN_WH.x / self._MIN_WH.y
        pre_cropping_rect = imgproc.unmargin_rect(
            base,
        ).cut_with_fixed_aspect_ratio(tgt_aspect_ratio)
        unmargined_rect = imgproc.unmargin_rect(base.crop(pre_cropping_rect)).move(
            *pre_cropping_rect.topleft
        )

        cropped_imgs: list[BgrImage] = []
        for img in imgs:
            cropped = img.crop(unmargined_rect)
            cropped_imgs.append(cropped)

        test_base = cropped_imgs[0]
        bar_area = self._detect_scroll_bar_area(
            test_base, tgt=test_base.rect.crop_by_ratio(x1=0.5, x2=1.0)  # only right side
        )
        scroll_area = Rect.from_xyxy(16, bar_area.y, bar_area.x, bar_area.y2)
        sorted_inputs = sorted(
            [
                _InputData(
                    imgs[i],
                    cropped_imgs[i],
                    self._detect_scroll_bar_thumb(cropped_imgs[i], bar_area),
                )
                for i in range(len(imgs))
            ],
            key=lambda x: (x.bar_thumb_area.y - bar_area.y) / x.bar_thumb_area.h,
        )
        return _PreProcessedData(sorted_inputs, bar_area, scroll_area)

    def _process(
        self,
        preprocessed: _PreProcessedData,
        overlap_ratio: float,
        match_thresh: float,
        ignore_failure: bool = False,
    ) -> StitchResult:
        if not 0.0 <= overlap_ratio < 1.0:
            raise InvalidSettingError(f"重複検索範囲比率が無効です。 : {overlap_ratio=}")
        if not 0.0 <= match_thresh < 1.0:
            raise InvalidSettingError(f"重複スコア閾値が無効です。 : {match_thresh=}")

        test_imgs = [input.cropped for input in preprocessed.inputs]
        scroll_area = preprocessed.scroll_area
        query_h = int(scroll_area.h * overlap_ratio)
        query_rect = Rect.from_xywh(scroll_area.x, scroll_area.y, scroll_area.w, query_h)

        # 結合位置を検出する
        base = test_imgs[0]
        matched_positions: list[_MatchingResult] = []
        for prev, cur in zip(test_imgs, test_imgs[1:]):
            # 次の画像の先頭部分を使って前の画像との類似点を探す
            query = cur.crop(query_rect)

            scores = match_template(
                prev,
                Template(query, 1, (base.w, base.h)),
                tgt=Rect.from_xyxy(
                    scroll_area.x, scroll_area.y, scroll_area.x2, scroll_area.y2 + query_h
                ),
            )
            _, score, _, loc = cv2.minMaxLoc(scores)
            if not ignore_failure and score < match_thresh:
                raise OverlapDetectionError(f"類似部分の検出に失敗しました。 : {score=}")
            print(f"重複部分検出成功。 : {score=:.4f} >= {match_thresh}, {overlap_ratio=}")

            matched_positions.append(_MatchingResult(int(loc[1]), score, scores))

        return StitchResult(preprocessed, query_rect, match_thresh, matched_positions)

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

    def _detect_scroll_bar_thumb(
        self, img: BgrImage, bar_area: Rect, binary_thresh: int = 144
    ) -> Rect:
        """スクロールバーのつまみ (暗色部分) のY座標を検出する.

        Args:
            img (BgrImage): 被検索対象画像.
            bar_area (Rect): img 中でスクロールバーを検索する領域 (px).
            binary_thresh (int, optional): 2値化時の閾値. デフォルトは 144.

        Raises:
            Exception: スクロールバーの検出に失敗した場合.

        Returns:
            Rect: 上端のY座標と高さ. (原点はimgと共通)
        """
        cropped = img.crop(bar_area)
        gray_scaled = cropped.to_gray()
        _, binarized = cv2.threshold(gray_scaled, binary_thresh, 255, cv2.THRESH_BINARY)
        horizontal_summed = np.sum(binarized, axis=1)

        # バーの暗色部分が存在するY座標のリスト
        black_ys = np.where(horizontal_summed < gray_scaled.shape[1] * 255)
        if len(black_ys) == 0 or black_ys[0].size == 0:
            raise ScrollbarDetectionError("スクロールバーが検出結果が異常です。")
        t = int(np.min(black_ys))
        b = int(np.max(black_ys))

        return Rect.from_xyxy(bar_area.x, bar_area.y + t, bar_area.x2, bar_area.y + b)
