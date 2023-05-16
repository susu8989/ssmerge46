from __future__ import annotations

from typing_extensions import Self

from geometry.vector import Vector2d, VectorLike


class Rect(tuple[float, float, float, float]):
    def __new__(cls, x: float, y: float, w: float, h: float) -> Self:
        if w < 0:
            x = x + w
            w = -w
        if h < 0:
            y = w + h
            h = -h
        return tuple.__new__(cls, (x, y, w, h))  # type: ignore

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def w(self) -> float:
        return self[2]

    @property
    def h(self) -> float:
        return self[3]

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def size(self) -> Vector2d:
        return Vector2d(self.w, self.h)

    @property
    def topleft(self) -> Vector2d:
        return Vector2d(self.x, self.y)

    @property
    def topright(self) -> Vector2d:
        return Vector2d(self.x2, self.y)

    @property
    def botleft(self) -> Vector2d:
        return Vector2d(self.x, self.y2)

    @property
    def botright(self) -> Vector2d:
        return Vector2d(self.x2, self.y2)

    @property
    def center(self) -> Vector2d:
        center_x = (self.x + self.x2) / 2
        center_y = (self.y + self.y2) / 2
        return Vector2d(center_x, center_y)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return self.w / self.h

    def move(self, x: float, y: float) -> Self:
        return self.from_xywh(self.x + x, self.y + y, self.w, self.h)

    def scale(self, scale: float | VectorLike) -> Self:
        if isinstance(scale, (int, float)):
            scale_x = scale_y = scale
        else:
            scale_x, scale_y = scale
        return self.from_xywh(
            scale_x * self.x, scale_y * self.y, scale_x * self.w, scale_y * self.h
        )

    def crop_by_ratio(self, x1=0.0, y1=0.0, x2=1.0, y2=1.0) -> Self:
        new_w = self.w * (x2 - x1)
        new_h = self.h * (y2 - y1)
        new_x = self.x + self.w * x1
        new_y = self.y + self.h * y1
        return self.from_xywh(new_x, new_y, new_w, new_h)

    def scale_from_center(self, scale_x: float, scale_y: float) -> Self:
        w = self.w * scale_x
        h = self.h * scale_y
        center_x, center_y = self.center
        x = center_x - w / 2
        y = center_y - h / 2
        return self.from_xywh(x, y, w, h)

    def intersects_with(self, other: Rect) -> bool:
        return self.intersection(other) == self.empty()

    def intersection(self, other: Rect) -> Self:
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x2, other.x2)
        bot = min(self.y2, other.y2)
        if left >= right or top >= bot:
            return self.empty()
        return self.from_xyxy(left, top, right, bot)

    def union(self, other: Rect) -> Self:
        left = min(self.x, other.x)
        top = min(self.y, other.y)
        right = max(self.x2, other.x2)
        bot = max(self.y2, other.y2)
        return self.from_xyxy(left, top, right, bot)

    def iou(self, other: Rect) -> float:
        intersection_area = self.intersection(other).area
        if intersection_area == 0.0:
            return 0.0
        return intersection_area / (self.area + other.area - intersection_area)

    def fit_inner(self, aspect_ratio: float) -> Self:
        center_x, center_y = self.center
        if self.aspect_ratio > aspect_ratio:
            # self is wider
            # then cut LEFT and RIGHT
            new_w = self.h * aspect_ratio
            new_rect = self.from_center(center_x, center_y, new_w, self.h)
        else:
            # self is narrower
            # then cut TOP and BOT
            new_h = self.w / aspect_ratio
            new_rect = self.from_center(center_x, center_y, self.w, new_h)
        return new_rect

    def expand(self, left: float, top: float, right: float, bot: float) -> Self:
        x, y, x2, y2 = self.to_xyxy_tuple()
        return self.from_xyxy(x - left, y - top, x2 + right, y2 + bot)

    def expand_all(self, length: float) -> Self:
        return self.expand(length, length, length, length)

    def to_int_tuple(
        self, _round: bool = True, one_if_zero: bool = True
    ) -> tuple[int, int, int, int]:
        x, y, w, h = map(round if _round else int, self)  # type: ignore
        if one_if_zero:
            if w < 1:
                w = 1
            if h < 1:
                h = 1
        return x, y, w, h

    def to_int_rect(self, _round: bool = True) -> Self:
        return self.from_xywh(*self.to_int_tuple(_round))

    def to_xywh_tuple(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.w, self.h

    def to_xyxy_tuple(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x2, self.y2

    def to_xywh_int_tuple(self) -> tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)

    def to_xyxy_int_tuple(self) -> tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.x2), int(self.y2)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.x}, {self.y}, {self.w}, {self.h})"

    def __and__(self, other: Rect) -> Self:
        return self.intersection(other)

    def __or__(self, other: Rect) -> Self:
        return self.union(other)

    @classmethod
    def empty(cls) -> Self:
        return cls(0, 0, 0, 0)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> Self:
        return cls(x, y, w, h)

    @classmethod
    def from_center(cls, center_x: float, center_y: float, w: float, h: float) -> Self:
        return cls(center_x - w / 2, center_y - h / 2, w, h)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> Self:
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return cls(x1, y1, x2 - x1, y2 - y1)

    @classmethod
    def from_wh(cls, w: float, h: float) -> Self:
        return cls(0, 0, w, h)

    def ratio_to_px(self, px_rect: Rect) -> Self:
        """正規化された矩形領域 (0-1) から, 指定した領域上のピクセル座標における矩形領域を求める.

        結果が画像サイズからはみ出す場合, その部分は切り取られる.

        Args:
            img (Cv2Image): H x W x C または H x W の配列で表される画像.

        Returns:
            Self: ピクセル座標 (0-W, 0-H) で表される矩形領域.
        """
        return self.scale(px_rect.size).intersection(px_rect).to_int_rect()

    def px_to_ratio(self, ratio_rect: Rect) -> Self:
        """ピクセル座標における矩形領域 (0-W, 0-H) から, 指定した領域上の正規化された矩形領域を求める.

        結果が画像サイズからはみ出す場合, その部分は切り取られる.

        Args:
            img (Cv2Image): H x W x C または H x W の配列で表される画像.

        Returns:
            Self: 正規化 (0-1) された座標で表現される矩形領域.
        """
        w, h = ratio_rect.size
        return self.scale((1 / w, 1 / h)).intersection(self.from_wh(1.0, 1.0))
