"""Geometric object module."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

from serialize import Serializable

# Operator = Callable[[Any, Any], Any]


@dataclass
class Vector2d(tuple, Serializable):
    x: float
    y: float

    def __new__(cls, x: float, y: float) -> Vector2d:
        return tuple.__new__(cls, (x, y))

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> Vector2d:
        norm = self.norm()
        return self.scale(norm, norm)

    def move(self, x: float, y: float) -> Vector2d:
        return type(self)(self.x + x, self.y + y)

    def dot(self, other: VectorLike) -> float:
        return self.x * other[0] + self.y * other[1]

    def scale(self, x: float, y: float) -> Vector2d:
        return type(self)(self.x * x, self.y * y)

    def to_int_tuple(self, _round: bool = True) -> tuple[int, int]:
        x, y = self
        if _round:
            return round(x), round(y)
        return int(x), int(y)

    def to_int_vec(self, _round: bool = True) -> Vector2d:
        return type(self)(*self.to_int_tuple(_round))

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    # def __add__(self, other: tuple[Any, ...]) -> tuple[Any, ...]:
    #     return self._call_op(operator.add, self, other)

    # def __sub__(self, other: tuple[float, ...]) -> Vector2d:
    #     return self._call_op(operator.sub, self, other)

    # def __mul__(self, other: Union[tuple[float, float], int, float]) -> Vector2d:
    #     return self._call_op(operator.mul, self, other)

    # def __truediv__(self, other: Union[tuple[float, float], int, float]) -> Vector2d:
    #     return self._call_op(operator.truediv, self, other)

    # def __floordiv__(self, other: Union[tuple[float, float], int, float]) -> Vector2d:
    #     return self._call_op(operator.floordiv, self, other)

    # def __mod__(self, other: Union[tuple[float, float], int, float]) -> Vector2d:
    #     return self._call_op(operator.mod, self, other)

    # def __pos__(self) -> Vector2d:
    #     return self

    # def __neg__(self) -> Vector2d:
    #     return type(self)(-self.x, self.y)

    # def __rmul__(self, other: Union[tuple[float, float], int, float]) -> Vector2d:
    #     return self._call_op(operator.mul, other, self)

    # def __rtruediv__(self, other) -> Vector2d:
    #     return self._call_op(operator.truediv, other, self)

    # def __rfloordiv__(self, other) -> Vector2d:
    #     return self._call_op(operator.floordiv, other, self)

    # def __rmod__(self, other) -> Vector2d:
    #     return self._call_op(operator.mod, other, self)

    # @classmethod
    # def _call_op(
    #     cls,
    #     op: Operator,
    #     a: Union[tuple[float, ...], int, float],
    #     b: Union[tuple[float, ...], int, float],
    # ) -> Vector2d:
    #     if isinstance(a, (int, float)):
    #         a = cls(a, a)
    #     if isinstance(b, (int, float)):
    #         b = cls(b, b)
    #     return cls(op(a[0], b[0]), op(a[1], b[1]))


VectorLike = Union[Vector2d, tuple[float, float]]


@dataclass(frozen=True)
class Rect(tuple, Serializable):
    x: float
    y: float
    w: float
    h: float

    def __new__(cls, x: float, y: float, w: float, h: float) -> Rect:
        if w < 0:
            x = x + w
            w = -w
        if h < 0:
            y = w + h
            h = -h
        return tuple.__new__(cls, (x, y, w, h))

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

    def move(self, x: float, y: float) -> Rect:
        return Rect.from_xywh(self.x + x, self.y + y, self.w, self.h)

    def scale(self, scale_x: float, scale_y: float) -> Rect:
        return Rect.from_xywh(
            scale_x * self.x, scale_y * self.y, scale_x * self.w, scale_y * self.h
        )

    def scale_from_center(self, scale_x: float, scale_y: float) -> Rect:
        w = self.w * scale_x
        h = self.h * scale_y
        center_x, center_y = self.center
        x = center_x - w / 2
        y = center_y - h / 2
        return Rect.from_xywh(x, y, w, h)

    def intersects_with(self, other: Rect) -> bool:
        return self.intersection(other) == ZERO_RECT

    def intersection(self, other: Rect) -> Rect:
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x2, other.x2)
        bot = min(self.y2, other.y2)
        if left >= right or top >= bot:
            return ZERO_RECT
        return Rect.from_xyxy(left, top, right, bot)

    def union(self, other: Rect) -> Rect:
        left = min(self.x, other.x)
        top = min(self.y, other.y)
        right = max(self.x2, other.x2)
        bot = max(self.y2, other.y2)
        return Rect.from_xyxy(left, top, right, bot)

    def iou(self, other: Rect) -> float:
        intersection_area = self.intersection(other).area
        if intersection_area == 0.0:
            return 0.0
        return intersection_area / (self.area + other.area - intersection_area)

    def fit_inner(self, aspect_ratio: float) -> Rect:
        center_x, center_y = self.center
        cur_ratio: float = self.w / self.h
        if cur_ratio > aspect_ratio:
            # self is wider
            # then cut LEFT and RIGHT
            new_w = self.h * aspect_ratio
            new_rect = Rect.from_center(center_x, center_y, new_w, self.h)
        else:
            # self is narrower
            # then cut TOP and BOT
            new_h = self.w / aspect_ratio
            new_rect = Rect.from_center(center_x, center_y, self.w, new_h)
        return new_rect

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

    def to_int_rect(self, _round: bool = True) -> Rect:
        return Rect.from_xywh(*self.to_int_tuple(_round))

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

    def __and__(self, other: Rect) -> Rect:
        return self.intersection(other)

    def __or__(self, other: Rect) -> Rect:
        return self.union(other)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> Rect:
        return cls(x, y, w, h)

    @classmethod
    def from_center(cls, center_x: float, center_y: float, w: float, h: float) -> Rect:
        return cls(center_x - w / 2, center_y - h / 2, w, h)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> Rect:
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return cls(x1, y1, x2 - x1, y2 - y1)


ZERO_RECT = Rect(0, 0, 0, 0)
