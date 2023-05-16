"""Geometric object module."""
from __future__ import annotations

import math

from typing_extensions import Self


class Vector2d(tuple[float, float]):
    def __new__(cls, x: float, y: float) -> Self:
        return tuple.__new__(cls, (x, y))  # type: ignore

    # - __magic__

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    # - public @property

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    # - public method

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> Self:
        norm = self.norm()
        return self.scale(norm)

    def move(self, x: float, y: float) -> Self:
        return type(self)(self.x + x, self.y + y)

    def dot(self, other: VectorLike) -> float:
        return self.x * other[0] + self.y * other[1]

    def scale(self, scale: float | VectorLike) -> Self:
        if isinstance(scale, (int, float)):
            scale_x = scale_y = scale
        else:
            scale_x, scale_y = scale
        return type(self)(self.x * scale_x, self.y * scale_y)

    def to_int_tuple(self, _round: bool = True) -> tuple[int, int]:
        x, y = self
        if _round:
            return round(x), round(y)
        return int(x), int(y)

    def to_int_vec(self, _round: bool = True) -> Self:
        return type(self)(*self.to_int_tuple(_round))


VectorLike = Vector2d | tuple[float, float]
