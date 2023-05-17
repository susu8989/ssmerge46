from __future__ import annotations

import errno
import inspect
import os
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from cv2wrap.color import Bgr, BgrLike, ColorLike, Hsv, HsvLike
from geometry import Rect, Vector2d, VectorLike


class Cv2Image(npt.NDArray[np.uint8], ABC):
    """Python OpenCV 用の画像配列の基底.

    実体は np.ndarray[np.dtype(np.uint8)] であり,
    次元数は最低でも2 (h, w, ...).
    """

    _DIMS: Tuple[int, ...] = ()  # like [-1, -1, 3]
    _DTYPE = np.uint8
    _IMREAD_FLAGS = cv2.IMREAD_COLOR

    def __new__(cls, arr: npt.ArrayLike) -> Self:
        if inspect.isabstract(cls):
            raise TypeError("Can't instantiate abstract class {}".format(cls.__name__))

        obj = np.array(arr, dtype=np.uint8).view(cls)
        if not obj._is_valid():
            raise ValueError(
                f"Invalid shape : input_shape={obj.shape}" + f", required_shape={cls._DIMS}"
            )
        return obj

    # - __magic__

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        args = (x.view(np.ndarray) if isinstance(x, self.__class__) else x for x in inputs)
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)

        if results is NotImplemented:
            return NotImplemented
        if method == "at":
            return

        if isinstance(results, np.ndarray) and is_valid_shape(results.shape, self._DIMS):
            return results.view(self.__class__)
        return results

    def __getitem__(self, key) -> Any:
        item = super().__getitem__(key)
        if isinstance(item, self.__class__) and not item._is_valid():
            return item.view(np.ndarray)
        return item

    # - public @property

    @property
    def w(self) -> int:
        return self.shape[1]

    @property
    def h(self) -> int:
        return self.shape[0]

    @property
    def wh(self) -> Vector2d:
        h, w = self.shape[:2]
        return Vector2d(w, h)

    @property
    def px(self) -> int:
        return self.w * self.h

    @property
    def rect(self) -> Rect:
        return Rect.from_wh(*self.wh)

    @property
    def aspect_ratio(self) -> float:
        return self.rect.aspect_ratio

    # - public @classmethod

    @classmethod
    def open(
        cls,
        path: str | PathLike,
        dtype: npt.DTypeLike = _DTYPE,
    ) -> Self:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        if path.is_dir():
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
        buf = np.fromfile(path, dtype)
        img = cls.decode(buf)
        return img

    @classmethod
    def decode(cls, buf: npt.NDArray[np.uint8]) -> Self:
        return cls(cv2.imdecode(buf, cls._IMREAD_FLAGS))

    # - public method

    def save(self, path: str | PathLike, *params: int) -> None:
        path = Path(path)
        if not path.parent.is_dir():
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), path.parent)
        ext = path.suffix
        buf = self.encode(ext, *params)
        with open(path, mode="w+b") as f:
            buf.tofile(f)

    def encode(self, ext: str, *params: int) -> npt.NDArray[np.uint8]:
        result, buf = cv2.imencode(ext, self, params)
        if not result:
            raise ValueError("Failed to decode image")
        return buf

    def crop(self, rect: Rect) -> Self:
        """矩形領域で切り抜く.

        返り値はビューであり, メモリは共有されることに注意.

        Args:
            rect (Rect): 切り抜く領域. img のピクセル数によって指定された矩形. 画像をはみ出す部分は無視される.
        Returns:
            Self: 画像のスライス.
        """
        if not rect:
            return self
        x1, y1, x2, y2 = rect.intersection(self.rect).to_xyxy_int_tuple()
        return self[y1:y2, x1:x2]

    def crop_fixed_ratio(self, aspect_ratio: float) -> Self:
        """指定したアスペクト比を持つ最大の領域で切り抜く.

        img が aspect_ratio より縦長の場合は上下, 横長の場合は左右が切り取られる.
        返り値はビューであり, メモリは共有されることに注意.

        Args:
            aspect_ratio (float): アスペクト比 (横/縦). img のピクセル数によって指定された矩形.
        Returns:
            Self: 画像のスライス.
        """
        cropping_rect = self.rect.fit_inner(aspect_ratio)
        return self.crop(cropping_rect)

    def put_img(self, img: Self, pos: VectorLike) -> Self:
        """指定した位置に画像を配置して上書きする.

        Args:
            img (Self): 配置する画像.
            pos (VectorLike): 配置する位置.

        Returns:
            Self: img を配置した新しいインスタンス.
        """
        rect = img.rect.move(*pos)
        intersection = self.rect.intersection(rect)

        w, h = intersection.size.to_int_tuple()
        x, y = map(int, pos)
        x2 = x + w
        y2 = y + h
        new = self.copy()
        new[y:y2, x:x2] = img[0:h, 0:w]
        return new

    def resize_img(self, new_wh: VectorLike) -> Self:
        new_wh = Vector2d(*new_wh).to_int_tuple()
        if self.wh == new_wh:
            return self.copy()
        return self.__class__(cv2.resize(self, new_wh))

    def scale_img(self, scale: float | VectorLike) -> Self:
        size = self.wh.scale(scale).to_int_tuple()
        return self.__class__(cv2.resize(self, size))

    def pad(
        self, color: ColorLike, px: Optional[int] = None, left=0, top=0, right=0, bot=0
    ) -> Self:
        if px:
            left = top = right = bot = px
        return self.__class__(
            cv2.copyMakeBorder(self, top, bot, left, right, cv2.BORDER_CONSTANT, value=color)
        )

    def create_mask_by_bgr(self, lower: BgrLike, upper: BgrLike) -> GrayImage:
        bgr = self.to_bgr()
        return GrayImage(cv2.inRange(bgr, lower, upper))

    def create_mask_by_bgr_pm(self, bgr: BgrLike, bgr_margins: tuple[int, int, int]) -> GrayImage:
        lower = Bgr(*map(lambda x: x[0] - x[1], zip(bgr, bgr_margins)))
        upper = Bgr(*map(lambda x: x[0] + x[1], zip(bgr, bgr_margins)))
        return self.create_mask_by_bgr(lower, upper)

    def create_mask_by_hsv(self, lower: HsvLike, upper: HsvLike) -> GrayImage:
        hsv = self.to_hsv()
        return GrayImage(cv2.inRange(hsv, lower, upper))

    def create_mask_by_hsv_pm(self, hsv: HsvLike, hsv_margins: tuple[int, int, int]) -> GrayImage:
        lower = Hsv(*map(lambda x: x[0] - x[1], zip(hsv, hsv_margins)))
        upper = Hsv(*map(lambda x: x[0] + x[1], zip(hsv, hsv_margins)))
        return self.create_mask_by_hsv(lower, upper)

    def canny(self, thresholld1: int, threshold2: int) -> GrayImage:
        return GrayImage(cv2.Canny(self, thresholld1, threshold2))

    def morph_open(self, kernel=np.ones((3, 3), np.uint8), iterations=1) -> Self:
        return self.__class__(cv2.morphologyEx(self, cv2.MORPH_OPEN, kernel, iterations=iterations))

    def morph_close(self, kernel=np.ones((3, 3), np.uint8), iterations=1) -> Self:
        return self.__class__(
            cv2.morphologyEx(self, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        )

    # - public @abstractmethod

    @abstractmethod
    def to_bgr(self) -> BgrImage:
        pass

    @abstractmethod
    def to_hsv(self) -> HsvImage:
        pass

    @abstractmethod
    def to_gray(self) -> GrayImage:
        pass

    # - prviate method

    def _is_valid(self) -> bool:
        ndim = self.ndim
        shape = self.shape
        if not self._DIMS:
            return ndim >= 2  # [x, y, ...]
        return is_valid_shape(shape, self._DIMS)


class BgrImage(Cv2Image):
    """Python OpenCV 用のBGR画像配列.

    実体は np.ndarray[np.dtype(np.uint8)] であり,
    次元数は3 (h, w, c).
    """

    _DIMS = (-1, -1, 3)
    _B = 0
    _G = 1
    _R = 2
    _IMREAD_FLAGS = cv2.IMREAD_COLOR

    def to_bgr(self) -> BgrImage:
        return self

    def to_hsv(self) -> HsvImage:
        return HsvImage(cv2.cvtColor(self, cv2.COLOR_BGR2HSV))

    def to_gray(self) -> GrayImage:
        return GrayImage(cv2.cvtColor(self, cv2.COLOR_BGR2GRAY))


class HsvImage(Cv2Image):
    """Python OpenCV 用のHSV画像配列.

    実体は np.ndarray[np.dtype(np.uint8)] であり,
    次元数は3 (h, w, c).
    """

    _DIMS = (-1, -1, 3)
    _H = 0
    _S = 1
    _V = 2
    _IMREAD_FLAGS = cv2.IMREAD_COLOR

    @classmethod
    def decode(cls, buf: npt.NDArray[np.uint8]) -> Self:
        return cls(BgrImage.decode(buf).to_hsv())

    def encode(self, ext: str, *params: int) -> npt.NDArray[np.uint8]:
        bgr = self.to_bgr()
        result, buf = cv2.imencode(ext, bgr, params)
        if not result:
            raise ValueError("Failed to decode image")
        return buf

    def to_bgr(self) -> BgrImage:
        return BgrImage(cv2.cvtColor(self, cv2.COLOR_HSV2BGR))

    def to_hsv(self) -> HsvImage:
        return self

    def to_gray(self) -> GrayImage:
        return self.to_bgr().to_gray()


class GrayImage(Cv2Image):
    """Python OpenCV 用のグレースケール画像配列.

    実体は np.ndarray[np.dtype(np.uint8)] であり,
    次元数は2 (h, w).
    """

    _DIMS = (-1, -1)

    def to_bgr(self) -> BgrImage:
        return BgrImage(cv2.cvtColor(self, cv2.COLOR_GRAY2BGR))

    def to_hsv(self) -> HsvImage:
        return self.to_bgr().to_hsv()

    def to_gray(self) -> GrayImage:
        return self

    def calc_white_ratio(self) -> float:
        white_pxs = cv2.countNonZero(self)
        return white_pxs / self.size


def is_valid_shape(shape_to_check: Sequence[int], shape_limitation: Sequence[int]) -> bool:
    if len(shape_to_check) != len(shape_limitation):
        return False
    return all(l == -1 or c == l for c, l in zip(shape_to_check, shape_limitation, strict=True))
