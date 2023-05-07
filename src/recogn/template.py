"""テンプレート画像モジュール."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Optional, Tuple, TypeVar, Union

import cv2

import cv2wrap as cv2wrap
from cv2wrap.color import Cv2Image
from geometry import Rect

_T = TypeVar("_T")


def _do_nothing(x: _T) -> _T:
    return x


@dataclass(frozen=True)
class Template(Generic[_T]):
    """テンプレート画像データクラス.

    Args:
        img (Cv2Image): テンプレート画像.
        value (T): マッチした際に返す値.
        ref_res: 基準解像度 (幅, 高さ). テンプレート画像を切り抜く前の画面全体の解像度.
        path (Optional[Path]): (ファイルから読み込んだ場合の) テンプレート画像のパス. Defaults to None.
        preprocess(Callable[[Cv2Image], Cv2Image]]): テンプレート画像に対して行う前処理. Defaults to do nothing.
    """

    img: Cv2Image
    value: _T
    ref_res: Tuple[int, int]
    path: Optional[Path] = None
    preprocess: Callable[[Cv2Image], Cv2Image] = _do_nothing
    # TODO: fix resolution

    def fit_resolution(self, new_res: Tuple[int, int]) -> Template[_T]:
        if self.ref_res == new_res:
            return self
        scale_x, scale_y = (new_res[i] / self.ref_res[i] for i in (0, 1))
        resized = cv2.resize(self.img, (0, 0), fx=scale_x, fy=scale_y)
        return type(self)(resized, self.value, new_res, self.path, self.preprocess)

    @classmethod
    def load(
        cls,
        file: Union[str, Path],
        value: _T,
        ref_res: Tuple[int, int],
        cropping: Optional[Rect] = None,
        preprocess: Optional[Callable[[Cv2Image], Cv2Image]] = None,
    ) -> Template[_T]:
        path = Path(file)
        img = cv2wrap.read(str(file))
        if cropping:
            img = cv2wrap.crop_img(img, cropping)

        if preprocess:
            img = preprocess(img)
        else:
            preprocess = _do_nothing
        return cls(img, value, ref_res, path, preprocess)
