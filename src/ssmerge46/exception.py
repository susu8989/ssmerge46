"""Custom Exceptions."""
from abc import ABC


class BaseBotError(ABC, Exception):
    """Botの処理中に発生した例外の基底."""

    # このパッケージ以下の例外は全てこのクラスを規定とする.


# ----


class BotConfigError(BaseBotError):
    """環境設定に失敗した場合の例外."""


# ----


class BotProcessingError(BaseBotError):
    """Botがメッセージ処理中に発生した例外."""


class InvalidInputError(BotProcessingError):
    """無効な入力データを受け取った場合の例外."""


class InvalidSettingError(InvalidInputError):
    """無効な入力パラメータを受け取った場合の例外."""


class InvalidInputImageError(InvalidInputError):
    """無効な入力画像を."""


class DetectionError(BotProcessingError):
    """画像の結合位置検出ロジック中でエラーが発生した場合の例外."""


class ScrollbarDetectionError(DetectionError):
    """画像からスクロールバーを検出できなかった場合の例外."""


class OverlapDetectionError(DetectionError):
    """画像から結合位置を検出できなかった場合の例外."""


class ImageProcessingError(BotProcessingError):
    """画像処理に失敗した場合の例外."""
