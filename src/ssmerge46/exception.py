from abc import ABC


class InvalidSettingError(Exception):
    pass


class InvalidInputImageError(Exception):
    pass


class DetectionError(ABC, Exception):
    pass


class ScrollbarDetectionError(DetectionError):
    pass


class OverlapDetectionError(DetectionError):
    pass


class CroppingError(Exception):
    pass
