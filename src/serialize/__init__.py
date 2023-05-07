from pathlib import Path
from typing import Type, TypeVar

from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.mixins.yaml import DataClassYAMLMixin

T = TypeVar("T", bound="Serializable")


class Serializable(DataClassJSONMixin, DataClassYAMLMixin):
    def save_json(self: T, dest: Path, encoding: str = "utf-8") -> None:
        write_to_file(str(self.to_json()), dest, encoding=encoding)

    @classmethod
    def load_json(cls: Type[T], src: Path, encoding: str = "utf-8") -> T:
        text = read_from_file(src, encoding=encoding)
        return cls.from_json(text)

    def save_yaml(self, dest: Path, encoding: str = "utf-8") -> None:
        write_to_file(str(self.to_yaml()), dest, encoding=encoding)

    @classmethod
    def load_yaml(cls: Type[T], src: Path, encoding: str = "utf-8") -> T:
        text = read_from_file(src, encoding=encoding)
        return cls.from_yaml(text)


def write_to_file(data: str, dest: Path, encoding: str) -> None:
    with open(dest, mode="w", encoding=encoding) as f:
        f.write(data)


def read_from_file(src: Path, encoding: str) -> str:
    with open(src, mode="r", encoding=encoding) as f:
        return f.read()


def load_json(clazz: Type[T], src: Path, encoding: str = "utf-8") -> T:
    text = read_from_file(src, encoding=encoding)
    return clazz.from_json(text)


def load_yaml(clazz: type[T], src: Path, encoding: str = "utf-8") -> T:
    text = read_from_file(src, encoding=encoding)
    return clazz.from_yaml(text)
