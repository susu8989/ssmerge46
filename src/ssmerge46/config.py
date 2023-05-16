from distutils.util import strtobool
from os import environ

TOKEN: str = environ["TOKEN"]
MASANORI: bool = bool(strtobool(environ.get("MASANORI", "0")))

MAX_RESOLUTION: int = 1080 * 1920
