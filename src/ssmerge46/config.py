"""環境変数と定数値."""

import logging
from os import environ
from pathlib import Path

import discord
import toml
from toml import TomlDecodeError

from ssmerge46.exception import BotConfigError

log = logging.getLogger("discord")


def _to_bool(val: str) -> bool:
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"Failed to convert value. : {val}")


def _get_version() -> str:
    try:
        d = toml.load(Path(__file__).parent.parent.parent / "pyproject.toml")
        return d["tool"]["poetry"]["version"]
    except (TypeError, TomlDecodeError) as e:
        log.warning(e, exc_info=True)
        return ""


VERSION = _get_version()

# Token for discord bot
TOKEN = environ.get("TOKEN", "")
if not TOKEN:
    raise BotConfigError("'TOKEN' environment variable is required.")
# Guild ID
GUILD = discord.Object(int(environ.get("GUILD", 0)))

FLASK_PORT = int(environ.get("FLASK_PORT", 8080))

# Max number of input files per request
MAX_ATTACHMENTS = int(environ.get("MAX_ATTACHMENTS", 12))
# Max number of pixels per input file (if num of imgs <= 4)
MAX_RESOLUTION = int(environ.get("MAX_RESOLUTION", 1080 * 1920))
# Max number of pixels per input file (if num of imgs <= 9)
MAX_RESOLUTION_2 = int(environ.get("MAX_RESOLUTION", 720 * 1280))
# Max number of pixels per input file
MAX_RESOLUTION_3 = int(environ.get("MAX_RESOLUTION", 540 * 960))

# Private joke
MASANORI = _to_bool(environ.get("MASANORI", "0"))
MASANORI_RATE = float(environ.get("MASANORI_RATE", 0.03))

AVAILABLE_FILETYPES = ("png", "jpg", "jpeg", "webp")
