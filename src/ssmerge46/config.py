"""環境変数と定数値."""
from os import environ

from ssmerge46.exception import BotConfigError


def _to_bool(val: str) -> bool:
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"Failed to convert value. : {val}")


# Token for discord bot
TOKEN = environ.get("TOKEN", "")
if not TOKEN:
    raise BotConfigError("'TOKEN' environment variable is required.")

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
