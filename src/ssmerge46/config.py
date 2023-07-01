from distutils.util import strtobool
from os import environ

# Token for discord bot
TOKEN = environ["TOKEN"]

# Max number of input files per request
MAX_ATTACHMENTS = int(environ.get("MAX_ATTACHMENTS", 12))
# Max number of pixels per input file (if num of imgs <= 4)
MAX_RESOLUTION = int(environ.get("MAX_RESOLUTION", 1080 * 1920))
# Max number of pixels per input file (if num of imgs <= 9)
MAX_RESOLUTION_2 = int(environ.get("MAX_RESOLUTION", 720 * 1280))
# Max number of pixels per input file
MAX_RESOLUTION_3 = int(environ.get("MAX_RESOLUTION", 540 * 960))

# Private joke
MASANORI = bool(strtobool(environ.get("MASANORI", "0")))
