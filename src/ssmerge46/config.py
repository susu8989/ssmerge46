from distutils.util import strtobool
from os import environ

# Token for discord bot
TOKEN = environ["TOKEN"]

# Max number of input files per request
MAX_ATTACHMENTS = int(environ.get("MAX_ATTACHMENTS", 12))
# Max number of pixels per input file
MAX_RESOLUTION = int(environ.get("MAX_RESOLUTION", 1080 * 1920))

# Private joke
MASANORI = bool(strtobool(environ.get("MASANORI", "0")))
