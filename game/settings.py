# Settings for the game

# Number of AI factions to spawn when the game begins.
AI_FACTION_COUNT = 3

# Size of the map (width, height)
MAP_SIZE = (10, 10)

# Minimum distance from player's settlement to spawn AI
MIN_DISTANCE_FROM_PLAYER = 2
AI_SPAWN_MAX_ATTEMPTS_MULTIPLIER = 3

# Duration of one game tick in seconds
TICK_SECONDS = 1

# Desired total game length in seconds. Tuning this value scales
# resource generation rates, building costs and project build times
# so that typical progress peaks around this duration.
TARGET_GAME_LENGTH = 600

# Baseline length used when numbers were originally balanced.  The
# scaling factor stays 1.0 with the default target above.
_BASE_GAME_LENGTH = 600

# Number of ticks expected for a full game at the target length
TARGET_TICKS = int(TARGET_GAME_LENGTH // TICK_SECONDS)

# Universal scaling factor applied to economic values
SCALE_FACTOR = TARGET_GAME_LENGTH / _BASE_GAME_LENGTH
