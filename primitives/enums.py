from enum import Enum, IntEnum


class CalibrationMode(Enum):
    """
    Represents calibration mode. If it was done by user or auto.
    """
    AUTOMATIC = 0
    LIGHTS_MANUAL = 1
    CORRIDORS_MANUAL = 2
    MANUAL = 3

    def __str__(self):
        return self.name.lower()


class Color(IntEnum):
    """
    Represents color states of traffic light
    """

    ORANGE = 0
    RED = 1
    RED_ORANGE = 2
    GREEN = 3
    NONE = 4


class Mode(Enum):
    """
    PipeBlock modes
    """

    CALIBRATION_VP = 0,
    CALIBRATION_CORRIDORS = 1,
    DETECTION = 2,
    SIGNAL = 3,
    END = 4,
