from enum import IntEnum


class ReturnStatus(IntEnum) :
    SUCCESS = 0
    SERVER_ERROR = -1
    LOCAL_TRACKER_ERROR = -2
    NO_OP = -99


class TrackingState(IntEnum) :
    TRACKING_ON = 0
    TRACKING_OFF = -1
    WAIT_FOR_NEXT_TIME_POINT = -2