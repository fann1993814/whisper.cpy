def to_timestamp(t: int, comma: bool):
    msec = int(t * 10)
    hours = int(msec / (1000 * 60 * 60))
    msec = int(msec - hours * (1000 * 60 * 60))
    minutes = int(msec / (1000 * 60))
    msec = int(msec - minutes * (1000 * 60))
    sec = int(msec / 1000)
    msec = int(msec - sec * 1000)

    return "{:02d}:{:02d}:{:02d}{}{:03d}".format(
        hours, minutes, sec, "," if comma else ".", msec)


def empty_log_callback(level, text, user_data):
    pass
