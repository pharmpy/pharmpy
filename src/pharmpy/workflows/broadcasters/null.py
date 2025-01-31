from .baseclass import Broadcaster


class NullBroadcaster(Broadcaster):
    def broadcast_message(severity, ctxpath, date, message):
        pass
