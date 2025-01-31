from .baseclass import Broadcaster


class NullBroadcaster(Broadcaster):
    def broadcast_message(self, severity, ctxpath, date, message):
        pass
