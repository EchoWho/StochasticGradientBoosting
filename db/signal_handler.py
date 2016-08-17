#!/usr/bin/env python
import signal


class SignalHandler(object):

    def __init__(self):
        self.sigint_capture = False
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signal, frame):
        print 'Interrupt Captured'
        self.sigint_capture = True

    def captured(self):
        return self.sigint_capture

    def reset(self):
        self.sigint_capture = False
