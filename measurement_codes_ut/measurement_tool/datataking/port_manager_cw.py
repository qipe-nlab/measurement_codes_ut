
from typing import Union, List, Any

import time


class PortManager(object):

    def __init__(self, device, dev_type):
        # self.port = Port(name, if_freq/1e9, max_amp=1.5)
        self.frequency = 8e9
        self.power = None
        self.current = 0.0
        self.status = False
        
        self.device = device
        self.type = dev_type

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.update_frequency()

    def update_frequency(self):
        try:
            time.sleep(0.01)
            if self.type=='VNA':
                self.device.start(self.frequency)
                self.device.stop(self.frequency)
                self.device.points(len(sweep['LO_freq']))
            else:
                self.device.frequency(self.frequency)
        except:
            pass

    def set_status(self, status):
        self.status = status
        self.update_status()

    def update_status(self):
        self.device.output(self.status)

    def set_power(self, power):
        self.power = power
        self.update_power()

    def update_power(self):
        try:
            self.device.power(self.power)
        except:
            pass

    def set_current(self, current):
        self.current = current
        self.update_current()

    def update_current(self):
        try:
            self.device.ramp_current(self.current, step=1e-8, delay=0)
        except:
            pass

