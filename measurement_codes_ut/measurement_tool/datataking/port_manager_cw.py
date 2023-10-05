
from typing import Union, List, Any

import time


class PortManager(object):
        
    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

            
    def __getattr__(self, name):
        return name
        

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.update_frequency()

    def update_frequency(self):
        try:
            time.sleep(0.01)
            self.device.frequency(self.frequency)
        except:
            pass

    def set_status(self, status):
        self.status = status
        self.update_status()

    def update_status(self):
        self.device.output(self.status)

    def enable_output(self):
        try:
            self.device.output(True)
        except:
            self.device.on()

    def disable_output(self):
        try:
            self.device.output(False)
        except:
            self.device.off()


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

