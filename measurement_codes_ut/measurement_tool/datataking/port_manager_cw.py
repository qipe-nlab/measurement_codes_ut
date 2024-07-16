
from typing import Union, List, Any

import time


class PortManager(object):

    def __init__(self, device, dev_type):
        """class for organizing ports used in CW

        Args:
            device (Instrument): Instrument object
            dev_type (int): Index to specify device type
                            0: VNA
                            1: LO
                            2: Current source
        """
        # self.port = Port(name, if_freq/1e9, max_amp=1.5)
        if dev_type != 2:
            self.frequency = 8e9
            self.power = 0
        else:
            self.current = 0.0
        self.status = False
        
        self.device = device
        self.type = dev_type
        self.sweep_index = 0

    def set_frequency(self, frequency):
        if self.type != 2:
            self.frequency = frequency
            self.update_frequency()

    def update_frequency(self):
        if self.type != 2:
            time.sleep(0.01)
            self.device.frequency(self.frequency)
        # except:
        #     pass

    def set_status(self, status):
        self.status = status
        self.update_status()

    def update_status(self):
        self.device.output(self.status)

    def enable_output(self):
        self.device.output(True)
        
    def disable_output(self):
        self.device.output(False)

    def set_power(self, power):
        if self.type != 2:
            self.power = power
            self.update_power()

    def update_power(self):
        if self.type != 2:
            self.device.power(self.power)

    def set_current(self, current):
        if self.type == 2:
            self.current = current
            self.update_current()
            

    def update_current(self):
        if self.type == 2:
            self.device.ramp_current(self.current, step=5e-8, delay=0)

