
from typing import Union, List, Any

from sequence_parser import Port, Sequence

import time


class PortManager(object):

    def __init__(self, name: str, lo, if_freq: int, sideband: str):
        self.port = Port(name, if_freq/1e9, max_amp=1.5)
        self.port_name = name
        self.frequency = 8e9
        self.status = False
        if sideband == 'lower':
            self.sideband = +1
        else:
            self.sideband = -1
        self.lo = lo
        self.window = None

    def set_frequency(self, frequency):
        self.frequency = frequency
        self.update_frequency()

    def update_frequency(self):
        time.sleep(0.01)
        self.lo.frequency(self.frequency + self.sideband*self.port.if_freq*1e9)

    def set_status(self, status):
        self.status = status
        self.update_status()

    def update_status(self):
        if "Rohde" in self.lo.IDN()['model']:
            if self.status:
                self.lo.on()
            else:
                self.lo.off()
        else:
            self.lo.output(self.status)

    def __repr__(self) -> str:
        """Return string representation of Port

        Returns:
            str: string representation
        """
        repr_str = ""
        repr_str += "Port name = {}\n".format(self.port_name)
        repr_str += "Port frequency (GHz) = \"{}\"\n".format(self.frequency*1e-9)
        repr_str += "Port IF frequency (MHz) = {}\n".format(self.port.if_freq*1e3)
        

        return repr_str
