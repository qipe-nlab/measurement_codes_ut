import numpy as np
from logging import getLogger

import numpy as np
import qcodes as qc
from qcodes.instrument_drivers.rohde_schwarz.SGS100A import RohdeSchwarz_SGS100A
from qcodes_drivers.E82x7 import E82x7
from qcodes_drivers.N51x1 import N51x1
from qcodes_drivers.M3102A import M3102A
from qcodes_drivers.M3202A import M3202A
from qcodes.instrument_drivers.yokogawa.GS200 import GS200
from qcodes_drivers.E4407B import E4407B
from qcodes_drivers.N5222A import N5222A
from qcodes_drivers.M9804A import M9804A
from qcodes_drivers.E5071C import E5071C

from .port_manager_cw import PortManager

import matplotlib.pyplot as plt

from measurement_codes_ut.measurement_tool import Session

logger = getLogger(__name__)


class InstrumentManagerBase(object):
    """Insturment management class for CW measurement"""

    def __init__(self, session: Session, save_path: str) -> None:
        """Constructor of CW measurement

        Args:
            session (Session): session of measurement
        """

        self.session = session
        self.station = qc.Station()

        self.lo = None
        self.lo_info = {"model": [], "address": [], "port": []}

        self.vna = None
        self.vna_info = {"model": [], "address": [], "port": []}
        
        self.current_source = None
        self.current_info = {"model": [],
                             "address": [], "port": []}
        
        self.port = {}


        self.lo_id = 0

        self.tags = ["CW", session.cooling_down_id, session.package_name]
        if save_path[-1] != "/" or save_path[-2:] != "\\":
            save_path += "\\"
        self.save_path = save_path

    def add_readout_line(self,
                         port_name: str,
                         vna_address: str,
                         ):
        if self.vna is not None:
            raise ValueError("More than 1 VNA is allocated.")
        
        vna_dummy = N5222A(f"lo_{self.lo_id}", vna_address)
        model = vna_dummy.IDN()['model']
        self.vna_info['model'].append(model)
        self.vna_info['address'].append(vna_address)
        self.vna_info['port'].append(port_name)
        vna_dummy.close()
        # print(model)
        if "N52" in model:
            vna = N5222A("vna", vna_address)
            vna.aux1.output_polarity("negative")
            vna.aux1.output_position("after")
            vna.aux1.trigger_mode("point")
            vna.meas_trigger_input_type("level")
            vna.meas_trigger_input_polarity("positive")
        elif "M98" in model:
            vna = M9804A("vna", vna_address)
            vna.aux_trig_1_output_polarity("negative")
            vna.aux_trig_1_output_position("after")
            vna.aux_trig_1_output_interval("point")
            vna.meas_trigger_input_type("level")
            vna.meas_trigger_input_polarity("positive")
        elif "E50" in model:
            raise ValueError("ENA-type device is not supported now.")
            # vna = E5071C("vna", vna_address)
            # vna.trigger_output_polarity("negative")
            # vna.trigger_output_position("after")
            # # vna.aux1.trigger_mode("point")
            # # vna.trigger_input_type("level")
            # vna.trigger_input_polarity("positive")

        
        vna.electrical_delay(0)  # s
        
        self.station.add_component(vna)
        self.vna = vna

        self.port[port_name] = PortManager(self.vna, "VNA")

    def add_drive_line(self,
                       port_name: str,
                       lo_address: str,
                       lo_power: int,):
        
        if self.lo is not None:
            raise ValueError("More than 1 LO is allocated.")
        
        lo_dummy = E82x7(f"lo_{self.lo_id}", lo_address)
        model = lo_dummy.IDN()['model']
        self.lo_info['model'].append(model)
        self.lo_info['address'].append(lo_address)
        self.lo_info['port'].append(port_name)
        lo_dummy.close()
        # print(model)
        if "E82" in model:
            lo = E82x7(f"lo_{self.lo_id}", lo_address)
            lo.trigger_input_slope("negative")
            lo.source_settled_polarity("low")
            lo.output(False)
        elif "N51" in model:
            lo = N51x1(f"lo_{self.lo_id}", lo_address)
            lo.output(False)
            print("Drive source other than E82x7 is not supported. May occur unexpected things.")
        elif "SGS" in model:
            lo = RohdeSchwarz_SGS100A(f"lo_{self.lo_id}", lo_address)
            lo.off()
            print("Drive source other than E82x7 is not supported. May occur unexpected things.")
        else:
            raise ValueError(f"Model {model} is not supported.")
        
        lo.power(lo_power)
        lo.frequency(10e9)
        self.lo = lo
        self.lo_address[self.lo_id] = lo_address
        self.station.add_component(lo)
        self.lo_id += 1

        self.port[port_name] = PortManager(self.lo, "LO")

    def add_current_source_bias_line(self, 
                                     port_name: str,
                                     current_source_address: str) -> None:
        """add current source line
        Args:
            port_name (str): port name
            current_source_adress (str): IP address of current source like "TCPIP0::192.168.100.8::inst0::INSTR"
        """

        
        if self.current_source is not None:
            raise ValueError("More than 1 current source is allocated.")
        self.current_info['model'].append("GS200")
        self.current_info['address'].append(current_source_address)
        self.current_info['port'].append(port_name)
        current_source = GS200(
            port_name+"_current_source", current_source_address)
        # current_source.ramp_current(0e-6, step=1e-7, delay=0)
        self.current_source = current_source
            # self.station.add_component(current_source)

            
        self.port[port_name] = PortManager(self.current_source, "Current source")

    def add_spectrum_analyzer(self, name="spectrum_analyzer", address='GPIB0::16::INSTR'):
        spectrum_analyzer = E4407B(name, address)
        self.spectrum_analyzer = spectrum_analyzer

    def set_wiring_note(self, wiring_info):
        self.wiring_info = wiring_info

    def __repr__(self):
        s = ""
        s += "*** Allocated devices and channel assignemnt ***\n"
        s += "{:20} {:20} {:40} {:15}\n".format(
            "device type", "device name", "device address", "port")
        s += "-" * (20 + 20 + 40  + 15) + "\n"
        
        try:
            for _ in range(len(self.vna_info['model'])):
                device_type = "VNA"
                device_name = self.vna_info['model'][_]
                address = self.vna_info['address'][_]
                dep_port = self.vna_info['port'][_]

                s += f"{device_type:20} {device_name:20} {address:40} {dep_port:15}\n"

        except:
            pass

        try:
            for _ in range(len(self.lo_info['model'])):
                device_type = "LO"
                device_name = self.lo_info['model'][_]
                address = self.lo_info['address'][_]
                dep_port = self.lo_info['port'][_]

                s += f"{device_type:20} {device_name:20} {address:40} {dep_port:15}\n"

        except:
            pass

        try:
        
            for _ in range(len(self.current_info['model'])):
                device_type = "Current source"
                device_name = self.current_info['model'][_]
                address = self.current_info['address'][_]
                dep_port = self.current_info['port'][_]

                s += f"{device_type:20} {device_name:20} {address:40} {dep_port:15}\n"

        except:
            pass

        s += "\n\n*** Current source status ***\n"
        s += "{:20} {:20}\n".format(
            "port name", "current (mA)")
        s += "-" * (20 + 20) + "\n"

        try:
            for key, value in self.current_source.items():
                name = key
                cur = value.current()*1e3

                s += f"{name:<20} {cur:<20.6f}\n"
        except:
            pass

        return s

    def close_all(self):
        for name, lo in self.lo.items():
            try:
                lo.close()
                print(f"Connection to LO {name} closed.")
            except:
                pass

        for name, vna in self.vna.items():
            try:
                vna.close()
                print(f"Connection to VNA {name} closed.")
            except:
                pass
        for name, cs in self.current_source.items():
            try:
                cs.close()
                print(f"Connection to Current Source {name} closed.")
            except:
                pass

        self.lo = {}
        self.lo_address = {}
        self.lo_info = {"model": [], "address": [], "port": []}

        self.vna = {}
        self.vna_info = {"model": [], "address": [], "port": []}
        self.current_source = {}
        self.current_info = {"model": [],
                             "address": [],  "port": []}

        self.lo_id = 0
