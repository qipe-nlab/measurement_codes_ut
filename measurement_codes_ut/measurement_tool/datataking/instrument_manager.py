import numpy as np
from logging import getLogger
import time

import numpy as np
import qcodes as qc
from qcodes_drivers.SGS100A import SGS100A
from qcodes_drivers.E82x7 import E82x7
from qcodes_drivers.N51x1 import N51x1
from qcodes_drivers.HVI_Trigger import HVI_Trigger
from qcodes_drivers.iq_corrector import IQCorrector
from qcodes_drivers.M3102A import M3102A
from qcodes_drivers.M3202A import M3202A
from qcodes.instrument_drivers.yokogawa.GS200 import GS200
from qcodes_drivers.E4407B import E4407B
from ...drivers.HS900xB import HS900xB
from ...drivers.APMSYN22 import APMSYN22
from ...drivers.LDA import Vaunix_LDA
from ...drivers.LDA_eth import Vaunix_LDA_Eth as LDA_eth
from ...drivers.SGS100A import SGS100A
from ...drivers.Valon501x import Valon501x

from sequence_parser import Port, Sequence
from sequence_parser.iq_port import IQPort

import matplotlib.pyplot as plt

from .. import Session
from qcodes_drivers.iq_corrector import IQCorrector
from qcodes.instrument.base import Instrument
from .port_manager import PortManager

logger = getLogger(__name__)


class InstrumentManagerBase(object):
    """Insturment management class for timedomain measurement"""

    def __init__(self, session: Session, trigger_address: str) -> None:
        """Constructor of time domain measurement

        Args:
            session (Session): session of measurement
            trigger_address (str): address of HVI trigger
            save_path (str): data save path
        """
        print("Creating a new insturment management class for timedomain measurement...", end="")

        self.session = session
        self.station = qc.Station()

        self.reset_manager()

        self.averaging_shot = False
        self.averaging_waveform = False

        self.tags = ["TD", session.cooling_down_id, session.package_name]
        
        self.save_path = session.save_path

        self.trigger_address = trigger_address
        self.repetition_margin = 50000
        self.acquisition_delay = 0
        self.num_shot = 1000

        if trigger_address != "":
            hvi_trigger = HVI_Trigger(
                "hvi_trigger", self.trigger_address, debug=False)
            hvi_trigger.output(False)
            hvi_trigger.digitizer_delay(self.acquisition_delay)
            hvi_trigger.trigger_period(self.repetition_margin)
            self.hvi_trigger = hvi_trigger

            self.station.add_component(hvi_trigger)

        print("done")

    def add_misc_control_line(self, port_name: str,
                              lo_address: str,
                              lo_power: int,
                              awg_chasis: int,
                              awg_slot: int,
                              awg_channel,
                              IQ_corrector,
                              if_freq: float,
                              sideband: str) -> None:
        """add misc control line

        Args:
            port_name (str): port name
            lo_adress (str): IP address of LO like "TCPIP0::192.168.100.8::inst0::INSTR"
            awg_chasis (int): Chasis index of AWG 
            awg_slot (int): Slot index of AWG
            channel_name (str): Channel name like "qubit", "readout", "ef", "fogi"
        """
        if isinstance(lo_address, tuple):
            addr, channel = lo_address
            
            if addr not in self.lo_address.values():
                lo_base = HS900xB(f"lo_{self.lo_id}", addr)
                model = lo_base.IDN()['model']
                self.lo_info['model'].append(model)
                self.lo_info['address'].append(addr)
                self.lo_info['channel'].append(f"CH{channel}")
                self.lo_info['port'].append(port_name)
                self.lo_shared[addr] = lo_base
            else:
                lo_base = self.lo_shared[addr]
                model = lo_base.IDN()['model']
                self.lo_info['model'].append(model)
                self.lo_info['address'].append(addr)
                self.lo_info['channel'].append(f"CH{channel}")
                self.lo_info['port'].append(port_name)
            ref_dict = {1: lo_base.CH1, 2: lo_base.CH2, 3: lo_base.CH3, 4: lo_base.CH4, 5: lo_base.CH5}
            lo = ref_dict[channel]
            lo.output(False)
            lo.power(lo_power)
            lo.parent.ref("ext10")
            lo.frequency(10e9 - if_freq)
            self.lo[port_name] = lo
            self.lo_address[self.lo_id] = addr
            self.station.add_component(lo)
            self.lo_id += 1

        elif lo_address != "":
            lo_dummy = E82x7(f"lo_{self.lo_id}", lo_address)
            model = lo_dummy.IDN()['model']
            self.lo_info['model'].append(model)
            self.lo_info['address'].append(lo_address)
            self.lo_info['channel'].append("")
            self.lo_info['port'].append(port_name)
            lo_dummy.close()
            # print(model)
            if lo_address not in self.lo_address.values():
                if "E82" in model:
                    lo = E82x7(f"lo_{self.lo_id}", lo_address)
                elif "N51" in model:
                    lo = N51x1(f"lo_{self.lo_id}", lo_address)
                elif "SGS" in model:
                    lo = SGS100A(f"lo_{self.lo_id}", lo_address)
                elif "AP" in model:
                    lo = APMSYN22(f"lo_{self.lo_id}", lo_address)
                elif "Valon" in model:
                    lo = Valon501x(f"lo_{self.lo_id}", lo_address)
                else:
                    raise ValueError(f"LO {model} is not supported.")
                # elif "M98" in model:
                #     lo = M9804A_LO(f"lo_{self.lo_id}", lo_address)
                #     lo.sweep_mode('continuous')
                #     lo.trigger_source("manual")

                lo.output(False)
                lo.power(lo_power)
                lo.frequency(10e9 - if_freq)
                self.lo[port_name] = lo
                self.lo_address[self.lo_id] = lo_address
                self.station.add_component(lo)
                self.lo_id += 1
            else:
                print(f"LO {lo_address} already allocated.")
        else:
                self.lo[port_name] = None
                self.lo_address[self.lo_id] = lo_address

        self.awg_info['model'].append("M3202A")
        self.awg_info['address'].append("")
        self.awg_info['channel'].append(
            f"chasis{awg_chasis} slot{awg_slot} ch{awg_channel}")
        self.awg_info['port'].append(port_name)
        if {'chasis': awg_chasis, 'slot': awg_slot} not in self.awg_ref.values():
            awg = M3202A(f"awg_{self.awg_id}",
                         chassis=awg_chasis, slot=awg_slot)
            awg.channels.stop()
            awg.flush_waveform()
            self.station.add_component(awg)
            self.awg[self.awg_id] = awg
            ref_dict = {1: awg.ch1, 2: awg.ch2, 3: awg.ch3, 4: awg.ch4}
            if isinstance(awg_channel, int):
                awg_ch = ref_dict[awg_channel]
            else:
                awg_ch = ref_dict[awg_channel[0]], ref_dict[awg_channel[1]]

            self.awg_ch[port_name] = self.awg_id, awg_ch
            if IQ_corrector is None:
                self.IQ_corrector[port_name] = None
            else:
                self.IQ_corrector[port_name] = IQCorrector(
                    awg_ch[0],
                    awg_ch[1],
                    IQ_corrector["calibration_path"],
                    lo_leakage_datetime=IQ_corrector["lo_leakage_datetime"],
                    rf_power_datetime=IQ_corrector["rf_power_datetime"],
                    len_kernel=41,
                    fit_weight=10,
                )
            self.awg_id += 1

        else:
            key = [k for k, v in self.awg_ref.items() if v == {
                'chasis': awg_chasis, 'slot': awg_slot}][0]
            idx = self.awg_ch[key][0]
            awg = self.awg[idx]
            ref_dict = {1: awg.ch1, 2: awg.ch2, 3: awg.ch3, 4: awg.ch4}
            if isinstance(awg_channel, int):
                awg_ch = ref_dict[awg_channel]
            else:
                awg_ch = ref_dict[awg_channel[0]], ref_dict[awg_channel[1]]
            self.awg_ch[port_name] = idx, awg_ch
            if IQ_corrector is None:
                self.IQ_corrector[port_name] = None
            else:
                self.IQ_corrector[port_name] = IQCorrector(
                    awg_ch[0],
                    awg_ch[1],
                    IQ_corrector["calibration_path"],
                    lo_leakage_datetime=IQ_corrector["lo_leakage_datetime"],
                    rf_power_datetime=IQ_corrector["rf_power_datetime"],
                    len_kernel=41,
                    fit_weight=10,
                )

        self.awg_ref[port_name] = {'chasis': awg_chasis, 'slot': awg_slot}
        self.port[port_name] = PortManager(
            port_name, self.lo[port_name], if_freq, sideband)

    def add_readout_line(self, port_name: str,
                         lo_address: str,
                         lo_power: int,
                         awg_chasis: int,
                         awg_slot: int,
                         awg_channel,
                         dig_chasis: int,
                         dig_slot: int,
                         dig_channel,
                         IQ_corrector,
                         if_freq: float = 125e6,
                         sideband: str = "lower") -> None:
        """add readout line
        Args:
            port_name (str): port name
            lo_adress (str): IP address of LO like "TCPIP0::192.168.100.8::inst0::INSTR"
            awg_chasis (int): Chasis index of AWG 
            awg_slot (int): Slot index of AWG
            awg_channel : int or tuple
            dig_chasis: int,
            dig_slot: int,
            dig_channel,
            IQ_corrector,
            if_freq: float
        """

        if not "readout" in port_name:
            raise ValueError(f'Port name must include "readout".')

        self.add_misc_control_line(
            port_name, lo_address, lo_power, awg_chasis, awg_slot, awg_channel, IQ_corrector, if_freq, sideband)

        self.dig_info['model'].append("M3102A")
        self.dig_info['address'].append("")
        self.dig_info['channel'].append(
            f"chasis{dig_chasis} slot{dig_slot} ch{dig_channel}")
        self.dig_info['port'].append(port_name)
        if port_name not in self.digitizer:
            dig = M3102A(f"dig_{self.dig_id}",
                         chassis=dig_chasis, slot=dig_slot)
            dig.channels.stop()
            self.digitizer[port_name] = dig
            self.station.add_component(dig)

            if dig_channel == 1:
                dig_ch = dig.ch1
            if dig_channel == 2:
                dig_ch = dig.ch2
            if dig_channel == 3:
                dig_ch = dig.ch3
            if dig_channel == 4:
                dig_ch = dig.ch4
            dig_ch.high_impedance(False)  # 50 Ohms
            dig_ch.half_range_50(4)  # V_pp / 2
            dig_ch.ac_coupling(False)  # dc coupling
            dig_ch.sampling_interval(2)  # ns
            dig_ch.trigger_mode("software/hvi")
            dig_ch.timeout(10000)

            self.digitizer_ch[port_name] = dig_ch
            self.acquire_port[port_name +
                              "_acquire"] = IQPort(port_name+"_acquire")
            self.dig_id += 1

        else:
            print(f"Digitizer already allocated.")

    def add_qubit_line(self, port_name: str,
                       lo_address: str,
                       lo_power: int,
                       awg_chasis: int,
                       awg_slot: int,
                       awg_channel: int,
                       IQ_corrector,
                       if_freq: float = 150e6,
                       sideband: str = "lower") -> None:
        """add single-qubit control line
        Args:
            port_name (str): port name
            lo_adress (str): IP address of LO like "TCPIP0::192.168.100.8::inst0::INSTR"
            awg_chasis (int): Chasis index of AWG 
            awg_slot (int): Slot index of AWG
        """
        self.add_misc_control_line(
            port_name, lo_address, lo_power, awg_chasis, awg_slot, awg_channel, IQ_corrector, if_freq, sideband)

    def add_current_source_bias_line(self, 
                                     port_name: str,
                                     current_source_address: str) -> None:
        """add current source line
        Args:
            port_name (str): port name
            current_source_adress (str): IP address of current source like "TCPIP0::192.168.100.8::inst0::INSTR"
        """
        self.current_info['model'].append("GS200")
        self.current_info['address'].append(current_source_address)
        self.current_info['channel'].append("")
        self.current_info['port'].append(port_name)
        if port_name not in self.current_source:
            current_source = GS200(
                port_name+"_current_source", current_source_address)
            # current_source.ramp_current(0e-6, step=1e-7, delay=0)
            self.current_source[port_name] = current_source
            # self.station.add_component(current_source)

        else:
            print(
                f"Current source {current_source_address} already allocated.")
            

    def add_cross_resonance_line(self, port_name: str,
                       port_name_source: str,
                       awg_chasis: int,
                       awg_slot: int,
                       awg_channel: int,
                       IQ_corrector,
                       if_freq: float = 150e6,
                       sideband: str = "lower") -> None:
        lo = self.lo[port_name_source]
        self.lo[port_name] = lo

        self.awg_info['model'].append("M3202A")
        self.awg_info['address'].append("")
        self.awg_info['channel'].append(
            f"chasis{awg_chasis} slot{awg_slot} ch{awg_channel}")
        self.awg_info['port'].append(port_name)
        if {'chasis': awg_chasis, 'slot': awg_slot} not in self.awg_ref.values():
            awg = M3202A(f"awg_{self.awg_id}",
                         chassis=awg_chasis, slot=awg_slot)
            awg.channels.stop()
            awg.flush_waveform()
            self.station.add_component(awg)
            self.awg[self.awg_id] = awg
            ref_dict = {1: awg.ch1, 2: awg.ch2, 3: awg.ch3, 4: awg.ch4}
            if isinstance(awg_channel, int):
                awg_ch = ref_dict[awg_channel]
            else:
                awg_ch = ref_dict[awg_channel[0]], ref_dict[awg_channel[1]]

            self.awg_ch[port_name] = self.awg_id, awg_ch
            if IQ_corrector is None:
                self.IQ_corrector[port_name] = None
            else:
                self.IQ_corrector[port_name] = IQCorrector(
                    awg_ch[0],
                    awg_ch[1],
                    IQ_corrector["calibration_path"],
                    lo_leakage_datetime=IQ_corrector["lo_leakage_datetime"],
                    rf_power_datetime=IQ_corrector["rf_power_datetime"],
                    len_kernel=41,
                    fit_weight=10,
                )
            self.awg_id += 1

        else:
            key = [k for k, v in self.awg_ref.items() if v == {
                'chasis': awg_chasis, 'slot': awg_slot}][0]
            idx = self.awg_ch[key][0]
            awg = self.awg[idx]
            ref_dict = {1: awg.ch1, 2: awg.ch2, 3: awg.ch3, 4: awg.ch4}
            if isinstance(awg_channel, int):
                awg_ch = ref_dict[awg_channel]
            else:
                awg_ch = ref_dict[awg_channel[0]], ref_dict[awg_channel[1]]
            self.awg_ch[port_name] = idx, awg_ch
            if IQ_corrector is None:
                self.IQ_corrector[port_name] = None
            else:
                self.IQ_corrector[port_name] = IQCorrector(
                    awg_ch[0],
                    awg_ch[1],
                    IQ_corrector["calibration_path"],
                    lo_leakage_datetime=IQ_corrector["lo_leakage_datetime"],
                    rf_power_datetime=IQ_corrector["rf_power_datetime"],
                    len_kernel=41,
                    fit_weight=10,
                )

        self.awg_ref[port_name] = {'chasis': awg_chasis, 'slot': awg_slot}
        
        self.port[port_name] = PortManager(
            port_name, lo, if_freq, sideband)
        

    def add_spectrum_analyzer(self, name="spectrum_analyzer", address='GPIB0::16::INSTR'):
        spectrum_analyzer = E4407B(name, address)
        self.spectrum_analyzer = spectrum_analyzer

    def add_variable_attenuator(self, name: str,
                       address: str,
                       channel: int) -> None:
        """method to add Vaunix variable attenuator 

        Args:
            name (str): name of the port
            address (str): IP address of the instrument
            channel (int): channel you use
        """
        vaunix_dll = r"F:\vaunix_dll"
        # vatt = LDA_eth('lda', address, dll_path=vaunix_dll, num_channels=4)
        if address not in self.vatt_info:
            vatt = LDA_eth(f'lda_{self.vatt_id}', address, dll_path=vaunix_dll, num_channels=4)
            self.station.add_component(vatt)
            self.vatt_info[address] = vatt
            self.vatt_id += 1
            # self.vatt_addr.append(address)
        else:
            vatt = self.vatt_info[address]
        ref_dict = {1: vatt.ch1, 2: vatt.ch2, 3: vatt.ch3, 4: vatt.ch4}
        vatt_ch = ref_dict[channel]
        self.vatt[name] = vatt_ch
    
    def add_iq_corrector(self, IQ_corrector, port_name, awg_channel):
        self.IQ_corrector[port_name] = IQCorrector(
            awg_channel[0],
            awg_channel[1],
            IQ_corrector["calibration_path"],
            lo_leakage_datetime=IQ_corrector["lo_leakage_datetime"],
            rf_power_datetime=IQ_corrector["rf_power_datetime"],
            len_kernel=41,
            fit_weight=10,
        )

        awg = self.awg_ch[port_name][0]
        self.awg_ch[port_name] = awg, awg_channel

    def set_wiring_note(self, wiring_info):
        self.wiring_info = wiring_info

    def set_acquisition_mode(self, averaging_shot, averaging_waveform):
        self.averaging_shot = averaging_shot
        self.averaging_waveform = averaging_waveform

    def set_average_window_coefficients(self, window):
        self.window = window

    def set_repetition_margin(self, time):
        self.repetition_margin = int(time)
        # self.hvi_trigger.trigger_period(int(time))

    def set_acquisition_delay(self, delay):
        self.acquisition_delay = int(delay//10+1) * 10
        # self.hvi_trigger.digitizer_delay(int(delay//10+1) * 10)

    def set_shots(self, num_shot):
        self.num_shot = num_shot

    def __repr__(self):
        s = ""
        s += "*** Allocated devices and channel assignemnt ***\n"
        s += "{:20} {:20} {:40} {:30} {:15}\n".format(
            "Device type", "Device name", "Device address", "Channel", "Port")
        s += "-" * (20 + 20 + 40 + 30 + 15) + "\n"
        for _ in range(len(self.lo_info['model'])):
            device_type = "LO"
            device_name = self.lo_info['model'][_]
            address = self.lo_info['address'][_]
            channel = self.lo_info['channel'][_]
            dep_port = self.lo_info['port'][_]

            s += f"{device_type:20} {device_name:20} {address:40} {channel:30} {dep_port:15}\n"

        for _ in range(len(self.awg_info['model'])):
            device_type = "AWG"
            device_name = self.awg_info['model'][_]
            address = self.awg_info['address'][_]
            channel = self.awg_info['channel'][_]
            dep_port = self.awg_info['port'][_]

            s += f"{device_type:20} {device_name:20} {address:40} {channel:30} {dep_port:15}\n"

        for _ in range(len(self.dig_info['model'])):
            device_type = "Digitizer"
            device_name = self.dig_info['model'][_]
            address = self.dig_info['address'][_]
            channel = self.dig_info['channel'][_]
            dep_port = self.dig_info['port'][_]

            s += f"{device_type:20} {device_name:20} {address:40} {channel:30} {dep_port:15}\n"

        for _ in range(len(self.current_info['model'])):
            device_type = "Current source"
            device_name = self.current_info['model'][_]
            address = self.current_info['address'][_]
            channel = self.current_info['channel'][_]
            dep_port = self.current_info['port'][_]

            s += f"{device_type:20} {device_name:20} {address:40} {channel:30} {dep_port:15}\n"

        for key, device in self.vatt.items():
            device_type = "LDA"
            device_name = device._parent.IDN()['model']
            address = device._parent.ip_address
            channel = f"ch{device.channel_number}"
            dep_port = key

            s += f"{device_type:20} {device_name:20} {address:40} {channel:30} {dep_port:15}\n"

        device_type = "HVI trigger"
        device_name = self.hvi_trigger.IDN()['model']
        address = self.trigger_address
        channel = ""
        dep_port = ""

        s += f"{device_type:20} {device_name:20} {address:40} {channel:30} {dep_port:15}\n"

        s += "\n\n*** Port status ***\n"
        s += "{:20} {:20} {:20}\n".format(
            "Port name", "Frequency (GHz)", "IF frequency (MHz)")
        s += "-" * (20 + 20 + 20) + "\n"
        for key, value in self.port.items():
            name = key
            freq = value.frequency*1e-9
            if_freq = value.port.if_freq*1e3

            s += f"{name:<20} {freq:<20.6f} {if_freq:<20.1f}\n"


        if len(self.current_source) > 0:
            s += "\n\n*** Current source status ***\n"
            s += "{:20} {:20} {:20}\n".format(
                "Port name", "Current (mA)", "Output")
            s += "-" * (20 + 20 + 20) + "\n"
            for key, value in self.current_source.items():
                name = key
                cur = value.current()*1e3
                if_output = value.output()

                s += f"{name:<20} {cur:<20.6f} {if_output:<20}\n"

                
        if len(self.vatt) > 0:
            s += "\n\n*** LDA status ***\n"
            s += "{:20} {:20} {:30}\n".format(
                "Port name", "Attenuation (dB)", "Working frequency (GHz)")
            s += "-" * (20 + 20 + 30) + "\n"
            for key, value in self.vatt.items():
                name = key
                att = value.attenuation()
                fw = value.working_frequency() * 1e-9

                s += f"{name:<20} {att:<20.1f} {fw:<20.4f}\n"

        return s
    

    def close_all(self):
        for c in tuple(self.station.components.values()):
            if isinstance(c, Instrument):
                if isinstance(c, str):
                    c = Instrument.find_instrument(c)

                self.station._monitor_parameters = [v for v in self.station._monitor_parameters
                                            if v.root_instrument is not c]
                # remove instrument from station snapshot
                self.station.remove_component(c.name)
                # del will remove weakref and close the instrument
                c.close_all()
                del c
            elif hasattr(c, "parent"):
                # c = c.parent
                self.station._monitor_parameters = [v for v in self.station._monitor_parameters
                                            if v.root_instrument is not c]
                # remove instrument from station snapshot
                self.station.remove_component(c.name)
                # del will remove weakref and close the instrument
                c.parent.close_all()
                del c

        self.reset_manager()
        # time.sleep(20)


    def close_all_except_trigger(self):
        self.close_all()
        if self.trigger_address != "":
            hvi_trigger = HVI_Trigger(
                "hvi_trigger", self.trigger_address, debug=False)
            hvi_trigger.output(False)
            hvi_trigger.digitizer_delay(self.acquisition_delay)
            hvi_trigger.trigger_period(self.repetition_margin)
            self.hvi_trigger = hvi_trigger

            self.station.add_component(hvi_trigger)

    def reset_manager(self):

        self.lo = {}
        self.lo_address = {}
        self.lo_shared = {}
        self.lo_info = {"model": [], "address": [], "channel": [], "port": []}
        self.awg = {}
        self.awg_ref = {}
        self.awg_info = {"model": [], "address": [], "channel": [], "port": []}
        self.digitizer_ch = {}
        self.digitizer = {}
        self.dig_info = {"model": [], "address": [], "channel": [], "port": []}
        self.current_source = {}
        self.current_info = {"model": [],
                             "address": [], "channel": [], "port": []}

        self.port = {}
        self.acquire_port = {}

        self.IQ_corrector = {}
        self.lo_id = 0
        self.awg_id = 0
        self.dig_id = 0
        self.awg_id_dict = {}
        self.awg_ch = {}
        
        self.vatt = {}
        self.vatt_info = {}
        self.vatt_id = 0
        
        self.hvi_trigger = None
        

