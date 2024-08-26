from typing import Any
import pyvisa as visa
import numpy as np

import qcodes.validators as vals
from qcodes import Instrument, VisaInstrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes.parameters import create_on_off_val_mapping
from qcodes.validators import Bool, Enum, Ints, Numbers
from qcodes.parameters import DelegateParameter
from functools import partial
from typing import Any, Literal, Optional, Union

ModeType = Literal["CURR", "VOLT"]


class ADCMT6541(VisaInstrument):
    """
    This is the QCoDeS driver for the ADCMT6541 voltage/current source.

    Status: beta-version. 

    Args:
      name: What this instrument is called locally.
      address: The VISA address of this instrument
      kwargs: kwargs to be passed to VisaInstrument class
      terminator: read terminator for reads/writes to the instrument.
    """

    def __init__(
        self, name: str, address: str, terminator: str = "\n", **kwargs: Any
    ) -> None:
        super().__init__(name, address, terminator=terminator, **kwargs)

        for idx in range(1,5):
            setattr(self, f"ch{idx}", ADCMT6541Channel(self, f"ch{idx}", idx))

        self.connect_message()
    
class ADCMT6541Channel(InstrumentChannel):
    """
    Class to control the ADCMT6541 channels ch1 to ch4.
    This class only supports the minimum fincutionality for outputting voltage/current. 
    To harness the full potential of the instrument, including features like measurement, pulse generation, and sweeps, additional coding is essential.
    """

    def __init__(self, parent: Instrument, name: str, channel: int) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is attached.
            name: The 'colloquial' name of the channel
            channel: The index of the channel
        """

        super().__init__(parent, name)
        self.channel = channel

        self.add_parameter(
            "output",
            label="Output State",
            get_cmd=self.state,
            set_cmd=lambda x: self.on() if x else self.off(),
            val_mapping={
                "off": 0,
                "on": 1,
                "suspend": -1,
            },
        )

        self.add_parameter(
            "source_mode",
            label="Source Mode",
            get_cmd=self._get_source_mode,
            set_cmd=self._set_source_mode,
            vals=Enum("VOLT", "CURR"),
        )

        # We need to get the source_mode value here as we cannot rely on the
        # default value that may have been changed before we connect to the
        # instrument (in a previous session or via the frontpanel).
        self.source_mode()

        self.add_parameter(
            "voltage_range",
            label="Voltage Source Range",
            unit="V",
            get_cmd=partial(self._get_range, "VOLT"),
            set_cmd=partial(self._set_range, "VOLT"),
            vals=Enum(3e0, 1e1),
            snapshot_exclude=self.source_mode() == "CURR",
        )

        self.add_parameter(
            "current_range",
            label="Current Source Range",
            unit="I",
            get_cmd=partial(self._get_range, "CURR"),
            set_cmd=partial(self._set_range, "CURR"),
            vals=Enum(3e-6, 30e-6, 300e-6, 3e-3, 30e-3, 300e-3, 500e-3),
            snapshot_exclude=self.source_mode() == "VOLT",
        )

        self.add_parameter("range", parameter_class=DelegateParameter, source=None)

        # The instrument does not support auto range. The parameter
        # auto_range is introduced to add this capability with
        # setting the initial state at False mode.
        self.add_parameter(
            "auto_range",
            label="Auto Range",
            set_cmd=self._set_auto_range,
            get_cmd=None,
            initial_cache_value=False,
            vals=Bool(),
        )

        self.add_parameter(
            "voltage",
            label="Voltage",
            unit="V",
            set_cmd=partial(self._get_set_output, "VOLT"),
            get_cmd=partial(self._get_set_output, "VOLT"),
            snapshot_exclude=self.source_mode() == "CURR",
        )

        self.add_parameter(
            "current",
            label="Current",
            unit="I",
            set_cmd=partial(self._get_set_output, "CURR"),
            get_cmd=partial(self._get_set_output, "CURR"),
            snapshot_exclude=self.source_mode() == "VOLT",
        )

        self.add_parameter(
            "output_level", parameter_class=DelegateParameter, source=None
        )

        if self.source_mode() == "VOLT":
            self.range.source = self.voltage_range
            self.output_level.source = self.voltage
        else:
            self.range.source = self.current_range
            self.output_level.source = self.current

        self.add_parameter(
            "voltage_limit",
            label="Voltage Protection Limit",
            unit="V",
            # vals=Ints(1, 30),
            get_cmd=partial(self._get_limit, "VOLT"),
            set_cmd=partial(self._set_limit, "VOLT"),
            set_parser=float,
        )

        self.add_parameter(
            "current_limit",
            label="Current Protection Limit",
            unit="I",
            # vals=Numbers(1e-3, 200e-3),
            get_cmd=partial(self._get_limit, "CURR"),
            set_cmd=partial(self._set_limit, "CURR"),
            set_parser=float,
        )

        # self.connect_message()

    def on(self) -> None:
        """Turn output on"""
        self.write(f"SCH{self.channel} OPR")
        # self.measure._output = True

    def off(self) -> None:
        """Turn output off"""
        self.write(f"SCH{self.channel} SBY")
        # self.measure._output = False

    def state(self) -> int:
        """Check state"""
        state = self.ask(f"SCH{self.channel} SBY?")[:3]
        state_mapping = {"OPR":1, "SBY":0, "SUS":-1}
        return state_mapping[state]

    def ramp_voltage(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the voltage from the current level to the specified output.

        Args:
            ramp_to: The ramp target in Volt
            step: The ramp steps in Volt
            delay: The time between finishing one step and
                starting another in seconds.
        """
        self._assert_mode("VOLT")
        self._ramp_source(ramp_to, step, delay)

    def ramp_current(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the current from the current level to the specified output.

        Args:
            ramp_to: The ramp target in Ampere
            step: The ramp steps in Ampere
            delay: The time between finishing one step and starting
                another in seconds.
        """
        self._assert_mode("CURR")
        self._ramp_source(ramp_to, step, delay)

    def _ramp_source(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the output from the current level to the specified output

        Args:
            ramp_to: The ramp target in volt/ampere
            step: The ramp steps in volt/ampere
            delay: The time between finishing one step and
                starting another in seconds.
        """
        saved_step = self.output_level.step
        saved_inter_delay = self.output_level.inter_delay

        self.output_level.step = step
        self.output_level.inter_delay = delay
        self.output_level(ramp_to)

        self.output_level.step = saved_step
        self.output_level.inter_delay = saved_inter_delay

    def _get_set_output(
        self, mode: ModeType, output_level: Optional[float] = None
    ) -> Optional[float]:
        """
        Get or set the output level.

        Args:
            mode: "CURR" or "VOLT"
            output_level: If missing, we assume that we are getting the
                current level. Else we are setting it
        """
        self._assert_mode(mode)
        if output_level is not None:
            self._set_output(output_level)
            return None
        elif mode == 'CURR':
            return float(self.ask(f"SCH{self.channel} SOI?")[3:])
        elif mode == 'VOLT':
            return float(self.ask(f"SCH{self.channel} SOV?")[3:])

    def _set_output(self, output_level: float) -> None:
        """
        Set the output of the instrument.

        Args:
            output_level: output level in Volt or Ampere, depending
                on the current mode.
        """
        auto_enabled = self.auto_range()

        if not auto_enabled:
            self_range = self.range()
            if self_range is None:
                raise RuntimeError(
                    "Trying to set output but not in auto mode and range is unknown."
                )
        else:
            mode = self.source_mode.get_latest()
            if mode == "CURR":
                self_range = 500e-3
            else:
                self_range = 10.0

        # print(self_range)

        # Check we are not trying to set an out of range value
        if self.range() is None or abs(output_level) > abs(self_range):
            # Check that the range hasn't changed
            if not auto_enabled:
                self_range = self.range.get_latest()
                if self_range is None:
                    raise RuntimeError(
                        "Trying to set output but not in"
                        " auto mode and range is unknown."
                    )
            # If we are still out of range, raise a value error
            # print(self_range)
            if abs(output_level) > abs(self_range):
                raise ValueError(
                    "Desired output level not in range"
                    f" [-{self_range:.3}, {self_range:.3}]"
                )

        if auto_enabled:
            mode = self.source_mode.get_latest()
            if mode == "CURR":
                cmd_str = f"SCH{self.channel} SIRX SOI{output_level:.5e}"
            if mode == "VOLT":
                cmd_str = f"SCH{self.channel} SVRX SOV{output_level:.5e}"
        else:
            mode = self.source_mode.get_latest()
            if mode == "CURR":
                cmd_str = f"SCH{self.channel} SOI{output_level:.5e}"
            if mode == "VOLT":
                cmd_str = f"SCH{self.channel} SOV{output_level:.5e}"
            
        self.write(cmd_str)


    def _set_auto_range(self, val: bool) -> None:
        """
        Enable/disable auto range.

        Args:
            val: auto range on or off
        """
        self._auto_range = val
        # # Disable measurement if auto range is on
        # if self.measure.present:
        #     # Disable the measurement module if auto range is enabled,
        #     # because the measurement does not work in the
        #     # 10mV/100mV ranges.
        #     self.measure._enabled &= not val

    def _assert_mode(self, mode: ModeType) -> None:
        """
        Assert that we are in the correct mode to perform an operation.

        Args:
            mode: "CURR" or "VOLT"
        """
        if self.source_mode() != mode:
            raise ValueError(
                "Cannot get/set {} settings while in {} mode".format(
                    mode, self.source_mode()
                )
            )

    def _set_source_mode(self, mode) -> None:
        """
        Set output mode and change delegate parameters' source accordingly.
        Also, exclude/include the parameters from snapshot depending on the
        mode. The instrument does not support 'current', 'current_range'
        parameters in "VOLT" mode and 'voltage', 'voltage_range' parameters
        in "CURR" mode.

        Args:
            mode: "CURR" or "VOLT"

        """
        if self.output() == "on":
            raise Exception("Cannot switch mode while source is on")

        if mode == "VOLT":
            mode_str = "VF"
            self.range.source = self.voltage_range
            self.output_level.source = self.voltage
            self.voltage_range.snapshot_exclude = False
            self.voltage.snapshot_exclude = False
            self.current_range.snapshot_exclude = True
            self.current.snapshot_exclude = True
        else:
            mode_str = "IF"
            self.range.source = self.current_range
            self.output_level.source = self.current
            self.voltage_range.snapshot_exclude = True
            self.voltage.snapshot_exclude = True
            self.current_range.snapshot_exclude = False
            self.current.snapshot_exclude = False

        self.write(f"SCH{self.channel} {mode_str}")
        self.source_mode.cache.set(mode)

    def _get_source_mode(self):
        """
        Query the source mode.

        """
        source_str = self.ask(f"SCH{self.channel} I?")
        mapping = {"V":"VOLT", 'I':'CURR'}
        return mapping[source_str[0]]

    def _set_range(self, mode: ModeType, output_range: float) -> None:
        """
        Update range

        Args:
            mode: "CURR" or "VOLT"
            output_range: Range to set. For voltage, we have the ranges [3e0, 10e1]. 
            For current, we have the ranges [3e-6, 30e-6, 300e-6, 3e-3, 30e-3, 300e-3, 500e-3]. 
            If auto_range = False, then setting the output can only happen if the set value is smaller than the present range.
        """
        self._assert_mode(mode)
        output_range = float(output_range)
        # self._update_measurement_module(source_mode=mode, source_range=output_range)
        cur_range = [3e-6, 30e-6, 300e-6, 3e-3, 30e-3, 300e-3, 500e-3]
        vol_range = [3e0, 1e1]
        val_convert = {"VOLT":{vol_range[i]:f"SVR{4+i}" for i in range(len(vol_range))}, "CURR":{cur_range[i]:f"SIR{-2+i}" for i in range(len(cur_range))}}
        if output_range in val_convert[mode]:
            self.write(f"SCH{self.channel} {val_convert[mode][output_range]}")
        else:
            if mode == 'CURR':
                larger_numbers = [(num, idx) for idx, num in enumerate(cur_range) if num > output_range]
                nearest_idx = min(larger_numbers, key=lambda x: x[0])[1]
                self.write(f"SCH{self.channel} SIR{-2+nearest_idx}")
            if mode == 'VOLT':
                larger_numbers = [(num, idx) for idx, num in enumerate(vol_range) if num > output_range]
                nearest_idx = min(larger_numbers, key=lambda x: x[0])[1]
                self.write(f"SCH{self.channel} SVR{4+nearest_idx}")
                

    def _get_range(self, mode: ModeType) -> float:
        """
        Query the present range.

        Args:
            mode: "CURR" or "VOLT"
        """
        self._assert_mode(mode)
        
        cur_range = [3e-6, 30e-6, 300e-6, 3e-3, 30e-3, 300e-3, 500e-3]
        vol_range = [3e0, 1e1]
        val_convert = {"VOLT":{f"SVR {4+i}":vol_range[i] for i in range(len(vol_range))}, "CURR":{f"SIR {-2+i}":cur_range[i] for i in range(len(cur_range))}}
        if mode == 'VOLT':
            range_value = self.ask(f"SCH{self.channel} SVR?")
        elif mode == 'CURR':
            range_value = self.ask(f"SCH{self.channel} SIR?")
        return float(val_convert[mode][range_value[:-1]])
        
    def _set_limit(self, mode: ModeType, limit: float) -> None:
        """
        Update limit value

        Args:
            mode: "CURR" or "VOLT"
            limit: limit value for voltage/current to set. Voltage limit must be below 10 V and current limit must be below 500 mA. 
        """
        # self._assert_mode(mode)
        limit = float(limit)
        # self._update_measurement_module(source_mode=mode, source_range=output_range)
        if mode == 'VOLT':
            if abs(limit) > 10:
                raise ValueError(
                "Cannot set voltage limit > 10 V."
            )
            self.write(f"SCH{self.channel} LMV{limit}")
        if mode == 'CURR':
            if abs(limit) > 0.5:
                raise ValueError(
                "Cannot set current limit > 0.5 A."
            )
            self.write(f"SCH{self.channel} LMI{limit}")

    def _get_limit(self, mode: ModeType) -> float:
        """
        Query the present limit value.

        Args:
            mode: "CURR" or "VOLT"

        Returns:
            High limit, Low limit
        """
        # self._assert_mode(mode)
        if mode == 'VOLT':
            limit_value = self.ask(f"SCH{self.channel} LMV?")[3:-1].split(',')
            return float(limit_value[0]), float(limit_value[1])
        if mode == 'CURR':
            limit_value = self.ask(f"SCH{self.channel} LMI?")[3:-1].split(',')
            return float(limit_value[0]), float(limit_value[1])


