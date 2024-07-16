from typing import Any, Dict, Optional

from qcodes.instrument import VisaInstrument
from qcodes import Function, Parameter
from qcodes.validators import Numbers


class APMSYN22(VisaInstrument):
    """
    QCodes driver for AnaPico APMSYN22 RF synthesizer. Based on N51x1 driver and AnaPico programming manual.
    For now only CW operation is supported and the output power is limited to a range that supports all frequencies.
    """

    def __init__(self, name: str, address: str, min_power: int = -30, max_power: int = 30, **kwargs: Any):
        super().__init__(name, address, terminator='\n', **kwargs)

        self._options = self.ask("*OPT?")

        self._complete = self.ask("*OPC?") # Hida added for a test

        self.power = Parameter(
            name="power",
            instrument=self,
            set_cmd='SOUR:POW {:.2f}',
            get_cmd='SOUR:POW?',
            unit='dBm',
            vals=Numbers(min_value=min_power,max_value=max_power)
        )

        self.frequency = Parameter(
            name="frequency",
            instrument=self,
            set_cmd='SOUR:FREQ {:.2f}',
            get_cmd='SOUR:FREQ?',
            get_parser=float,
            unit='Hz',
            vals=Numbers(min_value=100e3,max_value=22e9)
        )

        self.add_parameter('phase_offset',
                           label='Phase Offset',
                           get_cmd='SOUR:PHAS?',
                           get_parser=float,
                           set_cmd='SOUR:PHAS {:.2f}',
                           unit='rad'
                           )

        self.ref_in_freq = Parameter(
            name="ext_ref_in_frequency",
            instrument=self,
            set_cmd='SOUR:ROSC:EXT:FREQ {:.2f}',
            get_cmd='SOUR:ROSC:EXT:FREQ?',
            get_parser=float,
            unit='Hz',
            vals=Numbers(min_value=100e6,max_value=1e9)
        )

        self.ref_source = Parameter(
            name="ref_source",
            instrument=self,
            set_cmd='SOUR:ROSC:SOUR {}',
            get_cmd='SOUR:ROSC:SOUR?',
            val_mapping={
                "internal": "INT",
                "external": "EXT",
                "slave": "SLAV",
            },
        )

        self.output = Parameter(
            name="output",
            instrument=self,
            get_cmd="OUTP:STAT?",
            set_cmd="OUTP:STAT {}",
            val_mapping={False: "0", True: "1"},
        )

        self.output_blanking = Parameter(
            name="output_blanking",
            instrument=self,
            get_cmd="OUTP:BLAN:STAT?",
            set_cmd="OUTP:BLAN:STAT {}",
            val_mapping={False: "0", True: "1"},
        )

        self.connect_message()

    def get_idn(self) -> Dict[str, Optional[str]]:
        IDN_str = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, IDN_str.split(','))
        IDN: Dict[str, Optional[str]] = {
            'vendor': vendor, 'model': model,
            'serial': serial, 'firmware': firmware}
        return IDN
