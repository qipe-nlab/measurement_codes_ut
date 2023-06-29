import numpy as np
from logging import getLogger

import numpy as np
import qcodes as qc
from qcodes_drivers.E82x7 import E82x7
from qcodes_drivers.N51x1 import N51x1
from qcodes_drivers.HVI_Trigger import HVI_Trigger
from qcodes_drivers.iq_corrector import IQCorrector
from qcodes_drivers.M3102A import M3102A
from qcodes_drivers.M3202A import M3202A
from sequence_parser import Port, Sequence
from sequence_parser.iq_port import IQPort
from qcodes.instrument_drivers.yokogawa.GS200 import GS200
from .instrument_manager import InstrumentManagerBase
from plottr.data.datadict_storage import DataDict, DDH5Writer
from .port_manager import PortManager

import matplotlib.pyplot as plt

from measurement_codes_ut.measurement_tool import Session

logger = getLogger(__name__)


class TimeDomainInstrumentManager(InstrumentManagerBase):
    """Insturment management class for timedomain measurement"""

    def __init__(self, session: Session, trigger_address: str, save_path) -> None:
        """Constructor of time domain measurement

        Args:
            session (Session): session of measurement
            config_name (str): default config name of instruments
        """
        # print("Creating a new insturment management class for timedomain measurement...", end="")
        super().__init__(session, trigger_address, save_path)

    def take_data(self, sequences, dataset_name: str, lo_sweep=None):
        pass

    def set_wiring_note(self, wiring_info):
        self.wiring_info = wiring_info

    def load_sequence(self, sequence: Sequence, cycles: int, noise_variance=0):
        rng = np.random.default_rng()
        sequence.compile()
        self.seq_len = len(list(self.port.values())[0].port.waveform)
        for awg in self.awg.values():
            awg.stop_all()
            awg.flush_waveform()

        waveform_awg = {
            key: np.zeros(self.seq_len, dtype=complex) for key in self.awg_ch.keys()}
        waveform_idx = 0
        for key, port_manager in self.port.items():
            # awg_info = self.awg_info[key]
            awg_index, awg_ch = self.awg_ch[key]
            awg = self.awg[awg_index]
            port = port_manager.port
            port_manager.update_frequency()
            if "readout" in port.name:
                del waveform_awg[awg_index, awg_ch]
                dig_ch = self.digitizer_ch[key]
                waveform = port.waveform
                # plt.plot(waveform.real)
                if noise_variance > 0:
                    waveform += rng.normal(scale=np.sqrt(noise_variance),
                                           size=len(waveform))
                if self.IQ_corrector[port.name] is not None:
                    waveform_corrected = self.IQ_corrector[port.name].correct(
                        waveform)
                    for _ in range(2):  # i or q
                        wave = waveform_corrected[_]
                        awg.load_waveform(wave, waveform_idx,
                                          append_zeros=True)
                        awg_ch[_].queue_waveform(
                            waveform_idx, trigger="software/hvi", cycles=cycles)
                        waveform_idx += 1
                else:
                    awg.load_waveform(
                        waveform.real, waveform_idx, append_zeros=True)
                    awg_ch.queue_waveform(
                        waveform_idx, trigger="software/hvi", cycles=cycles)
                    waveform_idx += 1

                dig_ch.cycles(cycles)
                acq_port = self.acquire_port[key+"_acquire"]
                if len(acq_port.measurement_windows) == 0:
                    acquire_start = 0
                else:
                    acquire_start = int(acq_port.measurement_windows[0][0])
                    acquire_end = int(acq_port.measurement_windows[-1][1])
                    assert acquire_start % dig_ch.sampling_interval() == 0
                    assert acquire_end % dig_ch.sampling_interval() == 0
                dig_ch.points_per_cycle(1000)
                dig_ch.delay(acquire_start //
                             dig_ch.sampling_interval())

            else:
                waveform = port.waveform
                waveform_awg[awg_index, awg_ch] += waveform

        for key, waveform in waveform_awg.items():
            awg_index, awg_ch = self.awg_ch[key]
            awg = self.awg[awg_index]
            if self.IQ_corrector[key] is not None:
                try:
                    waveform_corrected = self.IQ_corrector[key].correct(
                        waveform)
                except NameError:
                    i = waveform.real
                    q = waveform.imag
                    waveform_corrected = i, q

                for _ in range(2):  # i or q
                    wave = waveform_corrected[_]
                    awg.load_waveform(wave, waveform_idx, append_zeros=True)
                    awg_ch[_].queue_waveform(
                        waveform_idx, trigger="software/hvi", cycles=cycles)
                    waveform_idx += 1
            else:
                awg.load_waveform(
                    waveform.real, waveform_idx, append_zeros=True)
                awg_ch.queue_waveform(
                    waveform_idx, trigger="software/hvi", cycles=cycles)
                waveform_idx += 1

    def run(self, sequence: Sequence, demodulate=True, averaging_shot=None, averaging_waveform=None, as_complex=True):

        self.load_sequence(sequence, self.num_shot)

        if averaging_shot is None:
            averaging_shot = self.averaging_shot
        if averaging_waveform is None:
            averaging_waveform = self.averaging_waveform

        self.hvi_trigger.digitizer_delay(self.acquisition_delay)  # ns
        self.hvi_trigger.trigger_period(
            int((self.repetition_margin + self.seq_len)/10+1)*10)  # ns

        self.set_acquisition_mode(averaging_shot, averaging_waveform)
        try:
            for name, lo in self.lo.items():
                if "Rohde" in self.lo_info[name]['model']:
                    lo.on()
                else:
                    lo.output(True)
            for awg_index, awg_ch in self.awg_ch.values():
                awg = self.awg[awg_index]
                if isinstance(awg_ch, tuple):
                    for _ in range(2):
                        awg_ch[_].start()
                else:
                    awg_ch.start()
            for dig in self.digitizer_ch.values():
                dig.start()
            self.hvi_trigger.output(True)
            data = {}
            for key, dig in self.digitizer_ch.items():
                d = dig.read()
                data[key] = d * dig.voltage_step()

            for awg in self.awg.values():
                awg.stop_all()
            for dig in self.digitizer_ch.values():
                dig.stop()
            self.hvi_trigger.output(False)

            if averaging_shot:
                data_return = {}
                for key, d in data.items():
                    data_return[key] = d.mean(axis=0)
                # assuming only 1 readout, should be extended to more than 1 readout
                # self.data = data_return
                if demodulate:
                    data_return = self.demodulate(
                        data_return, averaging_waveform, as_complex)

                return data_return
            else:
                data_return = {}
                for key, d in data.items():
                    data_return[key] = d
                # assuming only 1 readout, should be extended to more than 1 readout
                if demodulate:
                    data_return = self.demodulate(
                        data_return, averaging_waveform, as_complex)
                return data_return

        finally:
            self.hvi_trigger.output(False)
            for awg in self.awg.values():
                awg.stop_all()
            for dig in self.digitizer_ch.values():
                dig.stop()
            for name, lo in self.lo.items():
                if "Rohde" in self.lo_info[name]['model']:
                    lo.on()
                else:
                    lo.output(False)

    def stop(self):
        self.hvi_trigger.output(False)
        for awg in self.awg.values():
            awg.stop_all()
        for dig in self.digitizer_ch.values():
            dig.stop()
        for lo in self.lo.values():
            lo.output(False)

    def demodulate(self, data_all, averaging_waveform=True, as_complex=True):
        data_demod = {}
        for key, data in data_all.items():
            t = np.arange(data.shape[-1]) * \
                self.digitizer_ch[key].sampling_interval() * 1e-9
            if averaging_waveform:
                if self.port[key].window is None:
                    data_demod[key] = (
                        data * np.exp(2j * np.pi * self.port[key].port.if_freq*1e9 * t)).mean(axis=-1)
                else:
                    d = data * np.exp(2j * np.pi *
                                      self.port[key].port.if_freq*1e9 * t)
                    data_demod[key] = np.dot(d, self.port[key].window)
                if as_complex == False:
                    data_demod[key] = np.stack(
                        (data_demod[key].real, data_demod[key].imag), axis=-1)
            else:
                data_demod[key] = (
                    data * np.exp(2j * np.pi * self.port[key].port.if_freq*1e9 * t))
                if as_complex == False:
                    data_demod[key] = np.stack(
                        (data_demod[key].real, data_demod[key].imag), axis=-1)
        return data_demod

    def set_acquisition_mode(self, averaging_shot, averaging_waveform):
        self.averaging_shot = averaging_shot
        self.averaging_waveform = averaging_waveform

    def set_average_window_coefficients(self, port_name, window):
        self.port[port_name].window = window

    def set_repetition_margin(self, time):
        self.repetition_margin = int(time)
        # self.hvi_trigger.trigger_period(int(time))

    def set_acquisition_delay(self, delay):
        self.acquisition_delay = int(delay//10+1) * 10
        # self.hvi_trigger.digitizer_delay(int(delay//10+1) * 10)

    def set_shots(self, num_shot):
        self.num_shot = num_shot

    def prepare_experiment(self, writer, exp_file):
        writer.add_tag(self.tags)
        writer.backup_file([exp_file, __file__])
        writer.save_text("wiring.md", self.wiring_info)
        writer.save_dict("station_snapshot.json", self.station.snapshot())
