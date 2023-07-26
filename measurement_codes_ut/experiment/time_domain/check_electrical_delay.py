
from logging import getLogger
import os
import numpy as np
import matplotlib.pyplot as plt
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sklearn.decomposition import PCA
from tqdm import tqdm
import time

from measurement_codes_ut.measurement_tool.wrapper import AttributeDict
from sequence_parser import Port, Sequence
from sequence_parser.instruction import *

from measurement_codes_ut.helper.plot_helper import PlotHelper
from plottr.data.datadict_storage import datadict_from_hdf5

logger = getLogger(__name__)


class CheckElectricalDelay(object):
    experiment_name = "CheckElectricalDelay"
    input_parameters = [
        'cavity_dressed_frequency_cw',
        "cavity_readout_sequence_amplitude_expected_sn",
        "cavity_readout_trigger_delay",
    ]
    output_parameters = [
        "cavity_readout_electrical_delay",
    ]

    def __init__(self, num_shot=1000, repetition_margin=50e3, sweep_range=1e6, sweep_step=41):
        self.dataset = None
        self.num_shot = num_shot
        self.sweep_range = sweep_range
        self.sweep_step = sweep_step
        self.repetition_margin = repetition_margin

    def execute(self, tdm, calibration_notes,
                update_experiment=True, update_analyze=True):
        if update_experiment:
            self.dataset = self.take_data(tdm, calibration_notes)

        if update_analyze:
            if self.dataset is None:
                raise ValueError("Data is not taken yet.")
            self.analyze(self.dataset, calibration_notes)

        return self.dataset

    def take_data(self, tdm, note):
        note = note.get_calibration_parameters(
            self.__class__.experiment_name, self.__class__.input_parameters)

        readout_freq = note.cavity_dressed_frequency_cw

        readout_port = tdm.port['readout'].port
        acq_port = tdm.acquire_port['readout_acquire']
        qubit_port = tdm.port['qubit'].port

        ports = [readout_port, qubit_port, acq_port]

        tdm.set_acquisition_delay(note.cavity_readout_trigger_delay)
        tdm.set_acquisition_mode(averaging_shot=True, averaging_waveform=True)
        tdm.set_repetition_margin(self.repetition_margin)
        tdm.set_shots(self.num_shot)

        seq = Sequence(ports)
        seq.add(ResetPhase(phase=0), readout_port, copy=False)
        seq.trigger(ports)
        seq.add(Square(amplitude=note.cavity_readout_sequence_amplitude_expected_sn, duration=2000),
                readout_port, copy=False)
        seq.add(Acquire(duration=2000), acq_port)
        # seq.draw()
        seq.trigger(ports)

        shift = np.linspace((-self.sweep_range/2),
                            (self.sweep_range/2), self.sweep_step)
        
        tdm.port['readout'].frequency = readout_freq - 100e6 + shift

        tdm.sequence = seq
        tdm.variables = None

        dataset = tdm.take_data(dataset_name=self.__class__.experiment_name, as_complex=True, exp_file=__file__)
        # print(f"Experiment data saved in {self.data_path_all}")
        return dataset

    # override
    def analyze(self, dataset, calibration_note, savefig=True, savepath="./fig"):

        freq = dataset.data["readout_LO_frequency"]["values"]
        signal = dataset.data["readout_acquire"]["values"]

        phase = np.unwrap(np.angle(signal))
        a, b = np.polyfit(freq, phase, 1)
        electrical_delay = a / (2*np.pi)

        self.data_label = dataset.path.split("/")[-1][27:]

        plot = PlotHelper(f"{self.data_label}", 1, 3)
        plt.plot(freq, np.abs(signal), ".-", label="Amplitude")
        plot.label("Frequency shift (Hz)", "response")

        plot.change_plot(0, 1)
        plot.plot_fitting(freq, phase, y_fit=a*freq+b, label="Phase")
        plot.label("Frequency shift (Hz)", "Phase (rad)")

        plot.change_plot(0, 2)
        plt.plot(np.real(signal), np.imag(signal))
        plot.plot_complex(signal, label='IQ')
        plot.plot_complex(signal * np.exp(-1.j * 2*np.pi *
                          freq * electrical_delay), label='IQ corrected')
        plot.label("I", "Q")
        plt.tight_layout()

        if savefig:
            os.makedirs(savepath, exist_ok=True)
            plt.savefig(f"{savepath}/{self.data_label}.png",
                        bbox_inches='tight')
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.cavity_readout_electrical_delay = electrical_delay
        calibration_note.add_experiment_note(
            self.__class__.experiment_name, experiment_note, self.__class__.output_parameters)

    # override

    def report_stat(self):
        pass

    # override
    def report_visualize(self, dataset, note):
        pass
