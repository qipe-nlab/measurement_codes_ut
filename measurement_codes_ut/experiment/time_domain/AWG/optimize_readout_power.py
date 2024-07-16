
from logging import getLogger
import os
import numpy as np
import matplotlib.pyplot as plt
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sklearn.decomposition import PCA
from tqdm import tqdm

from measurement_codes_ut.measurement_tool.wrapper import AttributeDict
from sequence_parser import Port, Sequence, Variable, Variables
from sequence_parser.instruction import *

from measurement_codes_ut.helper.plot_helper import PlotHelper
from plottr.data.datadict_storage import datadict_from_hdf5
from measurement_codes_ut.fitting.qubit_spectral import QubitSpectral
from measurement_codes_ut.fitting.rabi_oscillation import RabiOscillation

from scipy.optimize import curve_fit

from measurement_codes_ut.fitting.gaussian_fitter import GaussianFitter


logger = getLogger(__name__)


class OptimizeReadoutPower(object):
    experiment_name = "OptimizeReadoutPower"
    input_parameters = [
        "cavity_readout_sequence_amplitude_expected_sn",
        "cavity_readout_trigger_delay",
        # "cavity_dressed_frequency",
        "cavity_readout_frequency",
        "qubit_dressed_frequency",
        "qubit_full_linewidth",
        "qubit_control_amplitude",
        "rabi_frequency",
        "pi_pulse_length",
        "pi_pulse_power",
        "readout_pulse_length"
    ]
    output_parameters = [
        "cavity_readout_amplitude",
    ]

    def __init__(self, num_shot=1000, repetition_margin=200e3, num_point=41, min_amplitude=0.0, max_amplitude=1.0):
        self.dataset = None
        self.num_shot = num_shot
        self.num_point = num_point
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
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

    def take_data(self, tdm, calibaration_note):
        note = calibaration_note.get_calibration_parameters(
            self.__class__.experiment_name, self.__class__.input_parameters)

        readout_port = tdm.port['readout'].port
        acq_port = tdm.acquire_port['readout_acquire']
        qubit_port = tdm.port['qubit'].port

        ports = [readout_port, qubit_port, acq_port]

        tdm.set_acquisition_delay(note.cavity_readout_trigger_delay)
        tdm.set_repetition_margin(self.repetition_margin)
        tdm.set_shots(self.num_shot)
        tdm.set_acquisition_mode(averaging_waveform=True, averaging_shot=True)

        readout_freq = note.cavity_readout_frequency

        qubit_freq = note.qubit_dressed_frequency

        tdm.port['readout'].frequency = readout_freq
        qubit_port.if_freq = qubit_freq/1e9

        readout_amp_range = np.linspace(
            self.min_amplitude, self.max_amplitude, self.num_point, endpoint=True)
        qubit_amp_range = [0, note.pi_pulse_power]
        
        qubit_amplitude = Variable("qubit_amplitude", qubit_amp_range, "V")
        readout_amplitude = Variable("readout_amplitude", readout_amp_range, "V")
        variables = Variables([qubit_amplitude, readout_amplitude])

        seq = Sequence(ports)
        seq.add(Gaussian(amplitude=qubit_amplitude, fwhm=note.pi_pulse_length/3, duration=note.pi_pulse_length, zero_end=True),
                qubit_port, copy=False)
        seq.add(Delay(10), qubit_port)
        seq.trigger(ports)
        seq.add(ResetPhase(phase=0), readout_port, copy=False)
        seq.add(Square(amplitude=readout_amplitude, duration=note.readout_pulse_length),
                readout_port, copy=False)
        seq.add(Acquire(duration=note.readout_pulse_length), acq_port)

        seq.trigger(ports)

        tdm.sequence = seq
        tdm.variables = variables

        dataset = tdm.take_data(dataset_name=self.__class__.experiment_name, as_complex=True, exp_file=__file__)

        return dataset

    def analyze(self, dataset, note, savefig=True, savepath="./fig"):

        power_list = dataset.data['readout_amplitude']['values'][:self.num_point]
        response = dataset.data['readout_acquire']['values'].reshape(
            2, self.num_point)

        distance_ge = abs(response[0] - response[1])

        self.data_label = dataset.path.split("/")[-1][27:]

        plt.figure()

        plt.plot(power_list, distance_ge, marker=".")
        plt.xlabel('DAC amplitude (V)')
        plt.ylabel('Distance g-e')
        
        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}.png")
        plt.show()

        readout_power_opt = power_list[np.argmax(distance_ge)]

        experiment_note = AttributeDict()
        # experiment_note.cavity_readout_window_coefficient = window_opt
        experiment_note.cavity_readout_amplitude = readout_power_opt
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters)

    def report_stat(self):
        pass

    def report_visualize(self, dataset, note):
        pass
