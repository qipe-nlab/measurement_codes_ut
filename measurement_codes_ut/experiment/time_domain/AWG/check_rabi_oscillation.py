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
from measurement_codes_ut.fitting import ResonatorReflectionModel
from measurement_codes_ut.fitting.qubit_spectral import QubitSpectral
from measurement_codes_ut.fitting.rabi_oscillation import RabiOscillation
from scipy.optimize import curve_fit


logger = getLogger(__name__)


class CheckRabiOscillation(object):
    experiment_name = "CheckRabiOscillation"
    input_parameters = [
        "cavity_readout_sequence_amplitude_expected_sn",
        "cavity_readout_trigger_delay",
        "cavity_dressed_frequency",
        "qubit_dressed_frequency",
        "qubit_full_linewidth",
        "qubit_control_amplitude",
        "readout_pulse_length"
    ]
    output_parameters = [
        "rabi_frequency",
        "rabi_pulse_amplitude",
    ]

    def __init__(self, num_shot=1000, repetition_margin=200e3, pulse_amplitude = 1, min_duration=10, max_duration=400):
        self.dataset = None
        self.num_shot = num_shot
        # self.num_cycle = num_cycle
        # self.num_point_per_cycle = num_point_per_cycle
        # self.rabi_minimum_duration = 100
        self.repetition_margin = repetition_margin
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.pulse_amplitude = pulse_amplitude

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

        readout_freq = note.cavity_dressed_frequency

        qubit_freq = note.qubit_dressed_frequency

        tdm.port['readout'].frequency = readout_freq
        qubit_port.if_freq = qubit_freq/1e9

        interval = 2  # ns (Digitizer sampling rate. Default 2ns)
        dur_range = interval * np.linspace((self.min_duration + interval - 1) // interval, self.max_duration // interval, num=40, dtype=int)

        duration = Variable("duration", dur_range, "ns")
        variables = Variables([duration])

        seq = Sequence(ports)

        seq.add(FlatTop(Gaussian(amplitude=self.pulse_amplitude, fwhm=10, duration=20), top_duration=duration),
                qubit_port, copy=False)
        seq.add(Delay(20), qubit_port)
        seq.trigger(ports)
        seq.add(ResetPhase(phase=0), readout_port, copy=False)
        seq.add(Square(amplitude=note.cavity_readout_sequence_amplitude_expected_sn, duration=note.readout_pulse_length),
                readout_port, copy=False)
        seq.add(Acquire(duration=note.readout_pulse_length), acq_port)

        seq.trigger(ports)

        tdm.sequence = seq
        tdm.variables = variables

        dataset = tdm.take_data(dataset_name=self.__class__.experiment_name, as_complex=False, exp_file=__file__)
        return dataset

    # override
    def analyze(self, dataset, note, savefig=True, savepath="./fig"):

        time = dataset.data["duration"]["values"]
        response = dataset.data["readout_acquire"]["values"]
        response_cplx = response[:, 0] + 1j*response[:, 1]

        max_time_length = max(time)

        pca = PCA()
        projected = pca.fit_transform(response)
        component = projected[:, 0]

        N = len(time)
        C_init = np.mean(component)
        y_n = component - C_init
        fft_data = np.fft.fft(y_n)
        freq_idx = np.argmax(np.abs(fft_data[:int(N/2)]))
        fft_peak = fft_data[freq_idx]
        T = time[-1] - time[0]
        del_freq = 1/T

        B_init = np.angle(fft_peak)
        f_init = del_freq*freq_idx
        A_init = np.max(abs(y_n))

        p_init = [A_init, f_init, B_init, C_init]

        def Cosin(t, A, freq , phi, C):
            return A * np.cos(t*2*np.pi * freq + phi) + C

        popt, pcov = curve_fit(Cosin, time, component, p0=p_init, maxfev=100000)
        fit_slice = 10001
        time_fit = np.linspace(min(time), max(time), fit_slice)
        component_fit = Cosin(time_fit, *popt)
        self.rabi_frequency = popt[1]

        self.data_label = dataset.path.split("/")[-1][27:]

        plotter = PlotHelper(f"{self.data_label}", 1, 3)
        plotter.plot_complex(
            response_cplx, line_for_data=True)
        plotter.label('I', 'Q')

        plotter.change_plot(0, 1)
        plotter.plot(time, projected[:, 0], label="PCA")
        plotter.plot(time, projected[:, 1], label="perp-PCA")
        plotter.label('Duration (ns)', "Response")

        plotter.change_plot(0, 2)
        plotter.plot_fitting(time, component, y_fit=component_fit, label="PCA")
        plotter.label("Duration (ns)", "Response")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}.png",
                        bbox_inches='tight')
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.rabi_frequency = self.rabi_frequency
        experiment_note.rabi_pulse_amplitude = self.pulse_amplitude
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters)
