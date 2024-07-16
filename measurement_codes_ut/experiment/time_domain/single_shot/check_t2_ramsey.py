
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
# from measurement_codes.fitting.rabi_oscillation import RabiOscillation
from measurement_codes_ut.fitting import DampedOscillation_plus_ConstantModel

from scipy.optimize import curve_fit

logger = getLogger(__name__)


class CheckT2Ramsey(object):
    experiment_name = "CheckT2Ramsey"
    input_parameters = [
        "cavity_readout_trigger_delay",
        "cavity_dressed_frequency",
        "cavity_readout_frequency",
        "qubit_dressed_frequency",
        "qubit_full_linewidth",
        "qubit_control_amplitude",
        "rabi_frequency",
        "pi_pulse_length",
        "pi_pulse_power",
        "cavity_readout_amplitude",
        "cavity_readout_window_coefficient",
        "readout_pulse_length",
        "measurement_axis",
        "readout_assignment_border",
        "readout_g_direction"
    ]
    output_parameters = [
        "t2_star",
        # "qubit_ramsey_frequency",
        # "ramsey_frequency"
    ]

    def __init__(self, num_shot=1000, repetition_margin=200e3, min_duration=100, max_duration=20000, hand_detune=0.25e6, num_sample: int = 51):
        self.dataset = None
        self.num_shot = num_shot
        self.hand_detune = hand_detune
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.num_sample = num_sample
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
        tdm.set_acquisition_mode(averaging_waveform=True, averaging_shot=False)
        tdm.set_shots(self.num_shot)

        readout_freq = note.cavity_readout_frequency

        qubit_freq = note.qubit_dressed_frequency

        tdm.port['readout'].frequency = readout_freq

        if tdm.lo['qubit'] is None:
            qubit_port.if_freq = (qubit_freq+self.hand_detune)/1e9
        else:
            tdm.port['qubit'].frequency = qubit_freq + self.hand_detune

        tdm.port['readout'].window = note.cavity_readout_window_coefficient

        half_pi_pulse_power = note.pi_pulse_power
        half_pi_pulse_length = note.pi_pulse_length * 0.5

        interval = 2  # ns (Digitizer sampling rate. Default 2ns)
        time_range = interval * np.linspace(
            (self.min_duration + interval - 1) // interval,
            self.max_duration // interval,
            num=self.num_sample,
            dtype=int
        )

        duration = Variable("duration", time_range, "ns")
        variables = Variables([duration])
        seq = Sequence(ports)
        seq.add(Gaussian(amplitude=half_pi_pulse_power, fwhm=half_pi_pulse_length/3, duration=half_pi_pulse_length, zero_end=True),
                qubit_port, copy=False)
        seq.add(Delay(duration), qubit_port)
        seq.add(Gaussian(amplitude=-half_pi_pulse_power, fwhm=half_pi_pulse_length/3, duration=half_pi_pulse_length, zero_end=True),
                qubit_port, copy=False)
        seq.trigger(ports)
        seq.add(ResetPhase(phase=0), readout_port, copy=False)
        seq.add(Square(amplitude=note.cavity_readout_amplitude, duration=note.readout_pulse_length),
                readout_port, copy=False)
        seq.add(Acquire(duration=note.readout_pulse_length), acq_port)

        seq.trigger(ports)

        tdm.sequence = seq
        tdm.variables = variables

        dataset = tdm.take_data(dataset_name=self.__class__.experiment_name, as_complex=False, exp_file=__file__)
        return dataset

    def analyze(self, dataset, note, savefig=False, savepath="./fig"):

        time = dataset.data['duration']['values']
        response = dataset.data['readout_acquire']['values']

        def get_pred(data, axis, border, which):
            component = np.dot(axis, data)
            if (component-border)*which > 0:
                return 0
            else:
                return 1

        component = np.zeros(self.num_sample)
        for i in range(self.num_sample):
            pred = np.mean([get_pred(data, note.measurement_axis, note.readout_assignment_border, note.readout_g_direction) for data in response[i]])
            component[i] = pred
        time_us = time

        model = DampedOscillation_plus_ConstantModel()
        params = model.guess(component, time)
        rst = model.fit(component, params=params, x=time)
        param_list = [
            rst.params['amplitude'].value,
            rst.params['frequency'].value,
            rst.params['phase'].value,
            rst.params['decay'].value,
            rst.params['c'].value
        ]
        param_error_list = [
            rst.params['amplitude'].stderr,
            rst.params['frequency'].stderr,
            rst.params['phase'].stderr,
            rst.params['decay'].stderr,
            rst.params['c'].stderr
        ]

        fitting_parameter_list = [
            'amplitude',
            'ramsey_frequency',
            'phase_offset',
            't2_star'
            'amplitude_offset',
        ]
        for index, item in enumerate(fitting_parameter_list):
            name = item
            value = param_list[index]
            value_error = param_error_list[index]
            setattr(self, name, value)
            if value_error:
                setattr(self, name+"_stderr", value_error)
            else:
                setattr(self, name+"_stderr", None)
        decay_rate = (1./param_list[3])
        t2_star = param_list[3]

        fit_slice = 1001
        time_fit = np.linspace(self.min_duration, self.max_duration, fit_slice)
        component_fit = rst.best_fit

        self.data_label = dataset.path.split("/")[-1][27:]

        plotter = PlotHelper(f"{self.data_label}", 1, 2)
        plotter.plot_complex(
            data=np.mean(response[:,:, 0]+1j*response[:,:, 1], axis=1), label="IQ")
        plotter.label("I", "Q")
        plotter.change_plot(0, 1)
        plotter.plot_fitting(time, component, y_fit=component_fit)
        plotter.label("Time (ns)", "P1")
        plt.tight_layout()
        if savefig:
            plt.savefig(
                f"{savepath}/{self.data_label}.png")
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.t2_star = t2_star
        experiment_note.ramsey_decay_rate = decay_rate
        # experiment_note.hand_detune = self.hand_detune
        # experiment_note.ramsey_frequency = self.ramsey_frequency
        # experiment_note.qubit_ramsey_frequency = note.qubit_dressed_frequency + \
        #     self.hand_detune - np.abs(self.ramsey_frequency)
        note.add_experiment_note(
            self.__class__.experiment_name, experiment_note, self.__class__.output_parameters,)
