
from logging import getLogger
import os
import numpy as np
import matplotlib.pyplot as plt
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sklearn.decomposition import PCA
from tqdm import tqdm

from measurement_codes_ut.measurement_tool.wrapper import AttributeDict
from sequence_parser import Port, Sequence, Circuit
from sequence_parser.instruction import *

from measurement_codes_ut.helper.plot_helper import PlotHelper
from plottr.data.datadict_storage import datadict_from_hdf5
from measurement_code_ut.fitting import ResonatorReflectionModel
from measurement_codes_ut.fitting.qubit_spectral import QubitSpectral
# from measurement_codes.fitting.rabi_oscillation import RabiOscillation
from measurement_code_ut.fitting import DampedOscillation_plus_ConstantModel

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
        # "half_pi_pulse_power",
        # "half_pi_pulse_length",
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
        tdm.set_shots(self.num_shot)

        readout_freq = note.cavity_readout_frequency

        qubit_freq = note.qubit_dressed_frequency

        tdm.port['readout'].frequency = readout_freq

        tdm.port['qubit'].frequency = qubit_freq + self.hand_detune

        tdm.port['readout'].window = note.cavity_readout_window_coefficient

        half_pi_pulse_power = note.pi_pulse_power
        half_pi_pulse_length = note.pi_pulse_length * 0.5

        time_step = self.num_sample
        time_range = np.linspace(
            self.min_duration, self.max_duration, time_step, dtype=int)

        seq_list = []
        for dur in time_range:
            seq = Sequence(ports)
            if dur % 2 != 0:
                seq.add(Delay(1), qubit_port)
            seq.add(Gaussian(amplitude=half_pi_pulse_power, fwhm=half_pi_pulse_length/3, duration=half_pi_pulse_length, zero_end=True),
                    qubit_port, copy=False)
            seq.add(Delay(dur), qubit_port)
            seq.add(Gaussian(amplitude=-half_pi_pulse_power, fwhm=half_pi_pulse_length/3, duration=half_pi_pulse_length, zero_end=True),
                    qubit_port, copy=False)
            seq.trigger(ports)
            seq.add(ResetPhase(phase=0), readout_port, copy=False)
            seq.add(Square(amplitude=note.cavity_readout_amplitude, duration=2000),
                    readout_port, copy=False)
            seq.add(Acquire(duration=2000), acq_port)

            seq.trigger(ports)
            seq_list.append(seq)

        data = DataDict(
            time=dict(unit="ns"),
            s11=dict(axes=["time"]),
        )
        data.validate()

        with DDH5Writer(data, tdm.save_path, name=self.__class__.experiment_name) as writer:
            tdm.prepare_experiment(writer, __file__)
            for i, seq in enumerate(tqdm(seq_list)):
                raw_data = tdm.run(seq, averaging_waveform=True,
                                   averaging_shot=True, as_complex=False)
                writer.add_data(
                    time=time_range[i],
                    s11=raw_data['readout'],
                )

        files = os.listdir(tdm.save_path)
        date = files[-1] + '/'
        files = os.listdir(tdm.save_path+date)
        self.data_path = files[-1]

        self.data_path_all = tdm.save_path+date+self.data_path + '/'

        dataset = datadict_from_hdf5(self.data_path_all+"data")
        print(f"Experiment data saved in {self.data_path_all}")
        return dataset

    def analyze(self, dataset, note, savefig=False, savepath="./fig"):

        time = dataset['time']['values']
        response = dataset['s11']['values']

        pca = PCA()
        projected = pca.fit_transform(response)
        component = projected[:, 0]
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

        plotter = PlotHelper(f"{self.data_path}", 1, 3)
        plotter.plot_complex(
            response[:, 0]+1j*response[:, 1], line_for_data=True)
        plotter.label("I", "Q")

        plotter.change_plot(0, 1)
        plotter.plot(time, projected[:, 0], label="PCA")
        plotter.plot(time, projected[:, 1], label="perp-PCA")
        plotter.label("Time (ns)", "Response")

        plotter.change_plot(0, 2)
        plotter.plot_fitting(time, component, y_fit=component_fit, label="PCA")
        plotter.label("Time (ns)", "Response")

        if savefig:
            plt.savefig(
                f"{savepath}/{self.data_path}.png")
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
