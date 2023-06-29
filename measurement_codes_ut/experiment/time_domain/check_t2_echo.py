
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


class CheckT2Echo(object):
    experiment_name = "CheckT2Echo"
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
        "rabi_t1",
        "t1",
    ]
    output_parameters = [
        "t2_echo",
    ]

    def __init__(self, num_shot=1000, repetition_margin=200e3, min_duration=100, max_duration=100e3,  num_sample: int = 51):
        self.dataset = None
        self.num_shot = num_shot
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

        tdm.port['qubit'].frequency = qubit_freq

        tdm.port['readout'].window = note.cavity_readout_window_coefficient

        half_pi_pulse_power = note.pi_pulse_power
        half_pi_pulse_length = note.pi_pulse_length * 0.5
        pi_pulse_power = note.pi_pulse_power
        pi_pulse_length = note.pi_pulse_length

        time_step = self.num_sample
        min_dur_base2 = np.log2(self.min_duration/2)
        max_dur_base2 = np.log2(self.max_duration/2)
        time_range = np.logspace(
            min_dur_base2, max_dur_base2, time_step, base=2, dtype=int) * 2

        seq_list = []
        for dur in time_range:
            seq = Sequence(ports)
            seq.add(Gaussian(amplitude=half_pi_pulse_power, fwhm=half_pi_pulse_length/3, duration=half_pi_pulse_length, zero_end=True),
                    qubit_port, copy=False)
            seq.add(Delay(dur/2), qubit_port)
            seq.add(Gaussian(amplitude=pi_pulse_power, fwhm=pi_pulse_length/3, duration=pi_pulse_length, zero_end=True),
                    qubit_port, copy=False)
            seq.add(Delay(dur/2), qubit_port)
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

        def exp_decay(x, a, c, b):
            return a * np.exp(-x*c) + b

        pca = PCA()
        pca.fit(response)
        component = pca.transform(response)[:, 0]

        b_init = component[-1]
        a_init = component[0] - component[-1]
        c_init = 1/note.t1

        p_init = [a_init, c_init, b_init]

        popt, pcov = curve_fit(exp_decay, time, component, p0=p_init)

        component_fit = exp_decay(time, *popt)

        plotter = PlotHelper(f"{self.data_path}", 1, 2)
        plotter.plot_complex(
            data=response[:, 0]+1j*response[:, 1], label="data")
        plotter.label("I", "Q")
        plotter.change_plot(0, 1)

        plotter.plot_fitting(time, component, y_fit=component_fit)
        # plt.xscale("log")
        plotter.label("Time (ns)", "Response")

        if savefig:
            plt.savefig(f"{savepath}/{self.data_path}.png")
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.t2_echo = 1./popt[1]
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters, )

    def report_stat(self):
        pass

    def report_visualize(self, dataset, note):
        pass
