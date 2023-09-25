
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


class FindDispersiveShift(object):
    experiment_name = "FindDispersiveShift"
    input_parameters = [
        "cavity_readout_sequence_amplitude_expected_sn",
        "cavity_readout_trigger_delay",
        "cavity_readout_electrical_delay",
        "cavity_dressed_frequency",
        "qubit_dressed_frequency",
        "pi_pulse_length",
        "pi_pulse_power",
        "cavity_external_decay_rate",
        "cavity_intrinsic_decay_rate",
        "readout_pulse_length"
    ]
    output_parameters = [
        "cavity_readout_frequency",
        "cavity_dressed_frequency_g",
        "cavity_dressed_frequency_e",
        "dispersive_shift",
    ]

    def __init__(self, num_shot=1000, repetition_margin=200e3, sweep_range=200e6, sweep_offset=-10e6, num_step: int = 51, model="unwrap overcoupled"):
        self.dataset = None
        self.num_shot = num_shot
        self.sweep_range = sweep_range
        self.sweep_offset = sweep_offset
        self.num_step = num_step
        self.repetition_margin = repetition_margin
        self.model = model

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


        tdm.port['qubit'].frequency = qubit_freq

        pi_pulse_power = note.pi_pulse_power

        shift = np.linspace(self.sweep_offset - 0.5*self.sweep_range,
                            + self.sweep_offset + 0.5*self.sweep_range, self.num_step)
        
        tdm.port['readout'].frequency = readout_freq + shift
        
        amp_list = [0, pi_pulse_power]
        
        amplitude = Variable("amplitude", amp_list, "V")
        variables = Variables([amplitude])

        seq = Sequence(ports)
        seq.add(Gaussian(amplitude=amplitude, fwhm=note.pi_pulse_length/3, duration=note.pi_pulse_length, zero_end=True),
                qubit_port, copy=False)
        seq.add(Delay(10), qubit_port)
        seq.trigger(ports)
        seq.add(ResetPhase(phase=0), readout_port, copy=False)
        seq.add(Square(amplitude=note.cavity_readout_sequence_amplitude_expected_sn, duration=note.readout_pulse_length),
                readout_port, copy=False)
        seq.add(Acquire(duration=note.readout_pulse_length), acq_port)

        seq.trigger(ports)

        tdm.sequence = seq
        tdm.variables = variables

        dataset = tdm.take_data(dataset_name=self.__class__.experiment_name, as_complex=True, exp_file=__file__)

        return dataset

    def analyze(self, dataset, note, savefig=False, savepath="./fig"):

        freq = dataset.data['readout_LO_frequency']['values'][:self.num_step]
        response = dataset.data["readout_acquire"]["values"].reshape(2, self.num_step)

        cavity_linewidth = note.cavity_external_decay_rate + \
            note.cavity_intrinsic_decay_rate
        difference = np.abs(
            response[0] - response[1])

        model = QubitSpectral()
        model.fit(freq, difference)
        cavity_readout_frequency = model.param_list[1]

        # since angle rotate quickly under optimized window, angle freq is corrected in advance
        transmission = np.exp(
            -1.0j*(2.*np.pi*note.cavity_readout_electrical_delay*1*freq))
        response[0] *= transmission
        response[1] *= transmission

        fit_curve = []
        for ge_index in range(2):
            model = ResonatorReflectionModel()
            params = model.guess(
                response[ge_index, :], freq, electrical_delay_estimation=self.model)
            rst = model.fit(response[ge_index, :], params=params, omega=freq)
            pred = rst.best_fit
            fit_curve.append(pred)
            error = np.mean(np.abs(pred-response[ge_index, :])**2)

            if ge_index == 0:
                cavity_dressed_frequency_g = rst.params['omega_0'].value
            if ge_index == 1:
                cavity_dressed_frequency_e = rst.params['omega_0'].value

        self.data_label = dataset.path.split("/")[-1][27:]
        plotter = PlotHelper(f"{self.data_label}", 1, 3)
        plotter.plot_complex(data=response[0, :], label="Ground")
        plotter.plot_complex(data=response[1, :], label="Excite")
        plotter.label("I", "Q")
        plotter.change_plot(0, 1)
        plotter.plot(freq, np.unwrap(np.angle(response[0, :])), label="ground")
        plotter.plot(freq, np.unwrap(np.angle(response[1, :])), label="excite")
        plotter.plot(freq, np.unwrap(
            np.angle(fit_curve[0])), label="ground fit")
        plotter.plot(freq, np.unwrap(
            np.angle(fit_curve[1])), label="excite fit")
        plotter.label("Frequency (Hz)", "Phase (rad)")
        plotter.change_plot(0, 2)
        plotter.plot(freq, difference, label='GE-distance')
        # plt.axvline(
        #     cavity_dressed_frequency_g["MHz"], color="blue", linestyle="--", label="cavity g freq")
        plt.axvline(
            cavity_readout_frequency, color="black", linestyle="--")
        # plt.axvline(
        #     cavity_dressed_frequency_e["MHz"], color="red", linestyle="--", label="cavity e freq")
        plotter.label("Frequency (Hz)", "G-E distance")
        plt.legend()
        plt.tight_layout()

        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}.png")
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.cavity_dressed_frequency_g = cavity_dressed_frequency_g
        experiment_note.cavity_dressed_frequency_e = cavity_dressed_frequency_e
        experiment_note.dispersive_shift = (
            cavity_dressed_frequency_g - cavity_dressed_frequency_e)/2
        experiment_note.cavity_readout_frequency = cavity_readout_frequency
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters)

    def report_stat(self):
        pass

    def report_visualize(self, dataset, note):
        pass
