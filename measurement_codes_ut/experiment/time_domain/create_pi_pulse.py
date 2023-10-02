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

logger = getLogger(__name__)


class CreatePiPulse(object):
    experiment_name = "CreatePiPulse"
    input_parameters = [
        "cavity_readout_sequence_amplitude_expected_sn",
        "cavity_readout_trigger_delay",
        "cavity_readout_skew",
        "cavity_dressed_frequency",
        "qubit_dressed_frequency",
        "qubit_full_linewidth",
        "qubit_control_amplitude",
        "rabi_frequency",
        "pi_pulse_length",
        "readout_pulse_length"
    ]
    output_parameters = [
        "pi_pulse_power",
    ]

    def __init__(self, num_shot=1000, repetition_margin=200e3, num_point: int = 51, ):
        self.dataset = None
        self.num_shot = num_shot
        self.num_point = num_point
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

        readout_freq = note.cavity_dressed_frequency

        qubit_freq = note.qubit_dressed_frequency

        tdm.port['readout'].frequency = readout_freq

        tdm.port['qubit'].frequency = qubit_freq

        rabi_period = (1./note.rabi_frequency)
        time_ratio = float(rabi_period / note.pi_pulse_length) * 3
        max_power = min(note.qubit_control_amplitude *
                        time_ratio, qubit_port.max_amp)

        amp_range = np.linspace(0.0, max_power, self.num_point)

        amplitude = Variable("amplitude", amp_range, "V")
        variables = Variables([amplitude])

        seq = Sequence(ports)
        seq.add(Gaussian(amplitude=amplitude, fwhm=note.pi_pulse_length/3, duration=note.pi_pulse_length, zero_end=True),
                qubit_port, copy=False)
        seq.trigger(ports)
        seq.add(Delay(note.cavity_readout_skew), readout_port)
        seq.add(Delay(note.cavity_readout_skew), acq_port)    
        seq.add(ResetPhase(phase=0), readout_port, copy=False)
        seq.add(Square(amplitude=note.cavity_readout_sequence_amplitude_expected_sn, duration=note.readout_pulse_length),
                readout_port, copy=False)
        seq.add(Acquire(duration=note.readout_pulse_length), acq_port)

        seq.trigger(ports)
        tdm.sequence = seq
        tdm.variables = variables

        dataset = tdm.take_data(dataset_name=self.__class__.experiment_name, as_complex=False, exp_file=__file__)
        return dataset

    def analyze(self, dataset, note, savefig=False, savepath="./fig"):

        power = dataset.data['amplitude']['values']
        response = dataset.data["readout_acquire"]["values"]

        pca = PCA()
        projected = pca.fit_transform(response)
        component = projected[:, 0]

        if np.where(component > component[0])[0].size > 0.5*component.size:
            component *= -1
        else:
            component *= +1

        def func(t, a, f, xofs, yofs):
            return a*np.cos(2*np.pi*(t-xofs)*f)+yofs

        amp = (np.max(component) - np.min(component))/2
        freq = 0.5/power[-1]
        xofs = 0.
        yofs = (np.max(component) + np.min(component))/2
        fit_param = curve_fit(func, power, component, [amp, freq, xofs, yofs])
        fit_curve = func(power, *fit_param[0])

        freq = np.abs(fit_param[0][1])
        half_cycle = 0.5/freq
        xofs = fit_param[0][2]
        mod_offset = np.mod(xofs + half_cycle/2, half_cycle) - half_cycle/2
        pi_pulse_power = mod_offset + half_cycle

        # if pi_pulse_power > qubit_port.max_amp:
        #     pi_pulse_power = 0.999
        #     logger.warning(f"pi_pulse_power is set{pi_pulse_power} > 1.0, so this is set 0.999. Fitting would fail.")

        self.data_label = dataset.path.split("/")[-1][27:]
        plotter = PlotHelper(f"{self.data_label}", 1, 3)
        plotter.plot_complex(
            response[:, 0] + 1.j * response[:, 1], line_for_data=True)
        plotter.label("I", "Q")

        plotter.change_plot(0, 1)
        plotter.plot(power, projected[:, 0], label="PCA")
        plotter.plot(power, projected[:, 1], label="perp-PCA")
        plotter.label("Pulse amplitude", "Response")

        plotter.change_plot(0, 2)
        plotter.plot(power, component, label="PCA")
        plotter.plot(power, fit_curve, label="PCA fit")
        plt.axvline(pi_pulse_power, color="black", linestyle="--")
        plotter.label("Pulse amplitude", "Response")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}.png")
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.pi_pulse_power = pi_pulse_power
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters,)

    def report_stat(self):
        pass

    def report_visualize(self, dataset, note):
        pass
