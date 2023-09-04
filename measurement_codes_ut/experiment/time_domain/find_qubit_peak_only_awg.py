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


logger = getLogger(__name__)


class FindQubitPeak_AWG(object):
    experiment_name = "FindQubitPeak_AWG"
    input_parameters = [
        "cavity_readout_sequence_amplitude_expected_sn",
        "cavity_readout_trigger_delay",
        "cavity_dressed_frequency",
        "qubit_frequency_cw",
    ]
    output_parameters = [
        "qubit_dressed_frequency",
        "qubit_full_linewidth",
        "qubit_control_amplitude",
    ]

    def __init__(self, num_shot=1000, repetition_margin=50e3, sweep_range=50e6, sweep_step=51, qubit_pump_amplitude=0.1, drive_length=10000):
        self.dataset = None
        self.num_shot = num_shot
        self.qubit_freq_range = sweep_range
        self.qubit_freq_step = sweep_step
        self.default_qubit_pump_amplitude = qubit_pump_amplitude
        self.drive_length = drive_length
        self.repetetion_margin = repetition_margin

    def execute(self, tdm, calibration_notes,
                update_experiment=True, update_analyze=True):
        if update_experiment:
            self.dataset = self.take_data(tdm, calibration_notes)

        if update_analyze:
            if self.dataset is None:
                raise ValueError("Data is not taken yet.")
            self.analyze(self.dataset, calibration_notes)

        return self.dataset

    def take_data(self, tdm, notes):
        note = notes.get_calibration_parameters(
            self.__class__.experiment_name, self.__class__.input_parameters)

        readout_port = tdm.port['readout'].port
        acq_port = tdm.acquire_port['readout_acquire']
        qubit_port = tdm.port['qubit'].port

        ports = [readout_port, qubit_port, acq_port]

        tdm.set_acquisition_delay(note.cavity_readout_trigger_delay)
        tdm.set_repetition_margin(self.repetetion_margin)
        tdm.set_shots(self.num_shot)
        tdm.set_acquisition_mode(averaging_waveform=True, averaging_shot=True)

        readout_freq = note.cavity_dressed_frequency

        qubit_freq = note.qubit_frequency_cw

        tdm.port['readout'].frequency = readout_freq

        detuning_range = np.linspace(-self.qubit_freq_range/2,self.qubit_freq_range/2, self.qubit_freq_step)
        detuning = Variable("detuning", detuning_range, "Hz")
        variables = Variables([detuning])

        seq = Sequence(ports)
        seq.add(SetDetuning(detuning), qubit_port)
        seq.add(Square(amplitude=self.default_qubit_pump_amplitude, duration=self.drive_length),
                qubit_port)
        # seq.add(Delay(self.drive_length), readout_port)
        seq.add(ResetPhase(phase=0), readout_port, copy=False)
        seq.trigger(ports)
        seq.add(Square(amplitude=note.cavity_readout_sequence_amplitude_expected_sn, duration=2000),
                readout_port, copy=False)
        seq.add(Acquire(duration=2000), acq_port)
        # seq.draw()

        tdm.sequence = seq
        tdm.variables = variables

        dataset = tdm.take_data(dataset_name=self.__class__.experiment_name, as_complex=False, exp_file=__file__)
        return dataset

    # override
    def analyze(self, dataset, note, savefig=True, savepath="./fig"):

        freq = dataset.data["duration"]["values"]
        signal = dataset.data["readout_acquire"]["values"]

        qubit_dressed_frequency = note.qubit_frequency_cw
        # qubit_full_linewidth = note.qubit_full_linewidth
        qubit_freq_start = qubit_dressed_frequency - self.qubit_freq_range/2
        qubit_freq_end = qubit_dressed_frequency + self.qubit_freq_range/2
        qubit_freq_num = self.qubit_freq_step

        # fit
        pca = PCA()
        projected = pca.fit_transform(signal)
        component = projected[:, 0]

        model = QubitSpectral()
        model.fit(freq, component)

        fitting_parameter_list = [
            "qubit_peak_height",
            "qubit_dressed_frequency",
            "qubit_full_linewidth",
            "base_amplitude",
        ]
        for index, item in enumerate(fitting_parameter_list):
            name = item
            value = model.param_list[index]
            value_error = model.param_error_list[index]
            setattr(self, name, value)
            setattr(self, name+"_stderr", value_error)

        ##### plot #####
        fit_slice = 1001
        component_fit = model.predict(
            np.linspace(min(freq), max(freq), fit_slice))
        
        self.data_label = dataset.path.split("/")[-1][27:]

        plot = PlotHelper(f"{self.data_label}", 1, 3)
        plot.plot_complex(signal[:, 0] + 1.j *
                          signal[:, 1], line_for_data=True)
        plot.label("I", "Q")

        plot.change_plot(0, 1)
        plot.plot_fitting(freq, projected[:, 0], label="PCA")
        plot.plot_fitting(freq, projected[:, 1], label="perp-PCA")
        plot.label("Frequency (Hz)", "Response")

        plot.change_plot(0, 2)
        plot.plot_fitting(freq, component, y_fit=component_fit, label="PCA")
        plot.label("Frequency (Hz)", "Response")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}.png",
                        bbox_inches='tight')
        plt.show()
        ##### plot #####

        experiment_note = AttributeDict()
        experiment_note.qubit_dressed_frequency = self.qubit_dressed_frequency
        experiment_note.qubit_full_linewidth = abs(self.qubit_full_linewidth)
        experiment_note.qubit_control_amplitude = self.default_qubit_pump_amplitude
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters,)

    # override

    def report_stat(self):
        pass

    # override
    def report_visualize(self, dataset, note):
        pass
