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
from measurement_codes_ut.fitting import ResonatorReflectionModel

logger = getLogger(__name__)


class FindCavityPeak(object):
    experiment_name = "FindCavityPeak"
    input_parameters = [
        'cavity_dressed_frequency_cw',
        "cavity_readout_sequence_amplitude_expected_sn",
        "cavity_readout_electrical_delay",
        "cavity_readout_trigger_delay",
    ]
    output_parameters = [
        "cavity_dressed_frequency",
        "cavity_dressed_frequency_linewidth",
        "cavity_external_decay_rate",
        "cavity_intrinsic_decay_rate",
    ]

    def __init__(self, num_shot=1000, repetition_margin=50e3, sweep_range=50e6, sweep_step=51, model="unwrap overcoupled"):
        self.dataset = None
        self.num_shot = num_shot
        self.readout_freq_range = sweep_range
        self.readout_freq_step = sweep_step
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

    def take_data(self, tdm, notes):

        note = notes.get_calibration_parameters(
            self.__class__.experiment_name, self.__class__.input_parameters)

        readout_freq = note.cavity_dressed_frequency_cw

        readout_port = tdm.port['readout'].port
        acq_port = tdm.acquire_port['readout_acquire']
        qubit_port = tdm.port['qubit'].port

        ports = [readout_port, qubit_port, acq_port]

        tdm.set_acquisition_delay(note.cavity_readout_trigger_delay)
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

        data = DataDict(
            frequency=dict(unit="Hz"),
            s11=dict(axes=["frequency"]),
        )
        data.validate()

        shift = np.linspace(-self.readout_freq_range/2,
                            self.readout_freq_range/2, self.readout_freq_step)

        with DDH5Writer(data, tdm.save_path, name=self.__class__.experiment_name) as writer:
            tdm.prepare_experiment(writer, __file__)
            for i, df in enumerate(tqdm(shift)):
                time.sleep(0.1)
                tdm.port['readout'].frequency = readout_freq + df
                raw_data = tdm.run(
                    seq, averaging_waveform=True, averaging_shot=True, as_complex=True)
                writer.add_data(
                    frequency=readout_freq + df,
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

    # override
    def analyze(self, dataset, note, savefig=True, savepath="./fig"):

        freq = dataset["frequency"]["values"]
        signal = dataset["s11"]["values"]

        electrical_delay = note.cavity_readout_electrical_delay

        # delay correction
        response = signal * np.exp(-1.j * 2*np.pi*freq * electrical_delay)

        # fitting
        model = ResonatorReflectionModel()
        params = model.guess(
            response, freq, electrical_delay_estimation=self.model)
        rst = model.fit(response, params=params, omega=freq)
        param_list = [
            rst.params['a'].value,
            rst.params['theta'].value,
            rst.params['tau'].value,
            rst.params['kappa_ex'].value,
            rst.params['kappa_in'].value,
            rst.params['omega_0'].value
        ]
        param_error_list = [
            rst.params['a'].stderr,
            rst.params['theta'].stderr,
            rst.params['tau'].stderr,
            rst.params['kappa_ex'].stderr,
            rst.params['kappa_in'].stderr,
            rst.params['omega_0'].stderr
        ]

        fitting_parameter_list = [
            "signal_amplitude",
            "cavity_readout_electrical_phase_offset",
            "cavity_readout_electrical_delay",
            "cavity_external_decay_rate",
            "cavity_intrinsic_decay_rate",
            "cavity_dressed_frequency",
        ]

        for index, item in enumerate(fitting_parameter_list):
            name = item
            value = param_list[index]
            value_error = param_error_list[index]
            self.__setattr__(name, value)
            if value_error:
                self.__setattr__(name+"_stderr", value_error)
            else:
                self.__setattr__(name+"_stderr", None)

        fit_slice = 1001
        freq_fit = np.linspace(min(freq), max(freq), fit_slice)
        response_fit = rst.eval(params=rst.params, omega=freq_fit)
        plot = PlotHelper(f"{self.data_path}", 1, 3)
        plot.plot_fitting(freq, np.abs(response), y_fit=np.abs(
            response_fit), label="Amplitude")
        plot.label("Frequency (Hz)", "Magnitude")
        plt.ylim(0, 1.1*np.max(np.abs(response)))
        plt.axvline(
            self.cavity_dressed_frequency, color="black", linestyle="--")

        plot.change_plot(0, 1)
        plot.plot_fitting(freq, np.unwrap(np.angle(response)),
                          y_fit=np.unwrap(np.angle(response_fit)))
        plot.label("Frequency (Hz)", "Phase (rad)")
        plt.axvline(
            self.cavity_dressed_frequency, color="black", linestyle="--")

        plot.change_plot(0, 2)
        plot.plot_complex(response, fit=response_fit)
        plot.label("I", 'Q')

        if savefig:
            os.makedirs(savepath, exist_ok=True)
            plt.savefig(
                f"{savepath}/{self.data_path}.png",
                bbox_inches='tight')
        plt.tight_layout()
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.cavity_dressed_frequency = self.cavity_dressed_frequency
        experiment_note.cavity_dressed_frequency_stderr = self.cavity_dressed_frequency_stderr
        experiment_note.cavity_external_decay_rate = self.cavity_external_decay_rate
        experiment_note.cavity_external_decay_rate_stderr = self.cavity_external_decay_rate_stderr
        experiment_note.cavity_intrinsic_decay_rate = self.cavity_intrinsic_decay_rate
        experiment_note.cavity_intrinsic_decay_rate_stderr = self.cavity_intrinsic_decay_rate_stderr
        # experiment_note.cavity_readout_electrical_delay = self.cavity_readout_electrical_delay
        # experiment_note.cavity_readout_electrical_delay_stderr = self.cavity_readout_electrical_delay_stderr
        experiment_note.cavity_dressed_frequency_linewidth = self.cavity_external_decay_rate + \
            self.cavity_intrinsic_decay_rate
        note.add_experiment_note(
            self.__class__.experiment_name, experiment_note, self.__class__.output_parameters)

    # override

    def report_stat(self):
        pass

    # override
    def report_visualize(self, dataset, note):
        pass
