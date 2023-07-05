
from logging import getLogger
import os
import numpy as np
import matplotlib.pyplot as plt
from plottr.data.datadict_storage import DataDict, DDH5Writer
from tqdm import tqdm

from measurement_codes_ut.measurement_tool.wrapper import AttributeDict
from sequence_parser import Sequence
from sequence_parser.instruction import *

from measurement_codes_ut.helper.plot_helper import PlotHelper
from plottr.data.datadict_storage import datadict_from_hdf5
# from setup_td import *

logger = getLogger(__name__)


class CheckReadoutPower(object):
    experiment_name = "CheckReadoutPower"
    input_parameters = [
        "cavity_dressed_frequency_cw"
    ]
    output_parameters = [
        "cavity_readout_sequence_amplitude_expected_sn",
    ]

    def __init__(self, num_shot=1000, repetition_margin=50e3, expected_sn_ratio=5, trial_amplitude=0.05):
        self.dataset = None
        self.num_shot = num_shot
        self.expected_sn_ratio = expected_sn_ratio
        self.default_readout_amplitude = trial_amplitude
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

    def take_data(self, tdm, notes):

        note = notes.get_calibration_parameters(
            self.__class__.experiment_name, self.__class__.input_parameters)

        readout_freq = note.cavity_dressed_frequency_cw

        readout_port = tdm.port['readout'].port
        acq_port = tdm.acquire_port['readout_acquire']
        qubit_port = tdm.port['qubit'].port

        tdm.port['readout'].frequency = readout_freq - 100e6
        ports = [readout_port, acq_port, qubit_port]

        tdm.set_repetition_margin(self.repetition_margin)
        tdm.set_acquisition_delay(1000)
        tdm.set_shots(self.num_shot)

        amplitude_list = [0.0, self.default_readout_amplitude]

        sequences = []
        for amplitude in amplitude_list:
            seq = Sequence(ports)
            seq.add(ResetPhase(phase=0), readout_port, copy=False)
            seq.trigger(ports)
            seq.add(Square(amplitude=amplitude, duration=2000),
                    readout_port, copy=False)
            seq.add(Acquire(duration=2000), acq_port)

            seq.trigger(ports)
            sequences.append(seq)

        data = DataDict(
            amplitude=dict(unit=""),
            s11=dict(axes=["amplitude"]),
        )
        data.validate()

        with DDH5Writer(data, tdm.save_path, name=self.__class__.experiment_name) as writer:
            tdm.prepare_experiment(writer, __file__)
            for i, seq in enumerate(tqdm(sequences)):
                raw_data = tdm.run(
                    seq, averaging_waveform=True, averaging_shot=False, as_complex=True)
                writer.add_data(
                    amplitude=amplitude_list[i],
                    s11=raw_data['readout'],
                )

        files = os.listdir(tdm.save_path)
        date = files[-1] + '/'
        files = os.listdir(tdm.save_path+date)
        self.data_path = files[-1]

        self.data_path_all = tdm.save_path+date+self.data_path + '/'

        dataset = datadict_from_hdf5(self.data_path_all+"data")
        return dataset

    # override
    def analyze(self, dataset, calibration_note, savefig=True, savepath="./fig"):

        vacuum = dataset["s11"]["values"][0]
        signal = dataset["s11"]["values"][1]
        noise_rms = (np.std(vacuum) + np.std(signal))/2
        signal_amplitude = np.abs(np.mean(signal) - np.mean(vacuum))
        sn_ratio = np.log10(signal_amplitude / noise_rms) * 10

        if sn_ratio < self.expected_sn_ratio:
            logger.warning(
                f"SN ratio is atmost {sn_ratio}, which is too weak compared to {self.expected_sn_ratio}. Pleaser consider reducing attenuation.")
            readout_amplitude_sn_0db = self.default_readout_amplitude
        else:
            attenuation = self.expected_sn_ratio - sn_ratio
            readout_amplitude_sn_0db = self.default_readout_amplitude * \
                10**(attenuation/10.)

        plot = PlotHelper(title=self.data_path)
        plot.plot_complex(data=vacuum, label="Vacuum")
        plot.plot_complex(data=signal, label="Signal")
        plot.label("I", "Q")
        plt.legend()
        plt.tight_layout()

        if savefig:
            os.makedirs(savepath, exist_ok=True)
            plt.savefig(f"{savepath}/{self.data_path}.png",
                        bbox_inches='tight')
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.cavity_readout_noise_rms = noise_rms
        experiment_note.cavity_readout_signal_at_0dB_gain_0dB_attenuation = signal_amplitude
        experiment_note.cavity_readout_sequence_amplitude_expected_sn = readout_amplitude_sn_0db
        calibration_note.add_experiment_note(
            self.__class__.experiment_name, experiment_note, self.__class__.output_parameters)

    # override
    def report_stat(self):
        pass

    # override
    def report_visualize(self, dataset, note):
        pass
