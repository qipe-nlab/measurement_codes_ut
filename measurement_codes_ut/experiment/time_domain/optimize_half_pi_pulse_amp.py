from logging import getLogger
from logging import getLogger
import os
import numpy as np
import matplotlib.pyplot as plt
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sklearn.decomposition import PCA
from tqdm import tqdm
from measurement_codes_ut.util import LPF

from measurement_codes_ut.measurement_tool.wrapper import AttributeDict
from sequence_parser import Port, Sequence, Circuit
from sequence_parser.instruction import *

from measurement_codes_ut.helper.plot_helper import PlotHelper
from plottr.data.datadict_storage import datadict_from_hdf5


logger = getLogger(__name__)


class OptimizeHalfPiAmp(object):
    experiment_name = "OptimizeHalfPiAmp"
    input_parameters = [
        "cavity_readout_amplitude",
        "cavity_readout_trigger_delay",
        "cavity_readout_window_coefficient",
        "cavity_readout_frequency",
        "qubit_dressed_frequency",
        "pi_pulse_power",
        "pi_pulse_length",
        "half_pi_pulse_power",
        "half_pi_pulse_drag",
    ]
    output_parameters = [
        "half_pi_pulse_power",
    ]

    def __init__(self, rep, amp_range, num_shot=1000, repetition_margin=200e3):
        self.name = self.__class__.experiment_name
        self.dataset = None
        self.rep = rep
        self.amp_range = amp_range
        self.num_shot = num_shot
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

        half_pi_pulse_length = note.pi_pulse_length * 0.5

        amp_range = self.amp_range
        phase_list = [0, np.pi]
        seq_list = []
        for phase in phase_list:
            seq_list_in = []
            for amp in amp_range:
                seq = Sequence(ports)
                rx90 = Sequence(ports)
                with rx90.align(qubit_port, 'left'):
                    rx90.add(Gaussian(amplitude=amp, fwhm=half_pi_pulse_length/3, duration=half_pi_pulse_length, zero_end=True),
                             qubit_port, copy=False)
                    rx90.add(Deriviative(Gaussian(amplitude=1j*amp*note.half_pi_pulse_drag, fwhm=half_pi_pulse_length /
                                                  3, duration=half_pi_pulse_length, zero_end=True)), qubit_port, copy=False)
                for _ in range(4*self.rep):
                    seq.call(rx90)

                seq.add(VirtualZ(-phase), qubit_port)
                seq.call(rx90)
                seq.add(VirtualZ(phase), qubit_port)

                seq.trigger(ports)

                seq.add(ResetPhase(phase=0), readout_port, copy=False)
                seq.add(Square(amplitude=note.cavity_readout_amplitude, duration=2000),
                        readout_port, copy=False)
                seq.add(Acquire(duration=2000), acq_port)

                seq.trigger(ports)
                seq_list_in.append(seq)
            seq_list.append(seq_list_in)
        # seq.draw()

        data = DataDict(
            amplitude=dict(unit=""),
            phase=dict(unit="rad"),
            s11=dict(axes=["amplitude", "phase"]),
        )
        data.validate()

        with DDH5Writer(data, tdm.save_path, name=self.__class__.experiment_name) as writer:
            tdm.prepare_experiment(writer, __file__)
            for i, seq_list_in in enumerate((seq_list)):
                for j, seq in enumerate(tqdm(seq_list_in)):
                    raw_data = tdm.run(seq, averaging_waveform=True,
                                       averaging_shot=True, as_complex=False)
                    writer.add_data(
                        amplitude=amp_range[j],
                        phase=phase_list[i],
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

    def analyze(self, dataset, note):

        amp_range = self.amp_range
        response = dataset['s11']['values'].reshape(2, len(self.amp_range), 2)

        pm_label = ["+", "-"]
        ge_label = ["0", "1"]

        response_dict = {
            "p": response[0],
            "m": response[1],
        }

        pca = PCA()
        for key, val in response_dict.items():
            pca.fit(val)

        component_dict = {}
        for key, val in response_dict.items():
            component_dict[key] = pca.transform(val)[:, 0]

        plot_amp = np.linspace(amp_range[0], amp_range[-1], 101)

        plt.figure(figsize=(8, 6))
        plt.title(f"{self.data_path}")

        fit_params = {}
        for key, val in component_dict.items():
            plt.plot(amp_range, val, ".-", label=key)
            slope, offset = np.polyfit(amp_range, val, 1)
            plt.plot(plot_amp, slope*plot_amp+offset, "k", label=f"fit {key}")
            fit_params[key] = [slope, offset]

        center = (fit_params["p"][1] - fit_params["m"][1]) / \
            (fit_params["m"][0] - fit_params["p"][0])

        plt.axvline(center, color="r", linestyle="-")
        plt.xlabel('Pulse amplitude')
        plt.ylabel('Pauli Z')
        # plt.ylim(-1, 1)
        plt.legend()
        plt.show()

        width = 0.5*(np.max(amp_range) - np.min(amp_range))

        self.hpi_center = center
        self.hpi_width = width

        experiment_note = AttributeDict()
        experiment_note.half_pi_pulse_power = self.hpi_center
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters)
