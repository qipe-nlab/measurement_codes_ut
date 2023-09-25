
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


class CheckXYRamsey(object):
    experiment_name = "CheckXYRamsey"
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
        "readout_pulse_length"
    ]
    output_parameters = [
        "qubit_dressed_frequency",
        "ramsey_frequency"
    ]

    def __init__(self, num_shot=1000, repetition_margin=200e3, min_duration=10, max_duration=1000, hand_detune=1e6):
        self.dataset = None
        self.num_shot = num_shot
        self.hand_detune = hand_detune
        self.min_duration = min_duration
        self.max_duration = max_duration
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

        readout_freq = note.cavity_readout_frequency

        qubit_freq = note.qubit_dressed_frequency

        tdm.port['readout'].frequency = readout_freq

        tdm.port['qubit'].frequency = qubit_freq + self.hand_detune

        tdm.port['readout'].window = note.cavity_readout_window_coefficient

        num_sample = 50
        delay_range = np.arange(
            self.min_duration/2, self.max_duration/2, num_sample, dtype=int) * 2
        delay_range = np.array(sorted(set(delay_range)))
        self.len_data = len(delay_range)
        phase_list = [np.pi, np.pi/2]

        half_pi_pulse_power = 0.5 * note.pi_pulse_power

        delay = Variable("duration", delay_range, "ns")
        phase = Variable("phase", phase_list, "rad")
        variables = Variables([delay, phase])

        seq = Sequence(ports)
        seq.add(Gaussian(amplitude=half_pi_pulse_power, fwhm=note.pi_pulse_length/3, duration=note.pi_pulse_length, zero_end=True),
                qubit_port, copy=False)
        seq.add(Delay(delay), qubit_port)
        seq.add(VirtualZ(phase), qubit_port)
        seq.add(Gaussian(amplitude=half_pi_pulse_power, fwhm=note.pi_pulse_length/3, duration=note.pi_pulse_length, zero_end=True),
                qubit_port, copy=False)
        seq.trigger(ports)
        seq.add(ResetPhase(phase=0), readout_port, copy=False)
        seq.add(Square(amplitude=note.cavity_readout_amplitude, duration=note.readout_pulse_length),
                readout_port, copy=False)
        seq.add(Acquire(duration=note.readout_pulse_length), acq_port)

        seq.trigger(ports)

        tdm.sequence = seq
        tdm.variables = variables

        dataset = tdm.take_data(dataset_name=self.__class__.experiment_name, as_complex=False, exp_file=__file__, sweep_axis=[1,0])
        return dataset

    def analyze(self, dataset, note, savefig=False, savepath="./fig"):

        tdata = dataset.data['duration']['values'][:self.len_data]
        response = dataset.data['readout_acquire']['values'].reshape(
            2, self.len_data, 2)  # phase, delay, IQ

        xdata = response[0]
        ydata = response[1]

        pca = PCA()
        pca.fit(xdata)
        pca.fit(ydata)
        paulix = pca.transform(xdata)[:, 0]
        pauliy = pca.transform(ydata)[:, 0]

        phase = np.unwrap(np.angle(paulix + 1j*pauliy))
        a, b = np.polyfit(tdata, phase, 1)
        detuning = a/(2*np.pi)*1e9
        
        self.data_label = dataset.path.split("/")[-1][27:]

        plt.figure(figsize=(15, 5))
        plt.suptitle(f"{self.data_label}")
        plt.subplot(131)
        plt.plot(xdata[:, 0], xdata[:, 1], ".-", label="X")
        plt.plot(ydata[:, 0], ydata[:, 1], ".-", label="Y")
        plt.axis("equal")
        plt.legend()
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.grid()
        plt.subplot(132)
        plt.plot(paulix, pauliy, ".-")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid()
        plt.subplot(133)
        plt.plot(tdata, phase, ".-", label="Data")
        plt.plot(tdata, a*tdata + b, label="Fit")
        plt.xlabel("Ramsey time (ns)")
        plt.ylabel("Phase (rad)")
        plt.grid()
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}_1.png")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.suptitle(f"{self.data_label}")
        plt.subplot(121)
        plt.plot(tdata, paulix, label="X")
        plt.plot(tdata, pauliy, label="Y")
        plt.xlabel("Time (ns)")
        plt.ylabel("IQ Proj")
        plt.legend()
        plt.grid()
        plt.subplot(122)
        dt = tdata[1] - tdata[0]
        freq = np.fft.fftfreq(tdata.size)*1/dt

        powerx = abs(np.fft.fft(paulix))**2
        powerx = powerx[freq > 0]/1e6
        powery = abs(np.fft.fft(pauliy))**2
        powery = powery[freq > 0]/1e6
        freq = freq[freq > 0]*1e9

        plt.plot(freq, powerx, ".-")
        plt.plot(freq, powery, ".-")
        plt.xlabel("Detuning (Hz)", fontsize=16)
        plt.ylabel("Power spectrum ($\mathrm{V}^2$)",  fontsize=16)
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(which='major', color='black', linestyle='-', alpha=0.1)
        plt.grid(which='minor', color='black', linestyle='-', alpha=0.1)
        plt.axvline(self.hand_detune, color="black", linestyle="--")
        plt.tight_layout()
        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}_2.png")
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.hand_detune = self.hand_detune
        experiment_note.ramsey_frequency = detuning
        experiment_note.qubit_dressed_frequency = note.qubit_dressed_frequency + \
            self.hand_detune - detuning
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters)

    def report_stat(self):
        pass

    def report_visualize(self, dataset, note):
        pass
