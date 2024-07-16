
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
from sklearn.metrics import confusion_matrix

from measurement_codes_ut.fitting.gaussian_fitter import GaussianFitter


logger = getLogger(__name__)


class OptimizeReadoutWindow(object):
    experiment_name = "OptimizeReadoutWindow"
    input_parameters = [
        "cavity_readout_sequence_amplitude_expected_sn",
        "cavity_readout_trigger_delay",
        # "cavity_dressed_frequency",
        "cavity_readout_frequency",
        "qubit_dressed_frequency",
        "qubit_full_linewidth",
        "qubit_control_amplitude",
        "rabi_frequency",
        "pi_pulse_length",
        "pi_pulse_power",
        "cavity_readout_amplitude",
        "readout_pulse_length"
    ]
    output_parameters = [
        "cavity_readout_window_coefficient",
        "measurement_axis",
        "readout_assignment_border",
        "readout_g_direction"
    ]

    def __init__(self, num_shot=1000, repetition_margin=200e3):
        self.dataset = None
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
        tdm.set_acquisition_mode(averaging_waveform=False, averaging_shot=False)

        readout_freq = note.cavity_readout_frequency

        qubit_freq = note.qubit_dressed_frequency

        tdm.port['readout'].frequency = readout_freq
        tdm.port['readout'].window = None
        if tdm.lo['qubit'] is None:
            qubit_port.if_freq = qubit_freq/1e9
        else:
            tdm.port['qubit'].frequency = qubit_freq


        qubit_amp_range = [0, note.pi_pulse_power]
        
        qubit_amplitude = Variable("qubit_amplitude", qubit_amp_range, "V")
        readout_amplitude = note.cavity_readout_amplitude
        variables = Variables([qubit_amplitude])

        seq = Sequence(ports)
        seq.add(Gaussian(amplitude=qubit_amplitude, fwhm=note.pi_pulse_length/3, duration=note.pi_pulse_length, zero_end=True),
                qubit_port, copy=False)
        seq.add(Delay(10), qubit_port)
        seq.trigger(ports)
        seq.add(ResetPhase(phase=0), readout_port, copy=False)
        seq.add(Square(amplitude=readout_amplitude, duration=note.readout_pulse_length),
                readout_port, copy=False)
        seq.add(Acquire(duration=note.readout_pulse_length), acq_port)

        seq.trigger(ports)

        tdm.sequence = seq
        tdm.variables = variables

        dataset = tdm.take_data(dataset_name=self.__class__.experiment_name, as_complex=True, exp_file=__file__)

        return dataset

    def analyze(self, dataset, note, savefig=True, savepath="./fig"):

        response = dataset.data['readout_acquire']['values'].reshape(
            2, self.num_shot, int(note.readout_pulse_length/2))
        self.data_label = dataset.path.split("/")[-1][27:]


        def moving_average(y, w):
            if w % 2 != 1:
                raise ValueError('w must be odd number')
            return np.convolve([1/w]*w, y, mode="full")[int(w/2):-int(w/2)]

        def gaussian(x, sig, A1, C1):
            inv_var = 1/(2*sig**2)
            y = A1*np.exp(-inv_var*(x-C1)**2)
            return y

        
        iq_data = response
        g_data = iq_data[0]  # shot, time
        e_data = iq_data[1]


        window_iq = moving_average(g_data.mean(axis=0) - e_data.mean(axis=0), 51).conjugate()
        window_iq /= np.sum(abs(window_iq)**2)
        time = 2 * np.arange(len(window_iq))
        plt.figure()
        plt.title(f"{self.data_label}")
        plt.plot(time, window_iq.real, label='Real')
        plt.plot(time, window_iq.imag, label='Imag')
        plt.xlabel('Time (ns)')
        plt.ylabel('Window')
        plt.legend()
        
        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}_window.png")
        plt.show()

        g_data_weighted = g_data @ window_iq
        e_data_weighted = e_data @ window_iq
        axis = [1,0]
        if np.mean(g_data_weighted.real) < np.mean(e_data_weighted.real):
            axis = [-1,0]
        g_projected = g_data_weighted.real * axis[0]
        e_projected = e_data_weighted.real * axis[0]

        xmin = min(min(g_projected), min(e_projected))
        xmax = max(max(g_projected), max(e_projected))
        
        fig, ax = plt.subplots(1, 2, figsize=(6,3))
        fig.suptitle(f"{self.data_label}")
        bin_size = 101
        H = ax[0].hist2d(g_data_weighted.real, g_data_weighted.imag, bins=bin_size)
        ax[0].set_title('g')
        ax[0].set_xlabel('Real')
        ax[0].set_ylabel('Imag')
        ax[0].set_aspect('equal')

        H = ax[1].hist2d(e_data_weighted.real, e_data_weighted.imag, bins=bin_size)
        ax[1].set_title('e')
        ax[1].set_xlabel('Real')
        ax[1].set_ylabel('Imag')
        ax[1].set_aspect('equal')
        fig.tight_layout()
        plt.show()
        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}_hist2d.png")

        fidelity_list = []
        border_range = np.linspace(xmin, xmax, 21)
        for border in border_range:
        # border = -0.00003
            pred = np.array([[0 if data-border > 0 else 1 for data in i_data] for i_data in [g_projected, e_projected]]).reshape(-1)
            label = np.array([0]*self.num_shot+[1]*self.num_shot)
            cm = confusion_matrix(label, pred, normalize="true")
            # print(cm)
            fidelity_list.append(np.trace(cm)/2)

        def gaussian(x, a, b, c, d):
            return a*np.exp(-b*(x-c)**2)+d
        a0 = max(fidelity_list)-min(fidelity_list)
        b0 = (2*np.std(g_projected)**2)**(-1)
        c0 = 0
        d0 = min(fidelity_list)
        popt, pcov = curve_fit(gaussian, border_range, fidelity_list, p0=[a0, b0, c0, d0])
        border = popt[2]
    # print()
        pred = np.array([[0 if data-border > 0 else 1 for data in i_data] for i_data in [g_projected, e_projected]]).reshape(-1)
        label = np.array([0]*self.num_shot+[1]*self.num_shot)
        cm = confusion_matrix(label, pred, normalize="true")

        f = np.trace(cm) / 2 
        window_opt = window_iq

        self.data_label = dataset.path.split("/")[-1][27:]

        fig, ax = plt.subplots(1, 2, figsize=(6,3))
        fig.suptitle(f"{self.data_label}")

        bin_size = 101
        ax[0].hist(g_projected, bins=bin_size, alpha=0.5, label='Prepare g')
        ax[0].hist(e_projected, bins=bin_size, alpha=0.5, label='Prepare e')
        ax[0].axvline(border, color='black', ls='--')
        ax[0].set_yscale('log')
        ax[0].legend()
        ax[0].set_xlabel('Signal (a.u.)')
        ax[0].set_ylabel('Count')
        ax[0].set_ylim(bottom=1)

        # cm = cm_list[chosen_index]
        f = np.mean([cm[0, 0], cm[1, 1]])
        ax[1].set_title(f'Fidelity : {f:.4f}')
        ax[1].imshow(cm, clim=(0, 1))
        ax[1].set_xticks([0, 1])
        ax[1].set_yticks([0, 1])
        ax[1].set_xticklabels(["0", "1"])
        ax[1].set_yticklabels(["0", "1"])
        ax[1].text(x=0, y=0, s=cm[0, 0], color='red')
        ax[1].text(x=0, y=1, s=cm[1, 0], color='red')
        ax[1].text(x=1, y=0, s=cm[0, 1], color='red')
        ax[1].text(x=1, y=1, s=cm[1, 1], color='red')
        ax[1].set_xlabel("Measured")
        ax[1].set_ylabel("Prepared")

        fig.tight_layout()
        
        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}.png")
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.cavity_readout_window_coefficient = window_opt
        experiment_note.measurement_axis = np.array(axis)
        experiment_note.readout_assignment_border = border
        experiment_note.readout_g_direction = 1
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters)

    def report_stat(self):
        pass

    def report_visualize(self, dataset, note):
        pass
