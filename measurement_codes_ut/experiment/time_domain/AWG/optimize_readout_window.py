
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
        "readout_assignment_border"
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
        qubit_port.if_freq = qubit_freq/1e9

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
        window_iq /= np.sum(abs(window_iq))
        time = 2 * np.arange(len(window_iq))
        plt.figure()
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
        # data_list.append([g_data_weighted, e_data_weighted])

        g_data_xy = np.array(
            [g_data_weighted.real, g_data_weighted.imag]).T
        e_data_xy = np.array(
            [e_data_weighted.real, e_data_weighted.imag]).T

        gf = GaussianFitter(g_data_xy, e_data_xy, n_peak=1, grid=101)
        popt0, popt1 = gf.fitter()
        x = gf.x
        data0_hist, data1_hist = gf.data0_hist, gf.data1_hist

        def DoubleGaussian(x, a0, mu0, a1, mu1, sigma):
            inv_var = 1/(2*sigma**2)
            y0 = a0 * np.exp(-(x-mu0)**2*inv_var)
            y1 = a1 * np.exp(-(x-mu1)**2*inv_var)
            return y0 + y1

        popt_0, pcov = curve_fit(DoubleGaussian, x, data0_hist, p0=[max(data0_hist), gf.mean1, 0.1*max(data0_hist), gf.mean2, gf.std1], maxfev = 10000)
        popt_1, pcov = curve_fit(DoubleGaussian, x, data1_hist, p0=[max(data1_hist), gf.mean2, 0.1*max(data1_hist), gf.mean1, gf.std2], maxfev = 10000)
        # print(popt_0, popt_1)
    # print(popt)
        border = (popt_0[1]+popt_0[3])/2
        which = popt_0[1] - border
        axis = gf.pca.components_[0]
        distance = abs(popt_0[1]-popt_0[3])
        noise = popt_0[-1]

        component_g = np.dot(axis, g_data_xy.T)
        prep_g = [0 if (compo-border)*which>=0 else 1 for compo in component_g]
        measured_g = 1 - np.count_nonzero(prep_g) / len(prep_g)
        component_e = np.dot(axis, e_data_xy.T)
        prep_e = [0 if (compo-border)*which>=0 else 1 for compo in component_e]
        measured_e = np.count_nonzero(prep_e) / len(prep_e)

        cm = np.array([[round(measured_g, 4), round(1-measured_g, 4)], [round(1-measured_e, 4), round(measured_e, 4)]])
        f = np.trace(cm) / 2 
        window_opt = window_iq

        self.data_label = dataset.path.split("/")[-1][27:]

        data0_hist, data1_hist = gf.data0_hist, gf.data1_hist
        threshold = gf.threshold

        fig, ax = plt.subplots(1, 3, figsize=(11,3))
        fig.suptitle(f"{self.data_label}")

        ax[0].scatter(g_data_weighted.real, g_data_weighted.imag,
                      color='blue', label='Ground', marker=".")
        ax[0].scatter(e_data_weighted.real, e_data_weighted.imag,
                      color='red', label='Excited', marker=".")
        ax[0].legend()
        ax[0].set_aspect('equal')
        ax[0].set_xlabel('I')
        ax[0].set_ylabel('Q')

        ax[1].plot(x, DoubleGaussian(x, *popt_0), ls='--', color='blue', label='Ground Double-Gaussian fit', lw=1)
        ax[1].plot(x, data0_hist, 'o', markersize=3,
                   color='blue', label='Ground')
        ax[1].plot(x, DoubleGaussian(x, *popt_1), ls='--', color='red', label='Excited Double-Gaussian fit', lw=1)
        ax[1].plot(x, data1_hist, 'o', markersize=3,
                   color='red', label='Excited')
        ax[1].axvline(border, linestyle='--',
                      color='black', label='Threshold')
        ax[1].set_xlabel('Projected')
        ax[1].set_ylabel('Counts')
        # ax[1].set_yscale('log')
        ax[1].legend()

        # cm = cm_list[chosen_index]
        f = np.mean([cm[0, 0], cm[1, 1]])
        ax[2].set_title(f'Fidelity : {f:.4f}')
        ax[2].imshow(cm, clim=(0, 1))
        ax[2].set_xticks([0, 1])
        ax[2].set_yticks([0, 1])
        ax[2].set_xticklabels(["0", "1"])
        ax[2].set_yticklabels(["0", "1"])
        ax[2].text(x=0, y=0, s=cm[0, 0], color='red')
        ax[2].text(x=0, y=1, s=cm[1, 0], color='red')
        ax[2].text(x=1, y=0, s=cm[0, 1], color='red')
        ax[2].text(x=1, y=1, s=cm[1, 1], color='red')
        ax[2].set_xlabel("Predicted")
        ax[2].set_ylabel("Prepared")

        fig.tight_layout()
        
        if savefig:
            plt.savefig(f"{savepath}/{self.data_label}.png")
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.cavity_readout_window_coefficient = window_opt
        experiment_note.measurement_axis = axis
        experiment_note.readout_assignment_border = border
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters)

    def report_stat(self):
        pass

    def report_visualize(self, dataset, note):
        pass
