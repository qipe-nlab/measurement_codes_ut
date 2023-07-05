
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
from measurement_codes_ut.fitting.rabi_oscillation import RabiOscillation

from scipy.optimize import curve_fit

from measurement_codes_ut.fitting.gaussian_fitter import GaussianFitter


logger = getLogger(__name__)


class OptimizeReadoutPowerAndWindow(object):
    experiment_name = "OptimizeReadoutPowerAndWindow"
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
    ]
    output_parameters = [
        "cavity_readout_amplitude",
        "cavity_readout_window_coefficient"
    ]

    def __init__(self, num_shot=1000, repetition_margin=200e3, num_point=41, min_amplitude=0.0, max_amplitude=1.0, fidelity_fluctuate=0.01):
        self.dataset = None
        self.num_shot = num_shot
        self.num_point = num_point
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.fidelity_fluctuate = fidelity_fluctuate
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

        amp_range = np.linspace(
            self.min_amplitude, self.max_amplitude, self.num_point)

        pi_pulse_power = note.pi_pulse_power

        seq_list = []
        for qubit in [0, 1]:
            seq_list_in = []
            for amp in amp_range:
                seq = Sequence(ports)
                seq.add(Gaussian(amplitude=pi_pulse_power*qubit, fwhm=note.pi_pulse_length/3, duration=note.pi_pulse_length, zero_end=True),
                        qubit_port, copy=False)
                seq.add(Delay(10), qubit_port)
                seq.trigger(ports)
                seq.add(ResetPhase(phase=0), readout_port, copy=False)
                seq.add(Square(amplitude=amp, duration=2000),
                        readout_port, copy=False)
                seq.add(Acquire(duration=2000), acq_port)

                seq.trigger(ports)
                seq_list_in.append(seq)
            seq_list.append(seq_list_in)

        data = DataDict(
            qubit_amplitude=dict(unit=""),
            readout_amplitude=dict(unit=""),
            s11=dict(axes=["qubit_amplitude", "readout_amplitude"]),
        )
        data.validate()

        with DDH5Writer(data, tdm.save_path, name=self.__class__.experiment_name) as writer:
            tdm.prepare_experiment(writer, __file__)
            for i, seq_list_in in enumerate(seq_list):
                for j, seq in enumerate(tqdm(seq_list_in)):
                    tdm.load_sequence(seq, cycles=self.num_shot)
                    raw_data = tdm.run(seq, averaging_shot=False,
                                       averaging_waveform=False, as_complex=True)
                    writer.add_data(
                        qubit_amplitude=i,
                        readout_amplitude=amp_range[j],
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

        power_list = dataset['readout_amplitude']['values'][:self.num_point]
        response = dataset['s11']['values'].reshape(
            2, self.num_point, self.num_shot, 1000)

        cm_list = []
        fitter_list = []
        window_list = []
        popt_list = []
        data_list = []

        def moving_average(y, w):
            if w % 2 != 1:
                raise ValueError('w must be odd number')
            return np.convolve([1/w]*w, y, mode="full")[int(w/2):-int(w/2)]

        def gaussian(x, sig, A1, C1):
            inv_var = 1/(2*sig**2)
            y = A1*np.exp(-inv_var*(x-C1)**2)
            return y

        for idx in range(len(power_list)):
            # qubit_amplitude, readout_amplitude, shot, time
            iq_data = response[:, idx, :, :]
            g_data = iq_data[0]  # shot, time
            e_data = iq_data[1]

            g_avg_shot = np.mean(g_data, axis=0)
            e_avg_shot = np.mean(e_data, axis=0)

            window_iq = moving_average(g_avg_shot - e_avg_shot, 11).conjugate()
            window_iq /= np.sum(abs(window_iq))
            window_list.append(window_iq)

            g_data_weighted = g_data @ window_iq
            e_data_weighted = e_data @ window_iq
            data_list.append([g_data_weighted, e_data_weighted])

            g_data_xy = np.array(
                [g_data_weighted.real, g_data_weighted.imag]).T
            e_data_xy = np.array(
                [e_data_weighted.real, e_data_weighted.imag]).T

            gf = GaussianFitter(g_data_xy, e_data_xy, n_peak=1, grid=101)
            popt0, popt1 = gf.fitter()
            x = gf.x
            popt_list.append([popt0, popt1])

            pred, cm = gf.get_pred()
            fitter_list.append(gf)
            cm = (cm.T/cm.sum(axis=1)).T

            cm_list.append(cm)

        fid = [np.mean([cm[0, 0], cm[1, 1]]) for cm in cm_list]

        max_fidelity = np.max(fid)
        # fidelity_fluctuate = 0.01
        allowed_fidelity = max_fidelity * (1 - self.fidelity_fluctuate)
        chosen_index = np.argmax(fid > allowed_fidelity)
        gf = fitter_list[chosen_index]
        readout_power_opt = power_list[chosen_index]
        window_opt = window_list[chosen_index]

        plt.figure(figsize=(8, 6))
        plt.title(f"{self.data_path}")
        plt.plot(power_list, fid, marker='.')
        plt.axvline(readout_power_opt, ls='--', color='black')
        plt.fill_between([
            min(power_list), max(power_list)], [max_fidelity, max_fidelity], [allowed_fidelity, allowed_fidelity], alpha=0.4)
        plt.ylim(0.5, 1)
        plt.xlabel('Readout Power')
        plt.ylabel('Readout fidelity')

        if savefig:
            plt.savefig(f"{savepath}/{self.data_path}.png")
        plt.show()

        x = gf.x
        popt0, popt1 = popt_list[chosen_index][0], popt_list[chosen_index][1]
        data0_fit = gaussian(x, *popt0)
        data1_fit = gaussian(x, *popt1)
        data0_hist, data1_hist = gf.data0_hist, gf.data1_hist
        threshold = gf.threshold
        g_data_weighted = data_list[chosen_index][0]
        e_data_weighted = data_list[chosen_index][1]

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{self.data_path}")

        ax[0].scatter(g_data_weighted.real, g_data_weighted.imag,
                      color='blue', label='Ground', marker=".")
        ax[0].scatter(e_data_weighted.real, e_data_weighted.imag,
                      color='red', label='Excited', marker=".")
        ax[0].legend()
        ax[0].set_aspect('equal')
        ax[0].set_xlabel('I')
        ax[0].set_ylabel('Q')

        ax[1].plot(x, data0_fit, '--', color='blue', lw=1)
        ax[1].plot(x, data0_hist, 'o', markersize=3,
                   color='blue', label='Ground')
        ax[1].plot(x, data1_fit, '--', color='red', lw=1)
        ax[1].plot(x, data1_hist, 'o', markersize=3,
                   color='red', label='Excited')
        ax[1].axvline(threshold, linestyle='--',
                      color='black', label='Threshold')
        ax[1].set_xlabel('Projected')
        ax[1].set_ylabel('Counts')
        ax[1].legend()

        cm = cm_list[chosen_index]
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
        plt.show()

        experiment_note = AttributeDict()
        experiment_note.cavity_readout_window_coefficient = window_opt
        experiment_note.cavity_readout_amplitude = readout_power_opt
        note.add_experiment_note(self.__class__.experiment_name,
                                 experiment_note, self.__class__.output_parameters)

    def report_stat(self):
        pass

    def report_visualize(self, dataset, note):
        pass
