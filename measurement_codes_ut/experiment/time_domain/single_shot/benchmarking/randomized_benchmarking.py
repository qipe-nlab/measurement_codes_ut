import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sequence_parser import Sequence
from sequence_parser.instruction import *
from measurement_codes_ut.experiment.time_domain.benchmarking.group import *
from sequence_parser.util.decompose import *
from plottr.data.datadict_storage import DataDict, DDH5Writer
from plottr.data.datadict_storage import datadict_from_hdf5
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm
import os


def exp_decay(x, a, b, p):
    y = a*p**x + b
    return y


def su2(self, matrix, target, rx90):
    """Execute the arbitrary single-qubit gate with virtual-Z decomposition (u3)
    Args:
        matrix (np.ndarray): matrix expression of the single-qubit gate
        target (int): index of the target qubit port
    """
    phases = matrix_to_su2(matrix)
    self.add(VirtualZ(phases[2]), target)
    self.call(rx90)
    self.add(VirtualZ(phases[1]), target)
    self.call(rx90)
    self.add(VirtualZ(phases[0]), target)


class RandomizedBenchmarking:
    experiment_name = "RandomizedBenchmarking"

    def __init__(
            self,
            num_shot=1000,
            repetitin_margin=100e3,
            random_circuit_count=20,
            min_length=1,
            max_length=1000,
            num_points=10,
            seed=0,
            interleaved=None,
    ):

        self.number_of_qubit = 1
        self.seed = seed
        self.interleaved = interleaved
        self.random_circuit_count = random_circuit_count
        sequence_length_list = np.logspace(np.log2(min_length), np.log2(
            max_length), num_points, base=2, dtype=int)
        sequence_list = []
        for length in sequence_length_list:
            sequence_list.append((length, random_circuit_count))
        self.sequence_list = sequence_list
        self.length_list = np.array(sequence_list).T[0].tolist()
        self.num_shot = num_shot
        self.repetition_margin = repetitin_margin
        self.group = CliffordGroup(1)
        Sequence.su2 = su2

    def execute(self, tdm, note, update_experiment=True, update_analyze=True):
        self.tdm = tdm
        if update_experiment:
            self.dataset = self.take_data(tdm, note)

        if update_analyze:
            if self.dataset is None:
                raise ValueError("Data is not taken yet.")
            self.analyze(self.dataset, note)

        return self.dataset

    def prepare(self, tdm, note):
        print("Preparing experiment...", end="")
        readout_port = tdm.port['readout'].port
        qubit_port = tdm.port['qubit'].port
        acq_port = tdm.acquire_port['readout_acquire']

        tdm.port['readout'].frequency = note.cavity_readout_frequency
        tdm.port['readout'].window = note.cavity_readout_window_coefficient
        if tdm.lo['qubit'] is None:
            qubit_port.if_freq = note.qubit_dressed_frequency/1e9
        else:
            tdm.port['qubit'].frequency = note.qubit_dressed_frequency
        tdm.set_acquisition_mode(averaging_waveform=True, averaging_shot=False)
        tdm.set_shots(self.num_shot)
        tdm.set_repetition_margin(self.repetition_margin)
        ports = [readout_port, qubit_port, acq_port]

        rx90 = Sequence(ports)

        with rx90.align(qubit_port, 'left'):
            rx90.add(Gaussian(amplitude=note.half_pi_pulse_power,
                              fwhm=note.half_pi_pulse_length/3,
                              duration=note.half_pi_pulse_length,
                              zero_end=True),
                     qubit_port)
            rx90.add(Deriviative(Gaussian(amplitude=1j*note.half_pi_pulse_power*note.half_pi_pulse_drag,
                                          fwhm=note.half_pi_pulse_length/3,
                                          duration=note.half_pi_pulse_length,
                                          zero_end=True)),
                     qubit_port)
        meas = Sequence(ports)
        meas.trigger(ports)
        meas.add(ResetPhase(phase=0), readout_port, copy=False)
        meas.add(Square(amplitude=note.cavity_readout_amplitude,
                 duration=note.readout_pulse_length), readout_port)
        meas.add(Acquire(duration=note.readout_pulse_length), acq_port)

        rb_seq = []
        for (length, random) in self.sequence_list:

            ## generate gate_array ##
            sequence_array = []
            if length == 0:
                for idx in range(random):
                    gate_array = []
                    sequence_array.append(gate_array)

            else:
                rand_gate_array = self.group.sample(
                    random*(length-1), seed=self.seed)
                rand_gate_array = rand_gate_array.reshape(
                    random, length-1, 2**self.number_of_qubit, 2**self.number_of_qubit)
                for rand_gates in rand_gate_array:
                    gate_array = []
                    gate = np.identity(2**self.number_of_qubit)
                    for rand in rand_gates:
                        gate_array.append(rand)
                        gate = rand@gate
                        if self.interleaved is not None:
                            gate = self.interleaved["gate"]@gate
                    gate_array.append(gate.T.conj())
                    sequence_array.append(gate_array)

            ## apply experiment ##
            for gate_array in sequence_array:
                seq = Sequence(ports)
                for pos, gate in enumerate(gate_array):
                    if int(np.log2(gate.shape[0])) == 1:
                        seq.su2(gate, target=qubit_port, rx90=rx90)
                    if self.interleaved is not None:
                        if pos != len(gate_array)-1:
                            seq.trigger(ports)
                            seq.call(self.interleaved["ansatz"])
                            seq.trigger(ports)
                seq.trigger(ports)
                seq.call(meas)
                rb_seq.append(seq)

        self.rb_seq = rb_seq
        tdm.sequence = rb_seq

        print("done")

    def take_data(self, tdm, note):
        self.prepare(tdm, note)
        tdm.variables = None

        dataset = tdm.take_data(dataset_name=self.__class__.experiment_name, as_complex=False, exp_file=__file__)

        return dataset

    def analyze(self, dataset, note):
        
        def get_pred(data, axis, border, which):
            component = np.dot(axis, data)
            if (component-border)*which > 0:
                return 0
            else:
                return 1
        response = dataset.data['readout_acquire']['values']
        len_data = len(response)

        component = np.zeros(len_data)
        for i in range(len_data):
            pred = np.mean([get_pred(data, note.measurement_axis, note.readout_assignment_border, note.readout_g_direction) for data in response[i]])
            component[i] = pred

        response_mean = 1 - component.reshape(
            len(self.length_list), self.random_circuit_count).mean(axis=1)
        response_std = component.reshape(
            len(self.length_list), self.random_circuit_count).std(axis=1)

        b0 = response_mean[-1]

        _x = np.array(self.length_list)
        _f = (response_mean - b0)
        _x = _x[np.where(_f > 0)]
        _f = _f[np.where(_f > 0)]
        _y = np.log(_f)
        # p0 = np.exp(np.mean(np.gradient(_y,_x)))

        _a, _b = np.polyfit(_x, _y, 1)
        p0 = np.exp(_a)
        a0 = np.exp(_b)

        popt, pcov = curve_fit(exp_decay, self.length_list, response_mean, p0=[
                               a0, b0, p0], maxfev=10000)

        length_fit = np.linspace(
            min(self.length_list), max(self.length_list), 10001)
        y_fit = exp_decay(length_fit, *popt)

        plt.figure()
        self.data_label = dataset.path.split("/")[-1][27:]
        plt.title(f"{self.data_label}")
        plt.errorbar(self.length_list, response_mean, yerr=response_std,
                     fmt='o', capsize=5, color='black', label='data')
        plt.plot(length_fit, y_fit, color='red',
                 label=f'Clifford fidelity: {(1+popt[2])/2:.5f} +/- {np.sqrt(pcov[2,2])/2:.5f}')
        plt.legend()
        plt.xlabel('Sequence length')
        plt.ylabel('Response')
        plt.show()
        print(f"Clifford fidelity: {(1+popt[2])/2} +/- {np.sqrt(pcov[2,2])/2}")
