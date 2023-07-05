import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sequence_parser import Sequence
from sequence_parser.instruction import *
from measurement_codes_ut.experiment.time_domain.benchmarking.group import *
from measurement_codes_ut.experiment.time_domain.benchmarking.randomized_benchmarking import RandomizedBenchmarking
from sequence_parser.util.decompose import *
from plottr.data.datadict_storage import DataDict, DDH5Writer
from plottr.data.datadict_storage import datadict_from_hdf5
from sklearn.decomposition import PCA
from tqdm import tqdm
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


class InterleavedRandomizedBenchmarking:
    experiment_name = "InterleavedRandomizedBenchmarking"

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
        assert interleaved is not None
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

        self.standard_rb = RandomizedBenchmarking(
            num_shot=num_shot,
            repetitin_margin=repetitin_margin,
            random_circuit_count=random_circuit_count,
            min_length=min_length,
            max_length=max_length,
            num_points=num_points,
            seed=seed,
            interleaved=None,)
        self.interleaved_rb = RandomizedBenchmarking(
            num_shot=num_shot,
            repetitin_margin=repetitin_margin,
            random_circuit_count=random_circuit_count,
            min_length=min_length,
            max_length=max_length,
            num_points=num_points,
            seed=seed,
            interleaved=interleaved)

    def execute(self, tdm, note, update_experiment=True, update_analyze=True):
        if update_experiment:
            self.dataset_s = self.standard_rb.take_data(tdm, note)
            self.dataset_i = self.interleaved_rb.take_data(tdm, note)

        if update_analyze:
            if self.dataset_i is None:
                raise ValueError("Data is not taken yet.")
            self.analyze(self.dataset_s, self.dataset_i)

        return [self.dataset_s, self.dataset_i]

    def prepare(self, tdm, note):
        print("Preparing experiment...", end="")
        readout_port = tdm.port['readout'].port
        qubit_port = tdm.port['qubit'].port
        acq_port = tdm.acquire_port['readout_acquire']

        tdm.port['readout'].frequency = note.cavity_readout_frequency
        tdm.port['readout'].window = note.cavity_readout_window_coefficient
        tdm.port['qubit'].frequency = note.qubit_dressed_frequency
        tdm.set_acquisition_mode(averaging_waveform=True, averaging_shot=True)
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
                 duration=2000), readout_port)
        meas.add(Acquire(duration=2000), acq_port)

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
            rb_seq_in = []
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
                rb_seq_in.append(seq)
            rb_seq.append(rb_seq_in)

        self.rb_seq = rb_seq
        print("done")

    def take_data(self, tdm, note):
        self.prepare(tdm, note)
        data = DataDict(
            circuit_length=dict(unit=""),
            circuit_index=dict(unit=""),
            s11=dict(axes=["circuit_length", "circuit_index"]),
        )
        data.validate()

        with DDH5Writer(data, tdm.save_path, name=self.__class__.experiment_name) as writer:
            tdm.prepare_experiment(writer, __file__)
            for i, rb_seq_in in enumerate(tqdm(self.rb_seq)):
                for j, seq in enumerate((rb_seq_in)):
                    raw_data = tdm.run(
                        seq, as_complex=False)
                    writer.add_data(
                        circuit_length=self.length_list[i],
                        circuit_index=j,
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

    def analyze(self, dataset_s, dataset_i):
        dataset_list = [dataset_s, dataset_i]
        popt_list = []
        pcov_list = []
        y_fit_list = []
        res_mean = []
        res_std = []
        for dataset in dataset_list:
            pca = PCA()
            response = pca.fit_transform(dataset['s11']['values'])[:, 0]
            response_mean = response.reshape(
                len(self.length_list), self.random_circuit_count).mean(axis=1)
            response_std = response.reshape(
                len(self.length_list), self.random_circuit_count).std(axis=1)

            if response_mean[0] < response_mean[-1]:
                response_mean *= -1
            res_mean.append(response_mean)
            res_std.append(response_std)
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

            popt_list.append(popt)
            pcov_list.append(pcov)
            y_fit_list.append(y_fit)

        new_p = popt_list[1][2] / popt_list[0][2]
        fg = (1 + (2**self.number_of_qubit - 1)*new_p) / \
            (2**self.number_of_qubit)
        colors = ['red', 'blue']
        labels = ['Reference', 'Interleaved']
        plt.figure(figsize=(8, 6))
        plt.title(f'{self.interleaved_rb.data_path}')
        for i in range(2):
            plt.errorbar(self.length_list, res_mean[i], yerr=res_std[i], label=f"{labels[i]} data",
                         fmt='o', capsize=5, color='black')
            if i == 0:
                plt.plot(length_fit, y_fit_list[i], color=colors[i],
                         label=f'Reference fit')
            if i == 1:
                plt.plot(length_fit, y_fit_list[i], color=colors[i],
                         label=f'Gate fidelity: {fg:.5f}')
        plt.legend()
        plt.xlabel('Sequence length')
        plt.ylabel('Response')
        plt.show()
        print(f'Gate fidelity: {fg}')
