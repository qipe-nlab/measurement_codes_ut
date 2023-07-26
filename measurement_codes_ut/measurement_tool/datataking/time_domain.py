import numpy as np
from logging import getLogger
import pathlib
import numpy as np
import itertools
import copy
import os
import qcodes as qc
from qcodes_drivers.E82x7 import E82x7
from qcodes_drivers.N51x1 import N51x1
from qcodes_drivers.HVI_Trigger import HVI_Trigger
from qcodes_drivers.iq_corrector import IQCorrector
from qcodes_drivers.M3102A import M3102A
from qcodes_drivers.M3202A import M3202A
from sequence_parser import Sequence, Variable, Variables
from tqdm.notebook import tqdm
from sequence_parser.iq_port import IQPort
from qcodes.instrument_drivers.yokogawa.GS200 import GS200
from .instrument_manager import InstrumentManagerBase
from plottr.data.datadict_storage import DataDict, DDH5Writer, datadict_from_hdf5
from .port_manager import PortManager


import matplotlib.pyplot as plt

from measurement_codes_ut.measurement_tool import Session
from measurement_codes_ut.measurement_tool.wrapper import Dataset

logger = getLogger(__name__)


class TimeDomainInstrumentManager(InstrumentManagerBase):
    """Insturment management class for timedomain measurement"""

    def __init__(self, session: Session, trigger_address: str, save_path) -> None:
        """Constructor of time domain measurement

        Args:
            session (Session): session of measurement
            config_name (str): default config name of instruments
        """
        # print("Creating a new insturment management class for timedomain measurement...", end="")
        super().__init__(session, trigger_address, save_path)
        self.sequence = None
        self.variables = None

    def take_data(self, 
                  dataset_name: str, 
                  dataset_subpath: str = "TD", 
                  sweep_axis: list = None, 
                  as_complex: bool = True, 
                  exp_file: str = None,
                  verbose: bool = True):
        """take data

        Args:
            dataset_name (str): dataset name
            dataset_subpath (str, optional): data is saved to specified subpath of datavault. Defaults to "".
            sweep_axis (Optional[list], optional): assignment of independent sweep parameters to axis. If None,
                i-th independent sweep parameter is assigned to i-th loop axis. Defaults to None.
            as_complex (bool, optional): If true, obtain data as complex data. If false, obtain data as I, Q data. Defaults to True.
            exp_file (str, optional): File name in which experiment is executed. Defaults to None, which saves no backup file except for this .py.
            verbose (bool, optional): If true, show tqdm progress bar. Defaults to True.

        Returns:
            Dataset: taken dataset
        """
        seq = self.sequence
        variables = self.variables

        flag = False
        for key, port in self.port.items():
            if isinstance(port.frequency, np.ndarray) or isinstance(port.frequency, list):
                flag = True
        if flag:
            dataset = self.take_data_lo_sweep(
                dataset_name, 
                dataset_subpath, 
                sweep_axis, 
                as_complex, 
                exp_file,
                verbose)
            
        else:

            # Re-construct variablse.command_list        
            if variables is not None:
                variables = self.set_variables(variables, sweep_axis)
                self.variables = variables
                var_dict = {key:dict(unit=value[0].unit) for key, value in zip(variables.variable_name_list, variables.variable_list)}
                var = True
            else:
                var_dict = {}
                var = False
            for port in seq.port_list:
                if "acquire" in port.name:
                    var_dict[port.name] = dict(axes=list(var_dict.keys()))

            data = DataDict(**var_dict)
            data.validate()

            
            save_path = self.save_path + dataset_subpath + "/"
            os.makedirs(save_path, exist_ok=True)
            
            files = os.listdir(save_path)
            file_date_all = [f + "/" for f in files]
            num_files = 0
            for date in file_date_all:
                num_files += len(os.listdir(save_path+date))

            exp_name = f"{num_files:05}-{dataset_name}"

            with DDH5Writer(data, save_path, name=exp_name) as writer:
                self.prepare_experiment(writer, exp_file)
                if var:
                    for update_command in (tqdm(variables.update_command_list) if verbose else variables.update_command_list):
                        seq.update_variables(update_command)
                        # seq.draw()
                        raw_data = self.run(seq, as_complex=as_complex)
                        write_dict = {key:seq.variable_dict[key][0].value for key in variables.variable_name_list}
                        for port in seq.port_list:
                            if "acquire" in port.name:
                                write_dict[port.name] = raw_data[str(port.name).replace("_acquire", "")]
                        writer.add_data(**write_dict)
                else:
                    raw_data = self.run(seq, as_complex=as_complex)
                    write_dict = {}
                    for port in seq.port_list:
                        if "acquire" in port.name:
                            write_dict[port.name] = raw_data[str(port.name).replace("_acquire", "")]
                    writer.add_data(**write_dict)
                
            print(f"Experiment id. {num_files} completed.")

            dataset = Dataset(self.session)
            dataset.load(num_files, save_path)
            data = dataset.data

        return dataset
    
    def take_data_lo_sweep(self, 
                  dataset_name: str, 
                  dataset_subpath: str = "TD", 
                  sweep_axis: list = None, 
                  as_complex: bool = True, 
                  exp_file: str = None,
                  verbose: bool = True):
        """take data

        Args:
            dataset_name (str): dataset name
            dataset_subpath (str, optional): data is saved to specified subpath of datavault. Defaults to "".
            sweep_axis (Optional[list], optional): assignment of independent sweep parameters to axis. If None,
                i-th independent sweep parameter is assigned to i-th loop axis. Defaults to None.
            as_complex (bool, optional): If true, obtain data as complex data. If false, obtain data as I, Q data. Defaults to True.
            exp_file (str, optional): File name in which experiment is executed. Defaults to None, which saves no backup file except for this .py.
            verbose (bool, optional): If true, show tqdm progress bar. Defaults to True.

        Returns:
            Dataset: taken data
        """
        lo_sweep_dict = {}
        for key, port in self.port.items():
            if isinstance(port.frequency, np.ndarray) or isinstance(port.frequency, list):
                lo_sweep_dict[key] = copy.copy(port.frequency)
        if len(lo_sweep_dict) >= 2:
            raise ValueError("Cannot sweep more than 1 LO frequency at the same time. {} are set as sweep parameters.".format(list(lo_sweep_dict.keys())))
        
        seq = self.sequence
        variables = self.variables
        if variables is not None:
            variables = self.set_variables(variables, sweep_axis)
            self.variables = variables
            var_dict = {key:dict(unit=value[0].unit) for key, value in zip(variables.variable_name_list, variables.variable_list)}
            var_other = True
        else:
            var_dict = {}
            var_other = False

        lo_sweep_key = list(lo_sweep_dict.keys())[0]
        lo_sweep_value = list(lo_sweep_dict.values())[0]

        
        var_dict[lo_sweep_key+"_LO_frequency"] = dict(unit='Hz')
        for port in seq.port_list:
            if "acquire" in port.name:
                var_dict[port.name] = dict(axes=list(var_dict.keys()))

        data = DataDict(**var_dict)
        data.validate()

        
            
        save_path = self.save_path + dataset_subpath + "/"
        os.makedirs(save_path, exist_ok=True)
        files = os.listdir(save_path)
        date = files[-1] + '/'
        num_files = len(os.listdir(save_path+date))

        exp_name = f"{num_files:05}-{dataset_name}"


        with DDH5Writer(data, save_path, name=exp_name) as writer:
            self.prepare_experiment(writer, exp_file)
            if var_other:
                for update_command in (tqdm(variables.update_command_list) if verbose else variables.update_command_list):
                    seq.update_variables(update_command)
                    
                    for lo in (tqdm(lo_sweep_value, leave=False) if verbose else lo_sweep_value):
                        self.port[lo_sweep_key].set_frequency(lo)
                        # seq.draw()
                        raw_data = self.run(seq, as_complex=as_complex)
                        write_dict = {key:seq.variable_dict[key][0].value for key in variables.variable_name_list}
                        write_dict[lo_sweep_key+"_LO_frequency"] = lo
                        for port in seq.port_list:
                            if "acquire" in port.name:
                                write_dict[port.name] = raw_data[str(port.name).replace("_acquire", "")]
                        writer.add_data(**write_dict)

            else:
                for lo in (tqdm(lo_sweep_value) if verbose else lo_sweep_value):
                    self.port[lo_sweep_key].set_frequency(lo)
                    raw_data = self.run(seq, as_complex=as_complex)
                    write_dict = {}
                    write_dict[lo_sweep_key+"_LO_frequency"] = lo
                    for port in seq.port_list:
                        if "acquire" in port.name:
                            write_dict[port.name] = raw_data[str(port.name).replace("_acquire", "")]
                    writer.add_data(**write_dict)

                
        print(f"Experiment id. {num_files} completed.")

        dataset = Dataset(self.session)
        dataset.load(num_files, save_path)

        return dataset


    def set_variables(self, variables, sweep_axis):
        if sweep_axis is not None:

            original_sweep_dims = [size for size in variables.variable_size_list]
            original_sweep_labels = [name for name in variables.variable_name_list]
            axis_count = len(list(set(sweep_axis)))
            if len(sweep_axis) != len(original_sweep_dims):
                raise ValueError("len(sweep_axis) must be equal to the number of sweep parameters. {} sweep parameters found.".format(len(original_sweep_dims)))
            if min(sweep_axis) != 0:
                raise ValueError("axis index must be starts from zero. However, there is no 0-th index or there is negative index in sweep_axis.")
            if max(sweep_axis) + 1 != axis_count:
                raise ValueError("There is skipped index in sweep_axis. max(sweep_axis)+1 must be equal to unique integer numbers in sweep_axis.")

            # gather information of sweep parameters belonging to i-th axis index.
            axis_groups = [[] for _ in range(axis_count)]
            for sweep_parameter_index, axis_index in enumerate(sweep_axis):
                assert(0 <= axis_index and axis_index < len(axis_groups))
                sweep_parameter_info = {
                    "index": sweep_parameter_index,
                    "dim": original_sweep_dims[sweep_parameter_index],
                    "label": original_sweep_labels[sweep_parameter_index]
                }
                axis_groups[axis_index].append(sweep_parameter_info)

            # check sweep parameters belonging to each axis have the same dimensions
            for axis_index in range(axis_count):
                assert(len(axis_groups[axis_index]) > 0)
                dims = [item["dim"] for item in axis_groups[axis_index]]
                labels = [item["label"] for item in axis_groups[axis_index]]
                check_flag = all([dim == dims[0] for dim in dims])
                if not check_flag:
                    raise ValueError("{}-th axis have parameters {} which have different dimension lists {}.".format(axis_index, labels, dims))

            axis_dims = [group[0]["dim"] for group in axis_groups]
            shape_axis_groups = tuple([len(l) for l in axis_groups])

            def dim2index(tuple_input, reference_tuple):
                tuple_list = list(itertools.product(*[range(x) for x in tuple_input]))
                tuple_list_all = []
                for t in tuple_list:
                    result = []
                    for inp, ref in zip(t, reference_tuple):
                        for _ in range(ref):
                            result.append(inp)
                    tuple_list_all.append(tuple(result))
                return tuple_list_all
            
            def reorder_list(lst, reference):
                new_mapping = {key:reference[i] for i, key in enumerate(lst)}
                
                sorted_list = []
                for _ in range(max(reference)+1):
                    elem = [k for k, v in new_mapping.items() if v == _]
                    sorted_list += elem
                
                return sorted_list
            
            var_name_list = reorder_list(variables.variable_name_list, sweep_axis)
            var_list = [tuple(i) for i in variables.variable_list]
            var_list = reorder_list(var_list, sweep_axis)
            var_list = [list(i) for i in var_list]
            var_size_list = reorder_list(variables.variable_size_list, sweep_axis)

            variables.variable_name_list = var_name_list
            variables.variable_list = var_list
            variables.variable_size_list = var_size_list
            
            t = tuple(axis_dims)
            sweep_index = dim2index(t, shape_axis_groups)

            variables.update_command_list = []
            tmp_var = dict(zip(variables.variable_name_list, [None]*len(variables.variable_name_list)))
            for tmp_index in sweep_index:
                update_command = {}
                for variable, idx in zip(variables.variable_list, tmp_index):
                    for var in variable:
                        if tmp_var[var.name] != var.value_array[idx]:
                            tmp_var[var.name] = var.value_array[idx]
                            update_command[var.name] = idx
                variables.update_command_list.append(update_command)

        return variables


        
        
    def set_wiring_note(self, wiring_info):
        self.wiring_info = wiring_info

    def load_sequence(self, sequence: Sequence, cycles: int, noise_variance=0):
        rng = np.random.default_rng()
        sequence.compile()
        self.seq_len = len(list(self.port.values())[0].port.waveform)
        for awg in self.awg.values():
            awg.stop_all()
            awg.flush_waveform()

        waveform_awg = {
            key: np.zeros(self.seq_len, dtype=complex) for key in self.awg_ch.keys()}
        waveform_idx = 0
        for key, port_manager in self.port.items():
            # awg_info = self.awg_info[key]
            awg_index, awg_ch = self.awg_ch[key]
            awg = self.awg[awg_index]
            port = port_manager.port
            port_manager.update_frequency()
            if "readout" in port.name:
                del waveform_awg[key]
                dig_ch = self.digitizer_ch[key]
                waveform = port.waveform
                # plt.plot(waveform.real)
                if noise_variance > 0:
                    waveform += rng.normal(scale=np.sqrt(noise_variance),
                                           size=len(waveform))

                if isinstance(awg_ch, tuple):
                    try:
                        waveform_corrected = self.IQ_corrector[port.name].correct(
                            waveform)
                    except AttributeError:
                        waveform_corrected = [waveform.real, waveform.imag]

                    for _ in range(2):  # i or q
                        wave = waveform_corrected[_]
                        awg.load_waveform(wave, waveform_idx,
                                          append_zeros=True)
                        awg_ch[_].queue_waveform(
                            waveform_idx, trigger="software/hvi", cycles=cycles)
                        waveform_idx += 1
                else:
                    awg.load_waveform(
                        waveform.real, waveform_idx, append_zeros=True)
                    awg_ch.queue_waveform(
                        waveform_idx, trigger="software/hvi", cycles=cycles)
                    waveform_idx += 1

                dig_ch.cycles(cycles)
                acq_port = self.acquire_port[key+"_acquire"]
                if len(acq_port.measurement_windows) == 0:
                    acquire_start = 0
                else:
                    acquire_start = int(acq_port.measurement_windows[0][0])
                    acquire_end = int(acq_port.measurement_windows[-1][1])
                    assert acquire_start % dig_ch.sampling_interval() == 0
                    assert acquire_end % dig_ch.sampling_interval() == 0
                acquire_len = (acquire_end - acquire_start) // (20*dig_ch.sampling_interval()) * 20
                dig_ch.points_per_cycle(acquire_len)
                dig_ch.delay(acquire_start //
                             dig_ch.sampling_interval())

            else:
                waveform = port.waveform
                waveform_awg[key] += waveform

        for key, waveform in waveform_awg.items():
            awg_index, awg_ch = self.awg_ch[key]
            awg = self.awg[awg_index]
            if isinstance(awg_ch, tuple):
                try:
                    waveform_corrected = self.IQ_corrector[key].correct(
                        waveform)
                except AttributeError:
                    i = waveform.real
                    q = waveform.imag
                    waveform_corrected = i, q

                for _ in range(2):  # i or q
                    wave = waveform_corrected[_]
                    awg.load_waveform(wave, waveform_idx, append_zeros=True)
                    awg_ch[_].queue_waveform(
                        waveform_idx, trigger="software/hvi", cycles=cycles)
                    waveform_idx += 1
            else:
                awg.load_waveform(
                    waveform.real, waveform_idx, append_zeros=True)
                awg_ch.queue_waveform(
                    waveform_idx, trigger="software/hvi", cycles=cycles)
                waveform_idx += 1

    def run(self, sequence: Sequence, demodulate=True, averaging_shot=None, averaging_waveform=None, as_complex=True):

        self.load_sequence(sequence, self.num_shot)

        if averaging_shot is None:
            averaging_shot = self.averaging_shot
        if averaging_waveform is None:
            averaging_waveform = self.averaging_waveform

        self.hvi_trigger.digitizer_delay(self.acquisition_delay)  # ns
        self.hvi_trigger.trigger_period(
            int((self.repetition_margin + self.seq_len)/10+1)*10)  # ns

        # self.set_acquisition_mode(averaging_shot, averaging_waveform)
        try:
            for name, cur in self.current_source.items():
                cur.output('on')
            for name, lo in self.lo.items():
                try:
                    lo.output(True)
                except:
                    lo.on()
            for awg_index, awg_ch in self.awg_ch.values():
                awg = self.awg[awg_index]
                if isinstance(awg_ch, tuple):
                    for _ in range(2):
                        awg_ch[_].start()
                else:
                    awg_ch.start()
            for dig in self.digitizer_ch.values():
                dig.start()
            self.hvi_trigger.output(True)
            data = {}
            for key, dig in self.digitizer_ch.items():
                d = dig.read()
                data[key] = d * dig.voltage_step()

            for awg in self.awg.values():
                awg.stop_all()
            for dig in self.digitizer_ch.values():
                dig.stop()
            self.hvi_trigger.output(False)

            if averaging_shot:
                data_return = {}
                for key, d in data.items():
                    data_return[key] = d.mean(axis=0)
                # assuming only 1 readout, should be extended to more than 1 readout
                # self.data = data_return
                if demodulate:
                    data_return = self.demodulate(
                        data_return, averaging_waveform, as_complex)

                return data_return
            else:
                data_return = {}
                for key, d in data.items():
                    data_return[key] = d
                # assuming only 1 readout, should be extended to more than 1 readout
                if demodulate:
                    data_return = self.demodulate(
                        data_return, averaging_waveform, as_complex)
                return data_return

        finally:
            self.stop()

    def stop(self):
        self.hvi_trigger.output(False)
        for awg in self.awg.values():
            awg.stop_all()
        for dig in self.digitizer_ch.values():
            dig.stop()
        for name, lo in self.lo.items():
            try:
                lo.output(False)
            except:
                lo.on()
        for name, cur in self.current_source.items():
            cur.output('off')

    def demodulate(self, data_all, averaging_waveform=True, as_complex=True):
        data_demod = {}
        for key, data in data_all.items():
            t = np.arange(data.shape[-1]) * \
                self.digitizer_ch[key].sampling_interval() * 1e-9
            if averaging_waveform:
                if self.port[key].window is None:
                    data_demod[key] = (
                        data * np.exp(2j * np.pi * self.port[key].port.if_freq*1e9 * t)).mean(axis=-1)
                else:
                    d = data * np.exp(2j * np.pi *
                                      self.port[key].port.if_freq*1e9 * t)
                    data_demod[key] = np.dot(d, self.port[key].window)
                if as_complex == False:
                    data_demod[key] = np.stack(
                        (data_demod[key].real, data_demod[key].imag), axis=-1)
            else:
                data_demod[key] = (
                    data * np.exp(2j * np.pi * self.port[key].port.if_freq*1e9 * t))
                if as_complex == False:
                    data_demod[key] = np.stack(
                        (data_demod[key].real, data_demod[key].imag), axis=-1)
        return data_demod

    def set_acquisition_mode(self, averaging_shot, averaging_waveform):
        self.averaging_shot = averaging_shot
        self.averaging_waveform = averaging_waveform

    def set_average_window_coefficients(self, port_name, window):
        self.port[port_name].window = window

    def set_repetition_margin(self, time):
        self.repetition_margin = int(time)
        # self.hvi_trigger.trigger_period(int(time))

    def set_acquisition_delay(self, delay):
        self.acquisition_delay = int(delay//10+1) * 10
        # self.hvi_trigger.digitizer_delay(int(delay//10+1) * 10)

    def set_shots(self, num_shot):
        self.num_shot = num_shot

    def prepare_experiment(self, writer, exp_file):
        writer.add_tag(self.tags)
        if exp_file is None:
            writer.backup_file([__file__])
        else:
            writer.backup_file([exp_file, __file__])
        writer.save_text("wiring.md", self.wiring_info)
        writer.save_dict("station_snapshot.json", self.station.snapshot())

    def show_sweep_plan(self):
        if self.variables is None:
            self.sequence.draw()
        else:
            update_command = self.variables.update_command_list[-1]
            self.sequence.update_variables(update_command)
            self.sequence.draw()