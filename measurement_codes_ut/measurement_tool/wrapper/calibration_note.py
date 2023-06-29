from typing import Any, List
from datetime import datetime
from logging import getLogger
import os
import numpy as np
import json
import glob

from measurement_codes_ut.measurement_tool.wrapper import AttributeDict

logger = getLogger(__name__)


class CalibrationNote(object):
    """This class manages process of calibration sequence.

    There are three parameter fields, named "globals", "initials", and "notes".
    "globals" field is accessible from every experiment. The parameters in "globals" can be overwritten by successive experiment.
    "initials"  field is a list of initial parameters. This values are used when a parameter is required by experiment but not in globals.
    "notes" is a list of dictionary and each dictionary represents an experiment.

    In each experiement, the intended steps are
    1. call "set_initial_value" for giving initially required values for experiment, such as estimated frequency of qubit.
    2. call "get_calibration_parameters" with list of required parameters. this returns AttributeDict with required parameters.
    3. do experiment and analysis
    4. call "update_experiment_note" to store every analysis results and dataset path, and to update globals field for successive experiments.
    6. call "get_experiment_note" to retrieve past experimental results
    """

    def __init__(self) -> None:
        """Constructor of calibration note
        """
        self.notes = []
        self.globals = AttributeDict()
        self.initials = AttributeDict()

    def set_initial_value(self, value_name: str, value: Any) -> None:
        """Set values to initial field

        Args:
            value_name (str): name of parameter
            value (LabradValue): value of parameter
        """
        self.initials[value_name] = AttributeDict()
        self.initials[value_name].value = value
        self.initials[value_name].timestamp = self._get_timestamp()

    def get_calibration_parameters(self, experiment_name: str, value_names: List[str]) -> AttributeDict:
        """Get calibration note with a given value names

        Args:
            experiment_name (str): name of experiment
            value_names (list of str): list of required parameter names for experiment

        Returns:
            AttributeDict : dictionary of required values

        Raises:
            ValueError: required value is not found in globals nor initials.
        """
        if isinstance(value_names, str):
            value_names = [value_names]
        note = AttributeDict()
        for value_name in value_names:
            # if found in global field, fetch it
            if value_name in self.globals.keys():
                note[value_name] = self.globals[value_name].value
            # if not found in global, fetch from initials
            elif value_name in self.initials.keys():
                note[value_name] = self.initials[value_name].value
                self.globals[value_name] = AttributeDict()
                self.globals[value_name].update_history = [
                    ("InitialParameter", self.initials[value_name].value, self._get_timestamp())]
                self.globals[value_name].value = self.initials[value_name].value
            # if not found, raise error
            else:
                raise ValueError(
                    "Key: {} is required for experiment but not found in known values".format(value_name))
        return note

    def add_experiment_note(self, experiment_name: str, experiment_note: dict, commit_value_names: List[str]) -> None:
        """Add experiment note

        Args:
            experiment_name (str): name of experiment
            experiment_note (AttributeDict): dictionary of all the obtained parameters
            commit_value_names (list of str): list of names to commit values to global
            dataset (Dataset): dataset obtained in experiment. If None, dataset is assumed to be lost
        """
        # add note
        new_note = AttributeDict()
        new_note.experiment_name = experiment_name
        new_note.note = experiment_note
        new_note.timestamp = self._get_timestamp()
        new_note.commit_value_names = commit_value_names
        # if hasattr(dataset, "dataset_path"):
        #     new_note.dataset_path = dataset.dataset_path
        #     new_note.dataset_number = dataset.dataset_number
        #     new_note.dataset_name = dataset.dataset_name
        # else:
        #     new_note.dataset_path = None
        #     new_note.dataset_number = None
        #     new_note.dataset_name = None
        self.notes.append(new_note)

        # update global fields according to commit_value_names
        for value_name in commit_value_names:
            if value_name not in experiment_note.keys():
                raise ValueError(
                    "Value to be commited {} is not in experiment results".format(value_name))

            # if value is not in globals, create new key
            if value_name not in self.globals.keys():
                self.globals[value_name] = AttributeDict()
                self.globals[value_name].update_history = []
            value = experiment_note[value_name]

            self.globals[value_name].value = value
            timestamp = self._get_timestamp()
            self.globals[value_name].timestamp = timestamp
            self.globals[value_name].update_history.append(
                (experiment_name, value, timestamp))

    def get_experiment_note(self, experiment_name: str, experiment_index: int = -1) -> AttributeDict:
        """Retrieve experiment note

        Args:
            experiment_name (str): name of experiment
            experiment_index (int): index of experiemnt, default to -1 (retrieve the latest results)

        Returns:
            AttributeDict: obtained note

        Raises:
            ValueError : specified experiment is not done
        """
        note_list = [
            note for note in self.notes if note.experiment_name == experiment_name]
        try:
            return note_list[experiment_index]
        except IndexError:
            raise ValueError(
                "Experiment {} is done {}-times, but try to refer {}-th".format(
                    experiment_name, len(note_list), experiment_index))

    def show_recent_experiment(self, tail: int = 10) -> List[tuple]:
        """Get list of recent experiment names and timestamps

        Args:
            tail (int, optional): Number of experiment to fetch. Defaults to 10.

        Returns:
            List[tuple]: list of tuple (timestamp, experiment_name)
        """
        tail = min(tail, len(self.notes))
        result = [(note.timestamp, note.experiment_name)
                  for note in self.notes[-tail:]]
        return result

    def remove_last_experiment_note(self) -> None:
        """Remove the recent experiment note

        Raises:
            ValueError: There is no experiment note to remove
        """
        if len(self.notes) == 0:
            raise ValueError("Cannot rollback")
        poped_note = self.notes.pop()
        commit_value_names = poped_note.commit_value_names
        for commit_value_name in commit_value_names:
            assert(commit_value_name in self.globals)
            last_update = self.globals[commit_value_name].update_history.pop()
            logger.info("pop {} <- {}".format(last_update, commit_value_name))
            if len(self.globals[commit_value_name].update_history) == 0 or \
                    len(self.globals[commit_value_name].update_history[-1]) == 2:
                self.globals.pop(commit_value_name)
            else:
                next_last_update = self.globals[commit_value_name].update_history[-1]
                self.globals[commit_value_name].value = next_last_update[1]
                self.globals[commit_value_name].timestamp = next_last_update[2]

    def to_attribute_dict(self) -> AttributeDict:
        """Serialize this class to AttributeDict

        Returns:
            AttributeDict: converted dict
        """
        new_note = AttributeDict()
        new_note["experiment"] = {}
        for index, note in enumerate(self.notes):
            identifier = "{:0>8}_{}".format(index, note.experiment_name)
            new_note["experiment"][identifier] = note
        new_note["global"] = self.globals
        new_note["initial"] = self.initials
        return new_note

    def from_attribute_dict(self, note: dict) -> None:
        """deserialize this class from AttributeDict

        Args:
            note (dict): serialized dictionary
        """
        for key in note.keys():
            if key == "global":
                self.globals = note["global"]
            elif key == "initial":
                self.initials = note["initial"]
            elif key == "experiment":
                sorted_identifiers = sorted(list(note[key].keys()))
                for identifier in sorted_identifiers:
                    self.notes.append(note[key][identifier])


    def to_json(self, path:str, name):
        os.makedirs(path, exist_ok=True)
        glob = self.globals
        mydic = {}
        for key in glob:
            value = glob[key].value
            valtype = ""
            if isinstance(value, np.ndarray) and any(np.iscomplex(value)):
                mydic[key] = {
                    "value": [value.real.tolist(), value.imag.tolist()],
                    "unit": "",
                    "type": "complex_array",
                }
            elif isinstance(value, np.ndarray) and (not any(np.iscomplex(value))):
                mydic[key] = {
                    "value": value.tolist(),
                    "unit": "",
                    "type": "real_array",
                }
            elif isinstance(value, float):
                mydic[key] = {
                    "value": value,
                    "unit": "",
                    "type": "float_value",
                }
            elif isinstance(value, int):
                mydic[key] = {
                    "value": value,
                    "unit": "",
                    "type": "integer_value",
                }
            else:
                raise ValueError("invalid type")
        doc = mydic
        jsonname = f"{path}/{name}.json"
        with open(jsonname, "w") as fout:
            json.dump(mydic, fout)
        print(f"output {jsonname}")

    def from_json(self, path, name):
        jsonname = f"{path}/{name}.json"
        with open(jsonname) as fin:
            mydic = json.load(fin)
        qubit_dic = {}
        for key, val in mydic.items():
            assert("type" in val)
            if val["type"] == "float_value":
                qubit_dic[key] = val["value"]
            elif val["type"] == "integer_value":
                qubit_dic[key] = val["value"]
            elif val["type"] == "complex_array":
                arr = np.array(val["value"])
                qubit_dic[key] = arr[0] + 1.j * arr[1]
            elif val["type"] == "real_array":
                arr = np.array(val["value"])
                qubit_dic[key] = arr
            else:
                raise ValueError("invalid type")
        self.add_experiment_note("calib_load", qubit_dic, qubit_dic.keys())

    def _get_timestamp(self) -> str:
        """Get time stamp

        Returns:
            str: string representation of timestamp
        """
        return datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # override
    def __getattr__(self, key: str) -> Any:
        """If get attribution access, return value in globals field

        Args:
            key (str): value to get
        """
        return self.globals[key].value

    def __str__(self) -> str:
        """String representation of this class

        Returns:
            str: string representation
        """
        s = ""
        for key in self.globals.keys():
            s += "{} : {}\n".format(key, self.globals[key].value)
        return s
