from typing import Union, List, Optional

import numpy as np
import os

from measurement_codes_ut.measurement_tool.wrapper import AttributeDict
from measurement_codes_ut.measurement_tool import Session
from plottr.data.datadict_storage import DataDict, DDH5Writer, datadict_from_hdf5


class Dataset(object):
    """Class for formatted dataset in data vault

    When data is taken with this library, dataset has information of sweeps for regenerate the previous measurements.

    This class maanges measured dataset for multiple inepdendents and dependents.
    Raw-data is stored in self.data_matrix as two dimensional array.
    The first dimension is an index of data sample.
    The second dimension is an index of independents and dependents in sample.
    This class assumes that measuremenets are done for every parameter set of sweeped independent values.
    """

    def __init__(self, session) -> None:
        """Constructor

        Args:
            session (Session): session of this measurement

        """
        
        # if isinstance(session, Session):
        #     pass
        # else:
        #     raise TypeError("Unknown argument type")



    def load(self, dataset_id: int, dataset_path: str="", log=True) -> None:
        """Load dataset

        Args:

        Raises:
        """
        save_path = dataset_path
        data_all = []
        files = os.listdir(save_path)
        for date in files:
            date = date + "/"
            for f in os.listdir(save_path+date):
                data_all.append(save_path+date+f)

        if dataset_id > len(data_all)-1:
            raise ValueError(f"Dataset id.{dataset_id} is larger than the number of experiments in {save_path}.")
        else:
            # print(data_all)
            if log:
                print(f"Load dataset id.{dataset_id} from {save_path}.")
            self.path = data_all[dataset_id]
            self.data = datadict_from_hdf5(self.path+"/data")
            self.name = data_all[dataset_id].split("/")[-1][33:]
            self.number = dataset_id
            lines = []
            s= ''
            with open(self.path+"/wiring.md", encoding='utf-8') as f:
                lines = f.readlines()   

            s = ''.join(lines) 
            self.wiring_info = s

    def __repr__(self) -> str:
        """Return string representation of Dataset

        Returns:
            str: string representation
        """
        repr_str = ""
        repr_str += "dataset_number = {:>05}\n".format(self.number)
        repr_str += "dataset_name = \"{}\"\n".format(self.name)
        repr_str += "dataset_path = {}\n".format(self.path)

        return repr_str
