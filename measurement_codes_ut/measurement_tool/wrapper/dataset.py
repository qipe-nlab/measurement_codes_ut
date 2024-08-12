from typing import Union, List, Optional

import numpy as np
import os
import shutil

from . import AttributeDict
from .. import Session
from plottr.data.datadict_storage import DataDict, DDH5Writer, datadict_from_hdf5


class Dataset(object):
    """Class for formatted dataset in data vault
    """

    def __init__(self, session) -> None:
        """Constructor
        """
        
        # if isinstance(session, Session):
        #     pass
        # else:
        #     raise TypeError("Unknown argument type")
        self.session = session


    def load(self, dataset_id: int, dataset_subpath: str="", log=True) -> None:
        """
        
        """
        save_path = self.session.save_path + dataset_subpath
        self.save_path = save_path
        data_all = []
        files = os.listdir(save_path)
        for date in files:
            date = date + "/"
            for f in os.listdir(save_path+date):
                data_all.append(save_path+date+f)

        if dataset_id > len(data_all)-1:
            raise ValueError(f"Dataset id.{dataset_id} is not in {save_path}.")
        else:
            # print(data_all)
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
            if log:
                print(f"Load dataset id.{dataset_id} from {save_path}.")

    def delete(self):
        assert hasattr(self, "number")
        data_all = []
        files = os.listdir(self.save_path)
        for date in files:
            date = date + "/"
            for f in os.listdir(self.save_path+date):
                data_all.append(self.save_path+date+f)
        
        if self.number != len(data_all) - 1:
            raise ValueError(f"Only the latest data can be deleted to avoid index mismatch.")
        else:
            # print(data_all)
            shutil.rmtree(data_all[self.number])
            
    
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
