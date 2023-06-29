
import numpy as np
from measurement_tool.units import LabradValue

class TableHelper(object):
    """Helper class for showing formatted list of experimental data with units and errors.

    This class helps to show experimental data with units and errors for easy-to-see format.
    This class support unit conversion, significant-digit calibration, and appropriate padding and aligning.
    """

    def __init__(self):
        """Initializer of this class
        """
        self.data_list = []

    def add(self, name, value, unit, error=None):
        """Add new value to table

        Args:
            name (str): Name of parameter
            value (float or LabradValue): Value of parameter
            unit (str or LabradUnit): Unit of parameter
            error (float or LabradValue, optional): Standard error of parameter. Defaults to None.
        """
        if not isinstance(unit, str):
            unit = str(unit)
        self.data_list.append( (name,value,unit,error) )

    def __str__(self, value_significant_digit = 8, error_significant_digit=5, ratio_significant_digit=5, padding = 2):
        """Convert table to string

        Args:
            value_significant_digit (int): Significant digit of value. Defaults to 8.
            error_significant_digit (int): Significant digit of standard error. Default to 5.
            ratio_significant_digit (int): Significant digit of ratio between value and standard error in percents. Default to 5.
            padding (int): The number of space between each element. Defaults to 2.

        Returns:
            str: formatted table.
        """
        data_list = self.data_list
        if type(data_list) != list:
            data_list = [data_list]
        
        names = [item[0] for item in data_list]
        values = [item[1] for item in data_list]
        units  = [item[2] for item in data_list]
        errors = [item[3] for item in data_list]

        def align_and_padding(string_list):
            field_length = max([len(string) for string in string_list]) + padding
            fields = [string.ljust(field_length) for string in string_list]
            return fields

        name_fields = align_and_padding(names)

        value_significant_digit_format = "{:." + str(value_significant_digit) + "g}"
        value_strs = []
        for index, value in enumerate(values):
            if type(value) != LabradValue:
                value_str = value_significant_digit_format.format(value)
            else:
                value_str = value_significant_digit_format.format(value[units[index]])
            value_strs.append(value_str)
        value_fields = align_and_padding(value_strs)

        unit_fields = align_and_padding(units)

        error_significant_digit_format = "{:." + str(error_significant_digit) + "g}"
        error_strs = []
        for index, error in enumerate(errors):
            if error is None:
                error_str = ""
            else:
                if type(error) != LabradValue:
                    error_str = error_significant_digit_format.format(error)
                else:
                    error_str = error_significant_digit_format.format(error[units[index]])
            error_strs.append(error_str)
        error_fields = align_and_padding(error_strs)

        error_unit_strs = []
        for index, error in enumerate(errors):
            if error is None:
                error_unit_str = ""
            else:
                error_unit_str = units[index]
            error_unit_strs.append(error_unit_str)
        error_unit_fields = align_and_padding(error_unit_strs)

        ratio_significant_digit_format = "{:." + str(ratio_significant_digit) + "g}"
        ratio_strs = []
        for index, error in enumerate(errors):
            if error is None:
                ratio_str = ""
            else:
                ratio = error / np.abs(values[index]) * 100
                ratio_str = ratio_significant_digit_format.format(ratio)
            ratio_strs.append(ratio_str)
        ratio_fields = align_and_padding(ratio_strs)

        table_string = ""
        for index,error in enumerate(errors):
            if error is not None:
                table_string += "{}= {}{}+- {}{}({}%)\n".format(name_fields[index], value_fields[index], unit_fields[index], error_fields[index], error_unit_fields[index], ratio_fields[index])
            else:
                table_string += "{}= {}{}\n".format(name_fields[index], value_fields[index], unit_fields[index])
        return table_string

