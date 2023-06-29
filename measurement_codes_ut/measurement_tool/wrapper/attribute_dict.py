
from __future__ import annotations
from typing import Optional, List, Any



class AttributeDict(dict):
    """Slightly modified AttrDict.

    Original AttrDict provides a way to access items in dict like access to attribute.

    Note:
    In the original AttrDict, when we access sub-directory with AttrDict, AttrDict returns dict object, not AttrDict object.
    Therefore, access to the attribute in sub-directory, such as "attr_dict.folder.item" and "attr_dict["folder"].item", is not allowed.
    This is very confusing behavior because we need to carefully check current dictionary-like object is dict or AttrDict.
    This class returns AttributeDict if the specified item is dict, by overriding __getattr__ and __getitem__ method.
    """

    def __init__(self, init_dict: Optional[dict] = None) -> None:
        """Initialize function

        Args:
            init_dict (dict): If not None, update object with a given dictionary.
        """
        # if initialized without dict, do nothing
        if init_dict is None:
            dict.__init__(self)
        # if initialized with dict-argument, convert sub-dict to AttributeDict
        elif isinstance(init_dict, dict):
            dict.__init__(self)
            for key, value in init_dict.items():
                if isinstance(value, dict):
                    value = AttributeDict(value)
                self.__setitem__(key, value)
        # if initialized with list-argument, convert sub-dict to AttributeDict
        elif isinstance(init_dict, list) or isinstance(init_dict, tuple):
            dict.__init__(self)
            for key, value in init_dict:
                if isinstance(value, dict):
                    value = AttributeDict(value)
                self.__setitem__(key, value)
        else:
            raise ValueError("Cannot convert values to dict")

    def __getattr__(self, key: str) -> Any:
        """get attribution, redirected to getitem

        Args:
            key (str): key

        Raises:
            AttributeError: key is not found

        Returns:
            Any: value
        """
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(key)

    def __getitem__(self, key: str) -> Any:
        """get item

        Args:
            key (str): key

        Returns:
            Any: value
        """
        return dict.__getitem__(self, key)

    def __setattr__(self, key: str, value: Any) -> None:
        """Set value, redirected to setitem

        Args:
            key (str): key to set
            value (Any): value to set
        """
        self.__setitem__(key, value)

    def __delattr__(self, name):
        dict.__delitem__(self, name)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set value

        Args:
            key (str): key to set
            value (Any): value to set
        """
        if isinstance(value, dict):
            value = AttributeDict(value)
        dict.__setitem__(self, key, value)

    def items(self, recursive: bool = False, ignore: str = "_") -> List[tuple]:
        """Get list of items

        Args:
            recursive (bool, optional): If true, recursively find items. efaults to False.
            ignore (str, optional): If specified, items which starts from this prefix str is eliminated from list.

        Returns:
            List[tuple]: List of tuple (key, value)
        """
        if not recursive:
            return super().items()

        def extract(d, prefix=''):
            parameters = []
            for key, val in sorted(d.items()):
                if ignore and key.startswith(ignore):
                    continue
                if isinstance(val, dict):
                    parameters.extend(extract(val, prefix + key + '.'))
                else:
                    parameters.append((prefix + key, val))
            return parameters
        return extract(self)

    def load_items(self, obj: List[tuple]) -> None:
        """Load from list of tuple items.

        Key will be separated by dot ".", and these are recognized as nested dictionary.
        For example, "folder1.folder2.item". This means using dot for item name is prohibited.

        Args:
            obj (List[tuple]): item to load
        """
        for path, value in obj:
            elements = path.split(".")
            subdirs = elements[:-1]
            key = elements[-1]
            cursor = self
            for subdir in subdirs:
                if subdir not in cursor.keys():
                    cursor[subdir] = {}
                cursor = cursor[subdir]
            cursor[key] = value

    def copy(self, deep: bool = True, directories: bool = True) -> AttributeDict:
        """Create copy of this dictionary

        NOTE: this class is a subclass of dict objects.
        Dict object is mutable object, i.e., copied variable will refer the same object.
        To create independent object, we need to iteratively create explicit copy.

        Args:
            deep (bool, optional): If true, recursively create copy. Defaults to True.
            directories (bool, optional): If ture, save dictionary addtional to key. Defaults to True.

        Returns:
            AttributeDict: copied instance
        """
        d = AttributeDict()
        for k, v in self.items():
            if isinstance(v, AttributeDict):
                if directories:
                    if deep:
                        d[k] = v.copy(deep, directories)
                    else:
                        d[k] = v
            else:
                d[k] = v
        return d
