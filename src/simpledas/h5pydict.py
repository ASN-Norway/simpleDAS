__all__ = [
    "DictFile",
    "load",
    "load_keys",
    "save",
]

from h5py import File, Group, Dataset
import numpy as np


class DictFile(File):
    '''
    Loads the whole hdf5-file into a dictionary tree with numpy-array data.
    Usage:
    Reading:
        with h5py.DictFile('test.hdf5','r') as f:
            data=f.load_dict()
    equivalent to:
        data = h5py.load('test.hdf5')
    Reading only a spesific field/group:
        with h5py.DictFile('test.hdf5','r') as f:
            data=f.load_dict(field='big')
    equivalent to:
        data = h5py.load('test.hdf5',field = 'big')
    Reading all data, except some spesific fields:
        with h5py.DictFile('test.hdf5','r') as f:
            data=f.load_dict(skipFields=['big'])
    equivalent to:
        data = h5py.load('test.hdf5',skipFields = ['big'])
    Saving:
        with h5py.DictFile(filename,'w') as f:
            f.save_dict(data)
    equivalent to:
        h5py.save('test.hdf5',data)

    Appending to an existing file:
        with h5py.DictFile(filename,'a') as f:
            f.save_dict(data)

    Only load the keys:
        with h5py.DictFile('test.hdf5','r') as f:
            keys=f.load_keys()
    equivalent to:
        h5py.load_keys('test.hdf5')

    '''

    def __init__(self, filename, mode, verbose=False, **kwargs):
        '''
        Paramters:
        ------------
        filename: str
             name of dict file
        mode: char
            file mode: 'r','w','a' etc
        verbose: bool
            print out a tree view of content

        '''
        self.verbose = verbose
        if verbose:
            print("Open file: " + filename)
        # import pdb;pdb.set_trace()
        super().__init__(filename, mode, **kwargs)

    def save_dict(self, data):
        """Save entire dictionary tree in hdf5 format"""
        # import pdb; pdb.set_trace()
        if self.verbose:
            print("Save dictinary tree to: " + str(self.filename))
        self.__recursion_save(data, self, 0)

    def load_dict(self, field=None, skipFields=[], getData=True):
        """Load entire dictionary tree in hdf5 format.
        Parameters:
        ----------
        field : hdf5 group or str
            Only load a spesific field
            If None, the whole file is loaded.
        skipFields: list
            list of fields not to load
        getData : list or bool
            is a list keys(at final level) that will be loaded or True to load all.
        """
        if self.verbose:
            line = "Load dictinary from "
            if field:
                line += f'field "{field}"'
            line += " within " + str(self.filename) + ":"
            print(line)

        data = {}
        self.skipFields = skipFields
        if isinstance(field, str):
            field = self[field]
        if isinstance(field, Dataset):
            data = field[()]
        else:
            self.__recursion_load(self if field is None else field, data, 0, getData)
        return data

    def load_keys(self):
        """Loads the hdf5-file tree into a dictionary, without the dataset.
        Instead the datasets are replaced by a tuple with the shape of array.
        """
        if self.verbose:
            print("Loading tree structure: " + str(self.filename))
        data = {}
        self.__recursion_load(self, data, 0, False)
        return data

    def __recursion_save(self, tree, parentgroup, depth=0):
        if tree is None or len(tree) == 0:
            if self.verbose:
                print("\t" * depth, "-")
        else:
            for key, val in tree.items():
                if isinstance(val, dict):
                    if self.verbose:
                        print("\t" * depth + str(key))
                    newgroup = parentgroup.create_group(str(key))
                    try:
                        self.__recursion_save(val, newgroup, depth + 1)
                    except RuntimeError as er:
                        print(val, newgroup, depth + 1)
                        raise (er)
                else:
                    if not isinstance(val, np.ndarray):
                        val = np.array(val)
                    try:
                        parentgroup.create_dataset(str(key), data=val)
                    except TypeError as e:
                        if isinstance(val, np.ndarray):
                            val2 = np.array(val, dtype='S')
                            parentgroup.create_dataset(str(key), data=val2)
                        else:
                            print(e)
                if self.verbose:
                    self.__print_info(str(key), val, depth)

    def __recursion_load(self, tree, parentdict, depth=0, getData=True):
        if tree is None or not isinstance(tree, Group):  # siste nivå
            if self.verbose:
                print("\t" * depth, "-")
        else:
            for key, val in tree.items():
                if key in self.skipFields:
                    continue
                if isinstance(key, str) and key.isdigit():
                    key = int(key)
                else:
                    key = str(key)
                if isinstance(val, Group):
                    if self.verbose:
                        print("\t" * depth + key)
                    parentdict[key] = {}
                    self.__recursion_load(val, parentdict[key], depth + 1, getData)
                else:
                    if (getData is True) or (isinstance(getData, list) and getData.count(key) > 0):
                        if val.dtype.char in 'SO':
                            if val.shape == ():
                                if val[()] == b'None':
                                    parentdict[key] = None
                                else:
                                    try:
                                        parentdict[key] = val[()].decode("utf-8")
                                    except:
                                        parentdict[key] = val[()]
                            else:
                                parentdict[key] = np.array(val).astype(str)
                        elif val.shape == ():  # convert numpy singleton to python scalar
                            parentdict[key] = val[()].item()
                        else:
                            parentdict[key] = np.array(val[()])
                    else:
                        parentdict[key] = val.shape
                    if self.verbose:
                        self.__print_info(key, val, depth)

    def __print_info(self, key, val, depth, maxlength=16):
        if len(val.shape) == 0 or len(val) <= maxlength:
            print("\t" * depth + ("{} ({}):".format(key, str(val.dtype))).ljust(30) + str(val[()]))
        else:
            print(
                "\t" * depth + ("{} ({}):".format(key, str(val.dtype))).ljust(30) + str(val.shape)
            )


def save(filename, data, **kwargs):
    '''save dictionary data to hdf5-file

    Parameters:
    -----------
    filename: str
         name of dict file
    data: dict
        dictionary with the data to be saved in hdf5-file
    verbose: bool
         print out a tree view of content
    '''
    with DictFile(filename, 'w', **kwargs) as f:
        f.save_dict(data)


def load(filename, field=None, skipFields=[], **kwargs):
    '''read dictionary data from hdf5-file

    Parameters:
    -----------
    filename: str
         name of dict file
    field : str
        Only load a spesific field
        If None, the whole file is loaded.
    skipFields: list
        list of fields not to load
    verbose: bool
        print out a tree view of content
    Returns:
    ----------
    data: dict
        dictionary with the fields in hdf5-file

    '''
    with DictFile(filename, 'r', **kwargs) as f:
        data = f.load_dict(field=field, skipFields=skipFields)

    return data


def load_keys(filename, **kwargs):
    """Loads the hdf5-file tree into a dictionary, without the dataset.
    Instead the datasets are replaced by a tuple with the shape of array.
    """
    with DictFile(filename, 'r', **kwargs) as f:
        data = f.load_keys()

    return data


if __name__ == '__main__':
    pass
