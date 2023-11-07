#!/usr/bin/env python3
"""
Module to print the content of HDF5-files.

Created on Fri Jun  8 14:47:38 2018

@author: ohwaagaard
"""


from argparse import ArgumentParser, RawDescriptionHelpFormatter
import copy
from simpledas import h5pydict
import os
import sys


def getValue(dd, subkey):
    '''
    Get value from subkey within dict dd.
    eg. index subkey using a/b instead of dd[a][b].

    '''
    sdd = copy.deepcopy(dd)
    for key in subkey.split('/'):
        sdd = sdd[key]
    return sdd


def print_file(filename, group=None, key=None, as_table=False, print_header=False):
    '''
    Print  group or key info from a hdf5 file

    Parameters
    ----------
    filename : str
        fullpath for the file to print info from.
    group : str, optional
        Name of sub group to print.
    key : str, optional
        Name of key or keys. Assume mulitple keys to be  comma separated.
    as_table : bool, optional
        Format output as a tabulated table. The default is False.
    print_header : bool, optional
          Display a header line above the table if as_table is true.

    Returns
    -------
    None.

    '''
    if key is None:
        verbose = True
    else:
        verbose = False

    with h5pydict.DictFile(filename, 'r', verbose=verbose) as f:
        if key is not None:
            width = len(filename) + 5
            # prints the header
            if as_table and print_header:
                try:
                    print(
                        f'%-{width}s' % 'Filename'
                        + ''.join([f'%-{len(k)+5}s' % k for k in key.split(',')])
                    )
                except ValueError as e:
                    print(f"\tKeyError {str(e)}")

            # print key values into columns
            line = f'%-{width}s' % filename
            for k in key.split(','):
                if group is not None:
                    k = group + '/' + k
                try:
                    if as_table:
                        val = f.load_dict(k, getData=False)
                        line += f'%-{max(len(k),len(str(val)))+5}s' % val
                    else:
                        line += f'\t{k}:{f.load_dict(k)}'
                except KeyError as e:
                    line += f"\tKeyError {str(e)}"
            print(line)
        else:
            f.load_dict(getData=not verbose, field=group)


def print_files(files=[], directory=None, group=None, key=None):
    '''Print info from HDF5 files. Print all files in directory when given.'''

    nFiles = 0
    if directory is not None:
        files += os.listdir(directory)
        files.sort()
    for file in files:
        if os.path.isdir(file):  # Recusiveley call for directories
            print_files(directory=file, group=group, key=key)
        else:
            if file.endswith(".hdf5"):
                if directory is not None:
                    print_file(
                        os.path.join(directory, file),
                        group=group,
                        key=key,
                        as_table=True,
                        print_header=(nFiles == 0),
                    )
                else:
                    print_file(
                        file, group=group, key=key, as_table=True, print_header=(nFiles == 0)
                    )
                nFiles += 1
    if nFiles == 0 and directory is not None:
        print("No HDF5 is directory.")


def main():
    parserArg = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parserArg.add_argument("path", help="Files or directory to print", type=str, nargs='+')
    parserArg.add_argument("-g", "--group", help="Group to print")
    parserArg.add_argument(
        "-t",
        "--traceback",
        default=False,
        action='store_true',
        help="Show python traceback. Is hidden by default",
    )
    parserArg.add_argument(
        "-k",
        "--key",
        help="Key to print,sub keys are separated by '/', eg  'timing/ppses'. Separate multiple keys with ','. ",
    )
    args = parserArg.parse_args()

    if not args.traceback:
        sys.tracebacklimit = 0
    print_files(args.path, group=args.group, key=args.key)


if __name__ == '__main__':
    main()
