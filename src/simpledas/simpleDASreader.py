"""
simpleDASreader v8.2 - OptoDAS hdf5 file v8 reader for Python.

    Copyright (C) 2021 Alcatecl Submarine Networks Norway AS,
    Vestre Rosten 77, 7075 Tiller, Norway, https://www.asn.com/

    This file is part of simpleDAS.

    simpleDAS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    simpleDAS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

__all__ = [
    "DASDataFrame",
    "load_DAS_files",
    "find_DAS_files",
    "get_data_indexes",
    "get_filemeta",
    "save_to_DAS_file",
    "str2datetime",
    "convert_index_to_rel_time",
    "create_time_axis",
    "unwrap",
    "wrap",
]

import datetime
import os
import re
import warnings

import numpy as np
import pandas as pd
from pandas._typing import Axes
from sympy import symbols, sympify
from collections.abc import Iterable
from simpledas import h5pydict


def load_DAS_files(
    filepaths: str | list[str],
    chIndex: None | slice | list[int] | np.ndarray = None,
    samples: None | slice | int = None,
    sensitivitySelect: int = 0,
    integrate: bool = True,
    unwr: bool = False,
    spikeThr: None | float = None,
    userSensitivity: None | dict = None,
    useLabeledColumns: None | bool = None,
    verbose: bool = False,
) -> DASDataFrame:
    """
    Loads OptoDAS recorded datafiles to pandas dataframe.


    Parameters
    ----------
    filepaths: string or list of strings
        Full path + filename of files to load.

    chIndex: 1d array, list, slice or None
        Channel indices to load.
        None => load all available channels (default).

    samples: slice, int, or None
        Time indices to read. When reading multiple files,
        the index counter continous into subsequent files.
        None  => load all available time samples (default).
        int   => Load this number of samples, starting from first sample.
        slice => Range of indices, i.e. samples=slice(start,stop)
        Note: Decimation should be performed after integration.
              Decimation with the slice.step parameter raises error.

    sensitivitySelect: int
        Scale (divide) output with one of the sensitivites given in
        filemeta['header']['sensitivities'][sensitivitySelect,0] with unit
        filemeta['header']['sensitivityUnits'][sensitivitySelect], where
        filemeta can be extracted with filemeta = get_filemeta(filename).
        Default is sensitivitySelect = 0 which results in gives output in
        unit strain  after integration (or strain/s before).
        Additional scalings that are not defined in filemeta['header']['sensitivities']:
        sensitivitySelect= -1 gives output in unit rad/m
        sensitivitySelect= -2 gives output in unit rad (phase per GL)
        sensitivitySelect= -3 uses sensitivity defined in userSensitivity
    integrate: bool
        Integrate along time axis of phase data (default).
        If false, the output will be the time differential of the phase.

    unwr: bool
        Unwrap strain rate along spatial axis before time-integration. This may
        be needed for strain rate amplitudes > 8π/(dt*gaugeLength*sensitivity).
        If only smaller ampliudes are expected, uwr can be set to False.
        Default is False.

    spikeThr: float or None
        Threshold (in rad/m/s) for spike detection and removal.
        Sweep rates exceeding this threshold in absolute value are set to zero
        before integration.
        If there is steps or spikes in the data one may try setting
        with spikeThr = 3/(gaugeLength*dt). Higher values may also be usefull.
        Default is None, which deactivates spike removal.
        Be aware that spike removal disables unwrapping, and uwr should
        therefore be set to False when spike removal is activated.

    userSensitivity: dict
        Define sensitivites not provided by filemeta['header']['sensitivities'],
        or overwrite the provided sensitivity.
        Set sensitivitySelect=-3 to use this field.
        sensitivity: float
            The user defined sensitivity
        sensitivityUnit: str
            The unit of the sensitivity. Should be rad/m/<wanted output unit>,
            e.g for temperature, sensitivityUnit = rad/m/K.
        Example usage:
        userSensitivity={'sensitivity': 9.36221e6, 'sensitivityUnit': 'rad/(m*strain)'}),
        which will output data in unit strain.

    useLabeledColumns: bool
        Control the naming of the columns of the dataframe.
        If True, use the channel labels defined in filemeta['header']['sensorType'],
        else if False use channel number.
        A ValueError is raised if True and channel label does not exist for all
        channels.
        If None, channel labels are used if available else use channel number.
    Returns
    -------
    dfdas: DASDataFrame
        row labels: absolute timestamp. pandas timeseries object
        col labels: channel number or label
        A correponding numpy array can be extracted as dfdas.to_numpy()

        dfdas.meta: dict
            meta information describing the data set
            fileVersion: int
                format version number of the input file
            time: datetime
                timestamp of first sample
            dx: float
                channel separation in m
            dt: float
                sample interval in s
            gaugeLength: float
                sensor gaugelength in m
            unit: str
                unit of data
            sensitivities: float
                the sensitivities available for scaling the data.
                if sensitivitySelect>=0 or -3, this is set to 1.0,
                since sensitivity is already applied.
            sensitivityUnits: str
                the units of the sensitivities
            experiment: str
                name of experiment
            filepaths: list of str
                full path to all loaded files in the dataset
            sensorType: list of str
                Name of sensor type. For das data this is a single value 'D'.
                In other configurations, each of the sensors channels may be named.


    """
    loadstart = datetime.datetime.now()

    if isinstance(filepaths, str):
        filepaths = [filepaths]

    if filepaths is None or len(filepaths) == 0:
        raise ValueError("No filepaths provided")

    if isinstance(samples, range):
        # warnings.warn('The use of range for samples is deprecated, use slice instead',DeprecationWarning)
        samples = slice(samples.start, samples.stop, samples.step)
    if isinstance(samples, int):
        samples = slice(0, samples)
    elif samples is None:
        samples = slice(None)
    if samples.step is not None and samples.step > 1:
        raise ValueError("Sample step>1 in time is not supported")

    if chIndex is None:
        chIndex = slice(None)
    elif not isinstance(chIndex, slice):
        chIndex = np.atleast_1d(chIndex)
        if len(chIndex) == 0:
            raise ValueError("Selected channels are not in found in file!")
        elif len(chIndex) == 1:
            chIndex = slice(chIndex[0], chIndex[0] + 1, 1)
        else:
            # if equal step between indeces, chIndex can be represented as a slice (more efficient reading)
            dchIndex = np.diff(chIndex)
            if np.all(dchIndex == dchIndex[0]):
                chIndex = slice(chIndex[0], chIndex[-1] + 1, dchIndex[0])

    # load all files
    samples_read = 0
    if verbose:
        print('Loading files:', end=' ')
    for n, filepath in enumerate(filepaths):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")
        if verbose:
            print(f"{os.path.basename(filepath)}", end=' ')
        with h5pydict.DictFile(filepath, "r") as f:
            if f["data"].ndim > 2:
                raise IndexError("simpleDAS does not handle data with more than 2 dimensions")
            nSamples_in_file, nChs_in_file = f["data"].shape
            if n == 0:
                nChs_out = np.arange(nChs_in_file)[chIndex].size
                # Some files might have an extra samples
                samples_max = (nSamples_in_file + 1) * len(filepaths)
                nSamples_out = np.arange(samples_max)[samples].size
                samples_rem = nSamples_out

                data = np.zeros((nSamples_out, nChs_out), dtype=np.float32)

                m = f.load_dict(skipFields=["data"])
                _fix_meta_back_compability(m, nSamples_in_file, nChs_in_file)

                if m["header"]["channels"][chIndex].size == 0:
                    raise IndexError(
                        f"chIndexes {chIndex} not in data (nChannels = {nChs_in_file})"
                    )

                sensitivity, unit_out, sensitivities_out, sensitivityUnits_out = _set_sensitivity(
                    m["header"], sensitivitySelect, userSensitivity
                )
                if not isinstance(sensitivity, np.ndarray) or sensitivity.size == 1:  # singelton
                    if sensitivity != 0.0:
                        scale = np.float32(m["header"]["dataScale"] / sensitivity)
                    else:
                        raise ValueError(
                            "Sensitivity value set to zero, use sensitivitySelect=-1 or -2 to avoid error"
                        )
                else:  # array with different sensitivity each channel
                    if nChs_in_file != len(sensitivity):
                        raise ValueError(
                            "Length of sensitivity vector does not match number of channels"
                        )
                    if np.any(sensitivity != 0.0):
                        sensitivity = np.atleast_2d(sensitivity[chIndex])
                        scale = np.array(
                            m["header"]["dataScale"] / sensitivity, dtype=np.float32
                        )  # make 2D
                    else:
                        raise ValueError(
                            "Sensitivity value set to zero, use sensitivitySelect=-1 or -2 to avoid error"
                        )

                if samples.stop is None or samples.stop > nSamples_in_file:
                    samples_to_read = slice(samples.start, None)
                else:
                    samples_to_read = slice(samples.start, samples.stop)
            else:
                if samples_rem >= nSamples_in_file:
                    samples_to_read = slice(None)
                else:
                    samples_to_read = slice(0, samples_rem)
            try:
                sub_data = f["data"][samples_to_read, chIndex]
            except:
                # h5py does not accept complex slices, do in two steps if neccessary
                warnings.warn("h5py does not accept complex slices. Slow read...", UserWarning)
                sub_data = f["data"][:, chIndex][samples_to_read, :]
            nSamples_read_file = sub_data.shape[0]
            data[samples_read : samples_read + nSamples_read_file, :] = (
                np.atleast_2d(sub_data) * scale
            )
            samples_read += nSamples_read_file
            samples_rem = nSamples_out - samples_read
    if verbose:
        print(' in %.1f s' % (datetime.datetime.now() - loadstart).total_seconds())
    if verbose:
        print('Conditioning... ', end='')
    data = data[:samples_read, :]

    if (unwr or spikeThr or integrate) and m["header"]["dataType"] == 2:
        unwr, spikeThr, integrate = (False,) * 3
        warnings.warn(
            "Options unwr, spikeThr or integrate can only be\
                             used with time differentiated phase data",
            UserWarning,
        )

    if unwr and m["header"]["spatialUnwrRange"]:
        data = unwrap(data, m["header"]["spatialUnwrRange"] / sensitivity, axis=1)

    if spikeThr is not None:
        data[np.abs(data) > spikeThr / sensitivity] = 0

    if integrate:
        unit_new = _combine_units([unit_out, "s"])
        # check that s in not in unit after integraion
        if not any([u == "s" for u in re.findall(r"[\w']+", unit_new)]):
            data = np.cumsum(data, axis=0) * m["header"]["dt"]
            unit_out = unit_new
        else:
            warnings.warn(
                "Data unit %s is not differentiated. Integration skipped." % unit_out,
                UserWarning,
            )
        m["header"]["name"] = m["header"]["name"].replace(" rate", "")

    # create timeaxis
    nstart = 0 if samples.start is None else samples.start
    tstart = datetime.datetime.utcfromtimestamp(m["header"]["time"] + nstart * m["header"]["dt"])
    nSamples, nCh = data.shape
    t = create_time_axis(tstart, nSamples, m["header"]["dt"])

    # create pandas dataframe
    meta = {
        key: m["header"][key]
        for key in [
            "dt",
            "dx",
            "gaugeLength",
            "experiment",
            "dataType",
            "dimensionRanges",
            "dimensionUnits",
            "dimensionNames",
            "name",
            "sensorType",
        ]
    }
    meta.update(
        fileVersion=m["fileVersion"],
        time=tstart,
        unit=unit_out,
        sensitivities=sensitivities_out,
        sensitivityUnits=sensitivityUnits_out,
        filepaths=filepaths,
    )
    # use sensortype as columns if used as sensor labels
    if useLabeledColumns or useLabeledColumns is None:
        if len(m["header"]["sensorType"]) != nChs_in_file:
            if useLabeledColumns is None:
                columns = m["header"]["channels"][chIndex]
            else:
                raise ValueError("Cannot use labeled columns")
        else:
            columns = m["header"]["sensorType"][chIndex]
    else:
        columns = m["header"]["channels"][chIndex]

    dfdas = DASDataFrame(data, columns=columns, index=t, meta=meta)
    [os.path.basename(f) for f in filepaths]
    if verbose:
        print("Total load time: %.1f s" % (datetime.datetime.now() - loadstart).total_seconds())

    return dfdas


def find_DAS_files(
    experiment_path: str,
    start: str | datetime.datetime,
    duration: float | datetime.timedelta,
    channels: None | slice | list[int] | np.ndarray = None,
    datatype: str = "dphi",
    show_header_info: bool = True,
    load_file_from_start: bool = True,
) -> tuple[list[str], np.ndarray, slice]:
    """
    Returns of list of fullpath to DAS files starting at date and time <start> and
    with duration <duration>.
    The filepaths will be returned given as <exppath>/<date>/<datatype>/<HHMMSS>.hdf5,
    where <date> and <HHMMSS> is extracted from <start>.

    Also returns the time sample indeces and channel indeces calculated from parameters
    start, duration and channels.

    Parameters:
    ----------
    experiment_path: str
        File path containing DAS files.
        The experiment_path should point to a directory with sublevels <date>/<datatype>,
        two levels above the location of the files.
        exppath may also point to the directory containing the DAS files,
        however then the date portion of start is ignored.

    start: str or datetime.datetime object
        Start time of data to extract on format 'YYYYmmdd HHMMSS'. May also be datetime.datetime object.
        YYYYmmdd may be excluded if exppath points directly to the DAS file directory.

    duration: float or datetime.timedelta object
        Duration in seconds to be loaded. May also be datetime.timedelta object.

    channels: None,list or slice
        list of DAS channels to read. Also accept slice object as slice(start_channel, stop_channel, step).
        Load all if None.

    datatype: str
        datatype/subfolder to read. Usually 'dphi' (default) or 'proc'.

    show_header_info: bool
        Show file header information of first file

    load_file_from_start: bool
        It True, load from start of the file containing the start timestamp, else
        calculated to sampleno corresponing to start timestamp

    Returns:
    ---------
    filepaths: list of strings
        Returns the full path to the files to load.

    chIndex: ndarray
        If input channels provided, returns the channel indices in the data array
        corresponding to DAS channels input, else return None.

    samples: slice
        The sample indices calculated from start and duration.
    """

    # Lambda function to test a datestring, will fail if d is not a date string
    def checkdate(d):
        return datetime.datetime.strptime(d, "%Y%m%d").strftime("%Y%m%d")

    exper_path = experiment_path
    dates_in_exp = []
    exppath_has_dates = False
    for d in sorted(os.listdir(exper_path)):  # get all dates
        if os.path.isdir(os.path.join(exper_path, d)):
            try:
                dates_in_exp.append(checkdate(d))
                exppath_has_dates = True
            except ValueError:
                pass
    if not exppath_has_dates:  # dates not found in exppath, parse levels up to find date
        path = exper_path
        dprev = ''
        for n in range(2):
            path, d = os.path.split(path)
            try:
                dates_in_exp.append(checkdate(d))
                exppath_has_dates = True
                exper_path = path
                if len(dprev) > 0:  # datatype down 1 level from date
                    datatype = dprev
                break
            except:
                dprev = d
                pass
        if len(dates_in_exp) == 0:
            dates_in_exp.append("19000101")  # placeholder for no dateinfo

    # get all files within selected start duration
    if isinstance(start, str):
        datetime_start = str2datetime(start)
    else:  # assume datetime object
        datetime_start = start
    if datetime_start.year == 1900 and exppath_has_dates:
        # no date set for start, set date to first date in experiment
        datetime_start += str2datetime(dates_in_exp[0], True) - datetime.datetime(1900, 1, 1)

    if not isinstance(duration, datetime.timedelta):
        duration = datetime.timedelta(seconds=duration)

    datetime_stop = datetime_start + duration

    ffidpaths = []
    header_times = []
    for date_in_exp in dates_in_exp:
        date1 = str2datetime(date_in_exp, True)
        if datetime_start.date() <= date1.date() <= datetime_stop.date():
            ffiddir = (
                os.path.join(exper_path, date_in_exp, datatype)
                if exppath_has_dates
                else exper_path
            )

            for ffid in sorted(os.listdir(ffiddir)):
                try:
                    ffidTime = str2datetime(date_in_exp + " " + os.path.splitext(ffid)[0])
                except:
                    continue
                if (
                    datetime_start - datetime.timedelta(seconds=11)
                    < ffidTime
                    <= datetime_stop + datetime.timedelta(seconds=1)
                ):
                    ffidpath = os.path.join(ffiddir, ffid)
                    # ffidTime may be off by upto 1s. Check headerTime
                    with h5pydict.DictFile(ffidpath, "r") as f:
                        header_time = datetime.datetime.utcfromtimestamp(
                            float(f["header"]["time"][()])
                        )
                    if datetime_start.year == 1900:
                        # remove date from header_time
                        header_time += datetime.datetime(1900, 1, 1) - datetime.datetime.combine(
                            header_time.date(), datetime.time(0)
                        )
                    if (
                        (datetime_start - datetime.timedelta(seconds=10))
                        < header_time
                        <= datetime_stop
                    ):
                        ffidpaths.append(ffidpath)
                        header_times.append(header_time)

    if len(ffidpaths) > 0:
        if load_file_from_start:
            start_time = 0.0
        else:
            start_time = max(0.0, (datetime_start - header_times[0]).total_seconds())
        trange = (start_time, start_time + duration.total_seconds())
        chIndex, samples = get_data_indexes(ffidpaths, trange, channels, show_header_info)
    else:
        chIndex = None
        samples = None
        warnings.warn("simpleDASreader.find_DAS_files did not find any files", UserWarning)

    return ffidpaths, chIndex, samples


def get_data_indexes(
    ffidpath: str | list[str],
    trange: None | list[float] | np.ndarray = None,
    channels: None | slice | list[int] | np.ndarray = None,
    show_header_info: bool = True,
):
    """
    Compute the  the time sample indexes and channel indeces calculated from parameters
    trange and channels.

    Can be used instead of find_DAS_files when the ffidpath is given as input
    rather than the start timestamp.

    Parameters
    ----------
    filepaths: str or list of str
        The full path to the files to load.
    trange: list of floats
        start and stop time interval in s relative to start of first file in ffidpath
    channels: None,list or slice
        list of DAS channels to read. Also accept slice object as slice(start_channel, stop_channel, step).
        Load all if None.
    show_header_info: bool
        Show file header information of first file


    Returns:
    ---------


    chIndex: ndarray
        If input channels provided, returns the channel indices in the data array
        corresponding to DAS channels input, else return None.
        To be used as input to load_DAS_files

    samples: slice
        The sample indices calculated from start and duration.
        To be used as input to load_DAS_files


    """
    if isinstance(ffidpath, list):
        ffidpath = ffidpath[0]
    with h5pydict.DictFile(ffidpath, "r") as f:
        m = f.load_dict(skipFields=["data"])
        nSamples, nChs = f["data"].shape
        _fix_meta_back_compability(m, nSamples, nChs)

    header = m["header"]
    headerTime = datetime.datetime.utcfromtimestamp(float(header["time"]))
    # calculate sample range
    if trange is None:
        samples = slice(None)
    else:
        if isinstance(trange, (int, float)):
            trange = (0.0, trange)
        elif len(trange) == 1:
            trange = (0.0, trange[0])

        nSamples_to_be_read = int(np.diff(trange)[0] / header["dt"] + 0.5)
        startSample = int(trange[0] / header["dt"] + 0.5)
        samples = slice(startSample, startSample + nSamples_to_be_read)

    if show_header_info:
        print("-- Header info file: %s --" % os.path.basename(ffidpath))
        print("\tExperiment:            %s" % header["experiment"])
        print("\tFile timestamp:        %s" % headerTime.strftime("%Y-%m-%d %H:%M:%S"))
        print("\tType of data:          {}, unit: {}".format(header["name"], header["unit"]))
        print("\tSampling frequency:    %.2f Hz" % (1.0 / header["dt"]))
        print("\tData shape:            %d samples x %d channels" % (nSamples, nChs))
        print("\tGauge length:          %.1f m" % header["gaugeLength"])
        print(
            "\tSensitivities:         %s"
            % ",".join(
                "{:.2e} {}".format(sens, unit)
                for sens, unit in zip(header["sensitivities"][:, 0], header["sensitivityUnits"])
            )
        )
        try:
            dim1 = header["dimensionRanges"]["dimension1"]
            print(
                "\tRegions of interest:   %s"
                % ",".join(
                    [
                        "%d:%d:%d"
                        % (start, stop, 1 if stop <= start else (stop - start) // (size - 1))
                        for start, stop, size in zip(dim1["min"], dim1["max"], dim1["size"])
                    ]
                )
            )
        except:
            pass

    # calculate channel indices
    if isinstance(channels, slice):
        channels = np.arange(header["channels"][-1])[channels]

    header_channels = header["channels"]
    if channels is not None:
        # get the intersection between channels in file and requested channels

        channels_found, chIndex, _ = np.intersect1d(header_channels, channels, return_indices=True)
        if show_header_info:
            with np.printoptions(threshold=20):
                print("\t%d channels requested:%s" % (len(channels), channels))
                print("\t%d channels found:    %s" % (len(channels_found), channels_found))
    else:
        chIndex = np.arange(len(header_channels))
    if show_header_info:
        print("-----------------------------------")

    if len(chIndex) == 0:
        raise ValueError("Selected channels are not in found in file!")

    return chIndex, samples


def get_filemeta(filepath: str, metaDetail: int = 1):
    """


     Parameters
     ----------
     filepath : str
         DAS file from which to extract meta data
     metaDetail: int
             Selects how much meta information from the file is outputed to filemeta
             1 => Load only metadata needed for DAS data interpretation (default)
             otherwise => Load all metadata

     Returns
     -------
    filemeta: dict
        optional extraction of meta information as is in the file
        The number of fields in filemeta depends on the metadetails input.
        Metadata fields relevant for end users are described in
        'OPT-TN1287 OptoDAS HDF5 file format description for externals.pdf'
    """
    with h5pydict.DictFile(filepath, "r") as f:
        m = f.load_dict(skipFields=["data"])
        if metaDetail == 1:
            mon = m["monitoring"]
            return dict(
                fileVersion=m["fileVersion"],
                header=m["header"],
                timing=m["timing"],
                cableSpec=m["cableSpec"],
                monitoring=dict(Gps=mon["Gps"], Laser=dict(itu=mon["Laser"]["itu"])),
            )
        else:
            return m


def save_to_DAS_file(
    dfdas: DASDataFrame,
    filepath: None | str = None,
    datatype: str = "processed",
    **auxmeta,
) -> tuple[str, dict]:
    """
    Save a DASDataframe to file in a format that is readable for load_DAS_files.
    Note that just a small subset of the meta information from original files are
    kept.
    Useful when a preprocessing step is required where the intermediate result
    should be saved.

    In order to copy all meta data to output file:
    dfdas = load_DAS_files(filename)
    filemeta = get_filemeta(filename,2)
    ... processing dfdas ...
    save_to_DAS_file(dfdas,**filemeta)

    Parameters
    ----------
    dfdas : DASDataframe
        The dataframe to be saved
    filepath : str, optional
        The filepath to which the data is saved.
        If None, the data is save to a file derived from the start time
        dfdas.meta['time'] and datatype in the path with datatype input parameter.
        E.g. a file read as /raid1/exp/exp1/20200101/dphi/120000.hdf5 will be
        saved as /raid1/exp/exp1/20200101/processed/120000.hdf5
        The default is None.
    datatype : str, optional
        A sub-directory desnoting the data. The default is 'processed'.
    auxmeta:
        additional meta groups saved to file.
    Returns
    -------
    filepath: str
        filepath to generated file

    """
    if filepath is None:
        timestr = dfdas.meta["time"].strftime("%H%M%S")
        filename = timestr + ".hdf5"
        filepath_orignal = dfdas.meta["filepaths"][0]
        path, filename_original = os.path.split(filepath_orignal)
        exp_path, datatype_original = os.path.split(path)
        path_new = os.path.join(exp_path, datatype)
        filepath = os.path.join(path_new, filename)

    path, filename = os.path.split(filepath)
    if not os.path.exists(path):
        os.makedirs(path)
    print("Saving to file: " + filepath)

    utctimestamp = dfdas.meta["time"].replace(tzinfo=datetime.timezone.utc).timestamp()
    dfdas._set_dimensionRange()
    header = dict(
        channels=dfdas.columns.values,
        dataScale=np.float32(1.0),
        spatialUnwrRange=np.float32(0.0),
    )
    header.update(**dfdas.meta)
    header.update(time=utctimestamp)

    fileVersion = header["fileVersion"]
    del header["fileVersion"]

    ffiddict = dict(header=dict(), fileVersion=fileVersion)
    ffiddict.update(data=np.asfarray(dfdas, np.float32), **auxmeta)
    ffiddict["header"].update(**header)

    h5pydict.save(filepath, ffiddict)

    return filepath, ffiddict


class DASDataFrame(pd.DataFrame):
    """
    Specialized DataFrame to include DAS meta info
    """

    _metadata = ["meta"]

    @property
    def _constructor(self):
        return DASDataFrame

    def __init__(
        self,
        data: np.ndarray | Iterable | dict | pd.DataFrame = None,
        index: None | Axes | np.ndarray = None,
        columns: None | Axes | np.ndarray = None,
        dtype: None | str = None,
        copy: None | bool = None,
        meta: dict = dict(),
    ):
        """
        Two-dimensional, size-mutable, potentially heterogeneous tabular data
        based on Pandas Dataframe  with additional meta data field.

        Data structure also contains labeled axes (rows and columns).
        Arithmetic operations align on both row and column labels. Can be
        thought of as a dict-like container for Series objects. The primary
        pandas data structure.

        Parameters
        ----------
        data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
            Dict can contain Series, arrays, constants, dataclass or list-like objects. If
            data is a dict, column order follows insertion-order.

            .. versionchanged:: 0.25.0
               If data is a list of dicts, column order follows insertion-order.

        index : Index or array-like
            Index to use for resulting frame. Will default to RangeIndex if
            no indexing information part of input data and no index provided.
        columns : Index or array-like
            Column labels to use for resulting frame when data does not have them,
            defaulting to RangeIndex(0, 1, 2, ..., n). If data contains column labels,
            will perform column selection instead.
        dtype : dtype, default None
            Data type to force. Only a single dtype is allowed. If None, infer.
        copy : bool or None, default None
            Copy data from inputs.
            For dict data, the default of None behaves like ``copy=True``.  For DataFrame
            or 2d ndarray input, the default of None behaves like ``copy=False``.

        meta: dict
            Additional meta information to be appended to data

        """
        super().__init__(data, index, columns, dtype, copy)
        self.meta = meta.copy()
        if len(meta):
            self._set_dimensionRange()

    def _set_dimensionRange(self):
        """
        Compute the meta|dimensionRanges
        """
        if np.issubdtype(self.columns.dtype, np.str_) or np.issubdtype(self.columns.dtype, object):
            chs = np.arange(len(self.columns))
        else:
            chs = np.array(self.columns)
        if len(chs) > 2:
            dchs = np.diff(chs)
            sdchs = dchs[:-1] != dchs[1:]
            sdchs &= np.r_[
                sdchs[1:], False
            ]  # dont include index when the next step is not changed
            stopind = np.r_[np.where(sdchs)[0] + 1, len(chs) - 1]
            startind = np.r_[0, stopind[:-1] + 1]
            size = np.array([len(chs[b:e]) + 1 for b, e in zip(startind, stopind)])
        else:
            startind = 0
            stopind = -1
            size = len(chs)
        self.meta["dimensionRanges"] = dict(
            dimension0={
                "max": self.shape[0] - 1,
                "min": 0,
                "size": self.shape[0],
                "unitScale": self.meta["dt"],
            },
            dimension1={
                "max": chs[stopind],
                "min": chs[startind],
                "size": size,
                "unitScale": self.meta["dx"],
            },
        )
        self.meta.update(
            dimensionSizes=np.r_[self.shape[0], np.sum(size)],
            dimensionUnits=np.array([b"s", b"m"]),
        )


def str2datetime(sdt: str, dateonly: bool = False) -> datetime.datetime:
    """
    Convert strings to datetime object
    Formats supported: YYYYmmdd HHMMSS or HHMMSS

    Parameters
    ----------
    sdt : str
        date/time string.

    Returns
    -------
    dateandtime : datetime
        datatime object

    """

    if isinstance(sdt, datetime.datetime):
        return sdt

    if dateonly:
        dt = datetime.datetime.strptime(sdt, "%Y%m%d")
    else:
        try:
            dt = datetime.datetime.strptime(sdt, "%H%M%S")
        except:
            dt = datetime.datetime.strptime(sdt, "%Y%m%d %H%M%S")
    return dt


def convert_index_to_rel_time(dfdas: DASDataFrame):
    """
    Convert the pandas index from datatime to float seconds relative to first sample

    Parameters
    ----------
    dfdas : pandas dataframe
        The dataframe to be modified


    """
    dfdas["t"] = (dfdas.index - dfdas.index[0]).total_seconds()  # seconds from start of data
    dfdas.set_index("t", inplace=True)  # set t as the new index


def create_time_axis(tstart: datetime.datetime, nSamples: int, dt: float):
    """
    Create a time axis with nSamples from tstart with sampling interval dt

    Parameters
    ----------
    tstart : datetime
        start time of axis
    nSamples : int
        number of samples
    dt : float
        sampling interval in s

    Returns
    -------
    tstart : DateTimeIndex
        start time of axis
    """

    return tstart + pd.timedelta_range(
        start="0s", periods=nSamples, freq="%dN" % int(dt * 1e9 + 0.5)
    )


def unwrap(phi: np.ndarray, wrapStep: float = 2 * np.pi, axis: int = -1) -> np.ndarray:
    """
    Unwrap phase phi by changing absolute jumps greater than wrapStep/2 to
    their wrapStep complement along the given axis. By default (if wrapStep is
    None) standard unwrapping is performed with wrapStep=2*np.pi.

    (Note: np.unwrap in the numpy package has an optional discont parameter
    which does not give an expected (or usefull) behavior when it deviates
    from default. Use this unwrap implementation instead of the numpy
    implementation if your data is wrapped with discontinuities that deviate
    from 2*pi.)
    """
    scale = 2 * np.pi / wrapStep
    return (np.unwrap(phi * scale, axis=axis) / scale).astype(phi.dtype)


def wrap(x: np.ndarray, wrapStep: float = 2 * np.pi) -> np.ndarray:
    """
    Inverse of the unwrap() function.
    Wraps x to the range [-wrapStep/2, wrapStep/2>.
    """
    if wrapStep > 0:
        return (x + wrapStep / 2) % wrapStep - wrapStep / 2
    else:
        return x


def _combine_units(units: list[str], operator: str = "*") -> str:
    """
    Combines units from a list of strings by chosen operator.

    Parameters
    ----------
    units : list
        List of strings. Units to combine by operator.
    operator: str
        String deciding what operator to combine units by.
        '*' => multiplication (default)
        '/' => division
    Returns
    -------------
    combinedUnit: str
        The combined unit.
    """
    knownUnits = (
        "rad",
        "dt",
        "strain",
        "W",
        "NormCts",
        "m",
        "K",
        "s",
        "Hz",
        "N",
        "Pa",
        "J",
    )
    usyms = symbols(knownUnits)
    udict = {u: s for u, s in zip(knownUnits, usyms)}
    try:
        units = [u.replace("·", "*").replace("ε", "strain") for u in units]
    except:
        pass
    combinedUnit = sympify(units[0], udict)
    for unit in units[1:]:
        if len(unit) > 0:
            if operator == "/":
                combinedUnit /= sympify(unit, udict)
            if operator == "*":
                combinedUnit *= sympify(unit, udict)

    return str(combinedUnit)


def _set_sensitivity(
    header: dict, sensitivitySelect: int, userSensitivity: None | dict = None
) -> tuple[float, str, float, str]:
    if sensitivitySelect >= 0:
        try:
            sensitivity = header["sensitivities"][sensitivitySelect, :]
            sensitivityUnit = header["sensitivityUnits"][sensitivitySelect]
        except KeyError:
            sensitivity = header["sensitivity"]
            sensitivityUnit = header["sensitivityUnit"]
        except IndexError:
            print("SensitivitySelect index %d not found in file." % sensitivitySelect)
            raise

        unit_out = _combine_units([header["unit"], sensitivityUnit], "/")
        # sensitivity applied, output sensitivity set to 1
        sensitivities_out = np.ones((1, 1), dtype=np.float32)
        sensitivityUnits_out = [""]

    elif sensitivitySelect == -1:
        unit_out = header["unit"]  # rad/m
        sensitivity = 1.0
        sensitivities_out = header["sensitivities"]
        sensitivityUnits_out = header["sensitivityUnits"]

    elif sensitivitySelect == -2:
        unit_new = _combine_units([header["unit"], "m"])  # should give rad
        if not any(
            [u == "m" for u in re.findall(r"[\w']+", unit_new)]
        ):  # check that unit does not have m
            unit_out = unit_new
            sensitivity = 1.0 / header["gaugeLength"]
            sensitivities_out = header["sensitivities"] * header["gaugeLength"]
            sensitivityUnits_out = [
                _combine_units([sensitivityUnit, "m"])
                for sensitivityUnit in header["sensitivityUnits"]
            ]
        else:
            unit_out = header["unit"]  # rad/m
            sensitivity = 1.0
            sensitivities_out = header["sensitivities"]
            sensitivityUnits_out = header["sensitivityUnits"]

    elif sensitivitySelect == -3 and userSensitivity is not None:  # use userdef sensitivity
        unit_out = _combine_units([header["unit"], userSensitivity["sensitivityUnit"]], "/")
        sensitivity = userSensitivity["sensitivity"]
        # sensitivity applied, output sensitivity set to 1
        sensitivities_out = np.ones((1, 1), dtype=np.float32)
        sensitivityUnits_out = [""]
    else:
        raise ValueError("Undefined sensitivitySelect")
    return sensitivity, unit_out, sensitivities_out, sensitivityUnits_out


def _fix_meta_back_compability(meta: dict, nSamples: int, nChannels: int):
    """
    Fix some back compability issues

    Parameters
    ----------
    meta : dict
        meta dict returned from load_DAS_file().
        Modified if not satisfy latest fileversion.
    nSamples : int
        samples in file
    nChannels : int
        channels in file
    """
    c = 2.99792458e8  # speed of light in vacuum

    if "experiment" not in meta["header"]:
        if "exp" in meta["header"]:
            meta["header"]["experiment"] = meta["header"].pop("exp")

    if "cableSpec" not in meta:
        meta["cableSpec"] = {
            "fiberOverLength": 1.0,
            "refractiveIndex": 1.4677,
            "zeta": 0.78,
        }

    if "dx" not in meta["header"] or "gaugeLength" not in meta["header"]:
        dx_fiber = meta["demodSpec"]["dTau"] * c / (2 * meta["cableSpec"]["refractiveIndex"])

        if "dx" not in meta["header"]:
            meta["header"]["dx"] = dx_fiber / meta["cableSpec"]["fiberOverLength"]
        if "gaugeLength" not in meta["header"]:
            meta["header"]["gaugeLength"] = meta["demodSpec"]["nDiffTau"] * dx_fiber

    if "dataScale" not in meta["header"]:
        meta["header"]["dataScale"] = (
            np.pi / 2**29 / meta["header"]["dt"] / meta["header"]["gaugeLength"]
        )
        meta["header"]["unit"] = "rad/m/s"
    if "spatialUnwrRange" not in meta["header"]:
        meta["header"]["spatialUnwrRange"] = (
            8 * np.pi / meta["header"]["dt"] / meta["header"]["gaugeLength"]
        )

    if "dimensionRanges" not in meta["header"]:
        meta["header"].update(dimensionRanges={})
        # time dimension
        meta["header"]["dimensionRanges"].update(
            dimension0={
                "min": 0,
                "max": nSamples - 1,
                "size": nSamples,
                "unitScale": meta["header"]["dt"],
            }
        )
        # position dimension
        try:
            procChain = meta["processingChain"] if "processingChain" in meta else meta["dimChain"]
            n = 0
            while "step-%d" % n in procChain:
                n += 1

            step = "step-%d" % (n - 1)
            cols = procChain[step]["cols"]

        except:
            cols = meta["demodSpec"]

        meta["header"]["dimensionRanges"].update(
            dimension1={
                "min": cols["roiStart"],
                "max": cols["roiEnd"],
                "size": (cols["roiEnd"] - cols["roiStart"] + 1) // cols["roiDec"],
                "unitScale": meta["header"]["dx"],
            }
        )

        meta["header"].update(
            dimensionSizes=np.r_[nSamples, nChannels],
            dimensionNames=["time", "distance"],
            dimensionUnits=["s", "m"],
            name="Phase rate per distance",
        )

    dim1 = meta["header"]["dimensionRanges"]["dimension1"]
    if not isinstance(dim1["min"], Iterable):
        dim1.update({key: np.r_[val] for key, val in dim1.items()})

    if "channels" not in meta["header"]:  # construct channels from roi
        roiChs = []
        # extract from roi in dimensionRanges
        dim1 = meta["header"]["dimensionRanges"]["dimension1"]
        for start, stop, size in zip(dim1["min"], dim1["max"], dim1["size"]):
            roiChs += [np.arange(start, stop + 1, (stop - start + 1) // size)]
        channels = np.sort(np.unique(np.concatenate(roiChs)))
        meta["header"]["channels"] = channels

    try:
        itu = int(meta["monitoring"]["Laser"]["itu"])
    except:
        itu = 51
    wavelength = c / (190e12 + itu * 1e11)

    if "sensitivity" in meta["header"]:  # should be plural
        meta["header"]["sensitivities"] = np.atleast_2d(meta["header"].pop("sensitivity"))
        meta["header"]["sensitivityUnits"] = np.atleast_1d(meta["header"].pop("sensitivityUnit"))

    if "sensitivities" not in meta["header"] or "NA" in meta["header"]["sensitivityUnits"]:
        meta["header"]["sensitivities"] = np.atleast_2d(
            4
            * np.pi
            * meta["cableSpec"]["zeta"]
            * meta["cableSpec"]["refractiveIndex"]
            / wavelength
        )
        meta["header"]["sensitivityUnits"] = np.atleast_1d("rad/m/strain")
    # bug in preliminary fileversion 8
    elif "sensitivityUnits" in meta["header"]:
        if len(meta["header"]["sensitivityUnits"]) and "unit" in meta["header"]:
            sensUnit_mod = list()
            for n, sensUnit in enumerate(meta["header"]["sensitivityUnits"].copy()):
                outUnit = _combine_units([meta["header"]["unit"], sensUnit], "/")
                # rad should not be in outunit, sensitvityunit is set to outunit
                if any([u == "rad" for u in re.findall(r"[\w']+", outUnit)]):
                    sensUnit = outUnit
                sensUnit_mod.append(sensUnit.replace("·", "*").replace("ε", "strain"))
            meta["header"]["sensitivityUnits"] = np.array(sensUnit_mod)

    if "unit" in meta["header"]:
        meta["header"]["unit"] = meta["header"]["unit"].replace("·", "*").replace("ε", "strain")
