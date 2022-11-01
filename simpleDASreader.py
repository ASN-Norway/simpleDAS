# -*- coding: utf-8 -*-
"""
simpleDASreader v8.0 - OptoDAS raw file v8 reader for Python. 

    Copyright (C) 2021 Alcatecl Submarine Networks Norway AS,
    Vestre Rosten 77, 7075 Tiller, Norway, https://www.asn.com/

    This file is part of simpleDASreader.
 
    simpleDASreader is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    simpleDASreader is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np 
import datetime
import os 
import re
import warnings
from sympy import symbols,sympify
import pandas as pd
import h5pydict


def load_DAS_files(filepaths, chIndex=None,samples=None, sensitivitySelect=0,
                  integrate=True, unwr=False, spikeThr=None,
                  userSensitivity=None):
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
        Note: Decimation should be performed after cumsum or antialiasing.
              Decimation with the slice.step parameter is not recommended.
              
    sensitivitySelect: int
        Scale (divide) signal with one of the sensitivites given in
        filemeta['header']['sensitivities'][sensitivitySelect,0] with unit 
        filemeta['header']['sensitivityUnits'][sensitivitySelect], where
        filemeta can be extracted with filemeta = get_filemeta(filename).
        Default is sensitivitySelect = 0 which results in gives signal in unit strain
        Additional scalings that are not defined in filemeta['header']['sensitivities']:
        sensitivitySelect= -1 gives signal in unit rad/m
        sensitivitySelect= -2 gives signal in unit rad
        sensitivitySelect= -3 uses sensitivity defined in userSensitivity
    integrate: bool
        Integrate along time axis of phase data (default).
        If false, the output will be the time differential of the phase.
        
    unwr: bool
        Unwrap strain rate along spatial axis before time-integration. This may 
        be needed for strain rate amplitudes > 8π/(dt*gaugeLength*sensitivity). 
        If only smaller ampliudes are expected, uwr can be set to False.
        Default is True.
        
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
            The user defined sensitivity in unit sensitivityUnit
        sensitivityUnit: str
            The unit of the sensitivity. Should be rad/m/<wanted output unit>,
            e.g for temperature, sensitivityUnit = rad/m/K.
        
    Returns
    -------
    signal: pandas dataframe
        row labels: absolute timestamp. pandas timeseries object
        col labels: channel number
        A correponding numpy array can be extracted as signal.to_numpy()    
                
        signal.meta: dict
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
            filepaths: list of str
                full path to all loaded files in the dataset
                
 
    """
    loadstart = datetime.datetime.now()
        
    if isinstance(filepaths,str):
        filepaths = [filepaths]        
    
    if isinstance(samples,range):
        #warnings.warn('The use of range for samples is deprecated, use slice instead',DeprecationWarning)
        samples=slice(samples.start,samples.stop,samples.step)
    if isinstance(samples, int):
        samples=slice(0, samples)
    elif samples is None:
        samples = slice(None)                
    if samples.step is not None and samples.step>1 and integrate:
        warnings.warn('Time decimation before cumsum (samples.step=%d>1)\
               is not recommended.' %samples.step,UserWarning)
    
    
    if chIndex is None: 
        chIndex=slice(None)
        
    # load all files
    samples_read=0
    for n,filepath in enumerate(filepaths):
        with h5pydict.DictFile(filepath,'r') as f:
            nSamples,nChs = f['data'].shape
            if n==0:
                m = f.load_dict(skipFields=['data'])
                _fix_meta(m,nSamples,nChs) 
                if len(m['header']['channels'][chIndex])==0:
                    raise IndexError(f'chIndexes {chIndex} not in data (nChannels = {nChs})')
                samples_max = (nSamples+1)*len(filepaths)
                channels_max = len(range(nChs)[chIndex]) if isinstance(chIndex,slice) else len(chIndex)
                data = np.zeros((samples_max,channels_max),dtype=np.float32)
            data[samples_read:samples_read+nSamples,:] = f['data'][:,chIndex]
            samples_read += nSamples
    data = data[:samples_read,:]
    #select output unit
    if sensitivitySelect>=0:
        try:
            sensitivity = m['header']['sensitivities'][sensitivitySelect,0] 
            sensitivityUnit =  m['header']['sensitivityUnits'][sensitivitySelect]
        except KeyError:
            sensitivity = m['header']['sensitivity']
            sensitivityUnit =  m['header']['sensitivityUnit']                              
        except IndexError:
            print('SensitivitySelect index %d not found in file.' % sensitivitySelect)
            raise
    
        unit_out=_combine_units([m['header']['unit'],sensitivityUnit],'/')        
        # sensitivity applied, output sensitivity set to 1
        sensitivities_out = np.ones((1,1),dtype=np.float32) 
        sensitivityUnits_out = ['']

    elif sensitivitySelect==-1:
        unit_out=m['header']['unit'] #rad/m
        sensitvity = 1.0
        sensitivities_out = m['header']['sensitivities']
        sensitivityUnits_out = m['header']['sensitivityUnits']

    elif sensitivitySelect==-2:
        unit_out=_combine_units([m['header']['unit'],'m']) #rad
        sensitvity = 1.0/m['header']['gaugeLength']
        sensitivities_out = m['header']['sensitivities']*m['header']['gaugeLength']
        sensitivityUnits_out = [_combine_units([sensitivityUnit,'m']) for sensitivityUnit in m['header']['sensitivityUnits']]

    elif sensitivitySelect==-3 and userSensitivity is not None: # use userdef sensitivity
         unit_out=_combine_units([m['header']['unit'],userSensitivity['sensitivityUnit']],'/')
         sensitivity = userSensitivity['sensitivity']
         # sensitivity applied, output sensitivity set to 1
         sensitivities_out = np.ones((1,1),dtype=np.float32) 
         sensitivityUnits_out = ['']
    else:
        raise ValueError('Undefined sensitivitySelect')    
    scale = np.float32(m['header']['dataScale']/sensitivity) 
    
    signalnd  = np.asfarray(data[samples,:],dtype=np.float32)*scale
    
    if unwr or spikeThr or integrate:
        if m['header']['dataType']<3:
            raise ValueError('Options unwr, spikeThr or integrate can only be\
                             used with time differentiated phase data')
    if unwr and m['header']['spatialUnwrRange']:
        signalnd=unwrap(signalnd,m['header']['spatialUnwrRange']*sensitivity,axis=1) 
    
    if spikeThr is not None:
        signalnd[np.abs(signalnd)>spikeThr*sensitivity] = 0
    
    if integrate:
        unit_new=_combine_units([unit_out, 's'])        
        if not any([u == 's' for u in re.findall(r"[\w']+",unit_new)]): #check that s in not in unit after integraion
            signalnd=np.cumsum(signalnd,axis=0)*m['header']['dt']
            unit_out = unit_new
        else:
            warnings.warn('Data unit %s is not differentiated. Integration skipped.'%unit_out,UserWarning)
        
    
   

    # create timeaxis
    nstart = 0 if samples.start is None else samples.start
    tstart = datetime.datetime.utcfromtimestamp(m['header']['time']+nstart*m['header']['dt'])
    nSamples,nCh= signalnd.shape
    t = create_time_axis(tstart,nSamples,m['header']['dt'])
    
    #create pandas dataframe
    meta = {key:m['header'][key] for key in ['dt','dx','gaugeLength','experiment','dataType','dimensionRanges','dimensionUnits','dimensionNames','name']}    
    meta.update(fileVersion = m['fileVersion'],
                time=tstart,
                unit=unit_out,
                sensitivities=sensitivities_out,
                sensitivityUnits=sensitivityUnits_out,
                filepaths=filepaths)
    
    
    
    signal = DASDataFrame(
        signalnd, columns=m['header']['channels'][chIndex], index=t, meta=meta)
    files = [os.path.basename(f) for f in filepaths]
    print('Loaded files '+str(files)+'  in %.1f s'%(datetime.datetime.now()-loadstart).total_seconds())
    
    return signal


def find_DAS_files(experiment_path, start, duration, channels=None, datatype='dphi',
                   show_header_info=True, load_file_from_start=True):
    ''' 
    Returns of list of fullpath to DAS files starting at date and time <start> and 
    with duration <duration>.
    The filepaths will be returned given as <exppath>/<date>/<datatype>/<HHMMSS>.hdf5,
    where <date> and <HHMMSS> is extracted from <start>.

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
    '''
    
    dates_in_exp=[]
    exppath_has_dates= False
    for d in sorted(os.listdir(experiment_path)): #get all dates
        if os.path.isdir(os.path.join(experiment_path,d)):
            try:
                dates_in_exp.append('%8d'%int(d)) #get dates
                exppath_has_dates = True
            except:
                pass
    if len(dates_in_exp)==0: #dates found in exppath
        dates_in_exp.append('19000101') #placeholder for no dateinfo
        
    #get all files within selected start duration
    if isinstance(start,str):        
        datetime_start = str2datetime(start)
    else: #assume datetime object
        datetime_start = start
    if not isinstance(duration, datetime.timedelta):
        duration = datetime.timedelta(seconds=duration)

    datetime_stop = datetime_start + duration
        
    ffidpaths = []
    headerTimes = []
    for date_in_exp in dates_in_exp:
        date1= str2datetime(date_in_exp,True)
        if datetime_start.date() <= date1.date() <= datetime_stop.date(): 
            
            ffiddir = os.path.join(experiment_path, date_in_exp,datatype) if exppath_has_dates else experiment_path
            
            for ffid in sorted(os.listdir(ffiddir)):
                try:
                    ffidTime = str2datetime(date_in_exp+' '+os.path.splitext(ffid)[0])
                except:
                    continue
                if datetime_start-datetime.timedelta(seconds=11) < ffidTime <= datetime_stop+datetime.timedelta(seconds=1):
                    ffidpath = os.path.join(ffiddir,ffid)
                    with h5pydict.DictFile(ffidpath,'r') as f: #ffidTime may be off by upto 1s. Check headerTime
                        headerTime = datetime.datetime.utcfromtimestamp(float(f['header']['time'][()]))
                    if datetime_start-datetime.timedelta(seconds=10) < headerTime <= datetime_stop:
                        ffidpaths.append(ffidpath)
                        headerTimes.append(headerTime)
                    
    
                            
    if len(ffidpaths)>0:
        with h5pydict.DictFile(ffidpaths[0],'r') as f:
            m = f.load_dict(skipFields=['data'])
            nSamples,nChs = f['data'].shape
            _fix_meta(m,nSamples,nChs)

        header = m['header']
        #calculate sample range
        nSamples_to_be_read = int(duration.total_seconds()/header['dt']+.5)
        #headerTime = datetime.datetime.utcfromtimestamp(float(header['time']))
        headerTime = headerTimes[0]
        if datetime_start.year == 1900: # replace the placeholder date 19000101 with date from headerTime
            datetime_start = datetime.combine(headerTime.date(), datetime_start.time())
        
        if load_file_from_start:
            startSample = 0 
        else:            
            startSample = max(0, int((datetime_start-headerTime).total_seconds()/header['dt']+.5))
        samples = slice(startSample,startSample+nSamples_to_be_read)
        
        if show_header_info:
            print('-- Header info file: %s --'% os.path.basename(ffidpaths[0]))
            print('\tExperiment:            %s'% header['experiment'])
            print('\tFile timestamp:        %s'%headerTime.strftime("%Y-%m-%d %H:%M:%S"))
            print('\tType of data:          %s, unit: %s'% (header['name'],header['unit']))
            print('\tSampling frequency:    %.2f Hz' %(1.0/header['dt']))
            print('\tData shape:            %d samples x %d channels' %(nSamples,nChs))
            print('\tGauge length:          %.1f m' % header['gaugeLength'])
            print('\tSensitivities:         %s' %','.join('%.2e %s' % (sens, unit) for sens,unit in zip(header['sensitivities'][:,0],header['sensitivityUnits'])))
            try:
                dim1 = header['dimensionRanges']['dimension1']
                print('\tRegions of interest:   %s' % ','.join(['%d:%d:%d' %(start,stop, (stop+1-start)//size)                        
                    for start,stop,size in zip(dim1['min'],dim1['max'],dim1['size'])]))
            except:
                pass
            
        #calculate channel indices
        if isinstance(channels,slice):
            try:
                channels = np.arange(header['channels'][-1])[channels]        
            except KeyError:
                demodSpec = h5pydict.load(ffidpaths[0], field='demodSpec')
                channels = np.arange(demodSpec['roiEnd'])[channels]
        
        if channels is not None:
            # get the intersection between channels in file and requested channels
            header_channels = header['channels']

            channels_found, chIndex, _ = np.intersect1d(header_channels, channels, return_indices=True)
            if show_header_info:
                with np.printoptions(threshold=20):
                    print('\t%d channels requested:%s' % (len(channels),channels))
                    print('\t%d channels found:    %s' % (len(channels_found),channels_found))
        else:
            chIndex = None
    else:
        chIndex = None
        samples = None
        warnings.warn('simpleDASreader.find_DAS_files did not find any files' ,UserWarning)
    if show_header_info: print('-----------------------------------')    
    return ffidpaths, chIndex, samples

def get_filemeta(filepath,metaDetail=1):
    '''
    

    Parameters
    ----------
    filepath : str
        DAS file from which to extract meta data 
    metaDetail: int
            Selects how much meta information from the file is outputed to filemeta
            1 => Load only metadata needed for DAS data interpretation (default)
            2 => Load all metadata

    Returns
    -------
   filemeta: dict
       optional extraction of meta information as is in the file    
       The number of fields in filemeta depends on the metadetails input.
       Metadata fields relevant for end users are described in
       'OPT-TN1287 OptoDAS HDF5 file format description for externals.pdf'
    '''        
    with h5pydict.DictFile(filepath,'r') as f:
        m = f.load_dict(skipFields=['data']) 
        if metaDetail==1:
            mon=m['monitoring']
            return dict(fileVersion = m['fileVersion'],
                      header     = m['header'],
                      timing     = m['timing'],
                      cableSpec  = m['cableSpec'],
                      monitoring = dict(Gps = mon['Gps'],
                                        Laser=dict(itu=mon['Laser']['itu'])))
        else:
            return m
        
def save_to_DAS_file(signal,filepath=None,datatype='processed'):
    '''
    Save a DASDataframe to file in a format that is readable for load_DAS_files.
    Note that just a small subset of the meta information from original files are 
    kept.
    Useful when a preprocessing step is required where the intermediate result
    should be saved.

    Parameters
    ----------
    signal : DASDataframe
        The dataframe to be saved
    filepath : str, optional
        The filepath to which the data is saved.
        If None, the data is save to a file derived from the start time 
        signal.meta['time'] and datatype in the path with datatype input parameter.
        E.g. a file read as /raid1/exp/exp1/20200101/dphi/120000.hdf5 will be
        saved as /raid1/exp/exp1/20200101/processed/120000.hdf5 
        The default is None.
    datatype : str, optional
        A sub-directory desnoting the data. The default is 'processed'.

    Returns
    -------
    filepath: str
        filepath to generated file

    '''
    if filepath is None:
        timestr = signal.meta['time'].strftime('%H%M%S')
        filename = timestr +'.hdf5'
        filepath_orignal = signal.meta['filepaths'][0]
        path, filename_original = os.path.split(filepath_orignal)
        exp_path,datatype_original = os.path.split(path)
        path_new = os.path.join(exp_path,datatype)
        filepath = os.path.join(path_new,filename)
    
    path,filename = os.path.split(filepath)       
    if not os.path.exists(path):
        os.makedirs(path)
    print('Saving to file: ' + filepath)
    
    utctimestamp = signal.meta['time'].replace(tzinfo=datetime.timezone.utc).timestamp()
    signal._set_dimensionRange()
    header = signal.meta.copy()
    header.update(channels = signal.columns,dataScale = np.float32(1.0),
                  time=utctimestamp,spatialUnwrRange=np.float32(0.0),
                  unit=header['unit'].encode(),sensitivityUnits = [s.encode() for s in header['sensitivityUnits']]
                  )
    fileVersion = header['fileVersion']
    del header['fileVersion']
    h5pydict.save(filepath,dict(fileVersion=fileVersion,header=header,data=np.asfarray(signal,np.float32)))   
    
    return filepath
    
    
class DASDataFrame(pd.DataFrame):
    '''
    Specialized DataFrame to include DAS meta info
    '''    
    _metadata = ["meta"]
    
    @property
    def _constructor(self):
        return DASDataFrame
    
    def __init__(self,data=None, index=None, columns=None, dtype=None, copy=None,meta=dict()):
        '''
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
    
        '''    
        super().__init__(data, index, columns, dtype, copy)
        self.meta = meta.copy()
        if len(meta):
            self._set_dimensionRange()
        
    def _set_dimensionRange(self):
        '''
        Compute the meta|dimensionRanges
        '''
                
        chs = np.array(self.columns)
        d2chs = np.diff(chs,n=2) #=0 within a roi
        d2chs[np.r_[False,np.logical_and(d2chs[:-1],d2chs[1:])]]=0 #remove subseqent indexes due to double differensiation
        stopind = np.r_[np.where(d2chs!=0)[0]+1,len(chs)-1] #where to split roi and create a new
        startind = np.r_[0,stopind[:-1]+1]
        self.meta['dimensionRanges']=dict(
            dimension0 = {'max':self.shape[0]-1,'min':0,
                          'size':self.shape[0],'unitScale':self.meta['dt']},
            dimension1 = {'max':chs[stopind],'min':chs[startind],
               'size':np.array([len(chs[b:e])+1 for b, e in zip(startind,stopind)]),
               'unitScale':self.meta['dx']})
        
        



def str2datetime(sdt,dateonly=False):
    '''
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

    '''
    
    if isinstance(sdt, datetime.datetime):
        return sdt
    
    if dateonly:
        dt = datetime.datetime.strptime(sdt,'%Y%m%d')    
    else:
        try:
            dt = datetime.datetime.strptime(sdt,'%H%M%S')
        except:
            dt = datetime.datetime.strptime(sdt,'%Y%m%d %H%M%S')    
    return dt


def convert_index_to_rel_time(signal):
    '''
    Convert the pandas index from datatime to float seconds relative to first sample

    Parameters
    ----------
    signal : pandas dataframe
        The dataframe to be modified
    inplace: bool
        Do inplace replacement of data frame else return a copy


    '''
    signal['t'] = (signal.index-signal.index[0]).total_seconds() # seconds from start of data
    signal.set_index('t',inplace=True) # set t as the new index

def create_time_axis(tstart,nSamples,dt):
    """
    Create a time axis with nSamples from tstart with sampling interval dt

    Parameters
    ----------
    tstart : datetime
        start time of axis
    dt : float 
        sampling interval in s

    Returns
    -------
    tstart : DateTimeIndex
        start time of axis
    """
    
    return tstart + pd.timedelta_range(start='0s',periods=nSamples,freq = '%dN'%int(dt*1e9+.5))
    
    
def unwrap(phi, wrapStep=2*np.pi, axis=-1):
    """
    Unwrap phase phi by changing absolute jumps greater than wrapStep/2 to
    their wrapStep complement along the given axis. By default (if wrapStep is
    None) standard unwrapping is performed with wrapStep=2*np.pi.
    
    (Note: np.unwrap in the numpy package has an optional discont parameter
    which does not give an expected (or usefull) behavior when it deviates
    from default. Use this unwrap implementation instead of the numpy
    implementation if your signal is wrapped with discontinuities that deviate
    from 2*pi.)
    """
    scale = 2*np.pi/wrapStep
    return (np.unwrap(phi*scale, axis=axis)/scale).astype(phi.dtype)

def wrap(x, wrapStep=2*np.pi):
    """ 
    Inverse of the unwrap() function.
    Wraps x to the range [-wrapStep/2, wrapStep/2>. 
    """
    if wrapStep>0:
        return (x + wrapStep/2)%wrapStep - wrapStep/2
    else:
        return x
    

        
    
def _combine_units(units, operator='*'):
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
    knownUnits = ('rad','dt','strain','W','NormCts','m','K','s','Hz','N','Pa','J')
    usyms = symbols(knownUnits)
    udict = {u:s for u,s in zip(knownUnits,usyms)}
    try:
        units = [u.replace('·','*').replace('ε','strain') for u in units]
    except:
        pass
    combinedUnit=sympify(units[0], udict)
    for unit in units[1:]:
        if len(unit)>0:
            if operator=='/': combinedUnit/=sympify(unit,udict)
            if operator=='*': combinedUnit*=sympify(unit,udict)
            
    return str(combinedUnit)


def _fix_meta(meta,nSamples,nChannels):
    """
    Fix some back compability issues 
    
    Parameters
    ----------
    meta : dict
        meta dict returned from load_DAS_file().
    """
    c=2.99792458e8 #speed of light in vacuum        
    
    if 'experiment' not in meta['header']:
        if 'exp' in meta['header']:            
            meta['header']['experiment'] = meta['header'].pop('exp')
            
    if not 'cableSpec' in meta:
          meta['cableSpec']={'fiberOverLength':1.0,
                             'refractiveIndex':1.4677,
                            'zeta':0.78}

    if 'dx' not in meta['header'] or 'gaugeLength' not in meta['header']:
        dx_fiber = meta['demodSpec']['dTau']*c/(2*meta['cableSpec']['refractiveIndex'])

        if 'dx' not in meta['header']:
            meta['header']['dx']= dx_fiber/meta['cableSpec']['fiberOverLength']
        if 'gaugeLength' not in meta['header']:
            meta['header']['gaugeLength']= meta['demodSpec']['nDiffTau']*dx_fiber

    if 'dataScale' not in meta['header']:
        meta['header']['dataScale']=np.pi/2**29/meta['header']['dt']/meta['header']['gaugeLength']
        meta['header']['unit']='rad/m/s'
    if 'spatialUnwrRange' not in meta['header']:
        meta['header']['spatialUnwrRange']=8*np.pi/meta['header']['dt']/meta['header']['gaugeLength']
    
 
    if 'dimensionRanges' not in meta['header']:
        meta['header'].update(dimensionRanges = {})
        #time dimension
        meta['header']['dimensionRanges'].update(dimension0 = \
                        {'min':0,'max':nSamples-1,
                         'size': nSamples,
                         'unitScale':meta['header']['dt']})
        #position dimension
        try:
            procChain = meta['processingChain'] if 'processingChain' in meta else meta['dimChain']            
            n=0
            while 'step-%d'%n in procChain:
                n+=1
                
            step = 'step-%d'%(n-1)
            cols = procChain[step]['cols']
            
        except:
            cols = meta['demodSpec']
            
            
        meta['header']['dimensionRanges'].update(dimension1 = \
                         {'min':cols['roiStart'],'max':cols['roiEnd'],
                          'size': (cols['roiEnd']-cols['roiStart']+1)//cols['roiDec'],
                          'unitScale':meta['header']['dx']})
            
        meta['header'].update(dimensionSizes = np.r_[nSamples,nChannels],
                              dimensionNames = ['time','distance'],
                              dimensionUnits = ['s','m'],
                              name = 'Phase rate per distance')
    
    if 'channels' not in meta['header']: #construct channels from roi
        roiChs = []
        dim1 = meta['header']['dimensionRanges']['dimension1']  #extract from roi in dimensionRanges       
        for start,stop,size in zip(dim1['min'],dim1['max'],dim1['size']):  #extract from roi in dimensionRanges       
            roiChs+= [np.arange(start,stop+1,(stop-start+1)//size)]
        channels =  np.sort(np.unique(np.concatenate(roiChs)))
        meta['header']['channels']=channels
     
     
    try:
        itu = int(meta['monitoring']['Laser']['itu'])
    except:
        itu = 51        
    wavelength=c/(190e12+itu*1e11)            
    
    if 'sensitivity' in meta['header']: # should be plural
        meta['header']['sensitivities'] = np.atleast_2d(meta['header'].pop('sensitivity'))
        meta['header']['sensitivityUnits'] = np.atleast_1d(meta['header'].pop('sensitivityUnit'))
    
    if 'sensitivities' not in meta['header'] or 'NA' in meta['header']['sensitivityUnits']:
        meta['header']['sensitivities'] = np.atleast_2d(4*np.pi*meta['cableSpec']['zeta']\
                                        *meta['cableSpec']['refractiveIndex']/wavelength)
        meta['header']['sensitivityUnit']=np.atleast_1d('rad/m/strain')
    elif 'sensitivityUnits' in meta['header'] and meta['header']['sensitivityUnits'][0] == 'strain/s': #bug in preliminary fileversion 8
        meta['header']['sensitivityUnits'] = ['rad/(strain*m)']
        
    if 'unit' in meta['header']:
        meta['header']['unit']=meta['header']['unit'].replace('·','*').replace('ε','strain')
    if 'sensitivityUnits' in meta['header']:
        meta['header']['sensitivityUnits'] = [u.replace('·','*').replace('ε','strain') for u in meta['header']['sensitivityUnits']]
    
