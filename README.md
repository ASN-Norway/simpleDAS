# simpleDAS


simpleDAS is python library that allow for simple reading, processing and saving of the ASN OptoDAS file format.

The primary data container used is pandas dataframe that is extended to include meta data. The dataframe is a representation of a 2D numpy that allows for user specified labels on each column and row.  

SimpleDAS outputs a dataframe with channel numbers as column labels and timestamps as the row labels. In addition the returned dataframe includes additional metainformation obtained from the read files.

Note that the namespace has been changed from `simpleDASreader` to `simpledas`.


## Install

The project structure has been modified in v 8.2 to satisfy the requirements for requirements for python packaging with [hatch](https://hatch.pypa.io/latest/). The code is moved to subdirectory src/simpledas.

We recommend to install `simpledas` in a [miniconda enviroment](https://docs.conda.io/projects/miniconda/en/latest/) or [python virtual enviroment](https://docs.python.org/3/library/venv.html). `simpledas` requires python>=3.8.  
The simplest way to import is to install with `pip` directly from git:  
`
pip install git+https://github.com/ASN-Norway/simpleDAS.git
`  

This code will install the simpledas along with other python packages in your python enviroment.

## Usage

### SimpleDAS
See the [docs](./doc/simpleDAS%20description.ipynb) for descriptions how to use simpledas reader and view the [examples](./examples)

### print_hdf5

print_hdf5 is command line executable which can be runned from terminal after installation.

Print a summary of the hdf5 file when a file is used as first argument.
```
print_hdf5 /raid1/fsi/exps/MixerCalibration/20220608/adcdec/074159.hdf5
```

print only the hdf5 group if the "g" option is used.
```
 print_hdf5 /raid1/fsi/exps/MixerCalibration/20220608/adcdec/074159.hdf5 -g monitoring/Gps

```
Example output
```
Open file: /raid1/fsi/exps/MixerCalibration/20220608/adcdec/074159.hdf5
Load dictinary from field "monitoring/Gps" within /raid1/fsi/exps/MixerCalibration/20220608/adcdec/074159.hdf5:
gpsPosE (float64):            1022.36
gpsPosH (float64):            205.0
gpsPosN (float64):            6321.76
gpsStatus (uint32):           0

```
