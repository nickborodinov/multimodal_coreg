# SIMS-MALDI MSI Co-Registration

Module to assist in co-registration of multimodal datasets. 

Written by Steven T King, Matthias Lorenz, and Nikolay Borodinov for the Center for Nanophase Materials Science at Oak Ridge National Laboratory (2019)

## About

This module provides methods for the semi-automated alignment of multimodal imaging datasets. Originally designed for the alignment and analysis of MALDI-TOF and ToF-SIMS imaging mass spectrometry data, this module has been expanded to allow for the inclusion of arbitrarily many image datasets in a singular data structure. The images included in each dataset may then be rapidly and bidirectionally coregistered with any other included dataset.

## Version 0.4.0

### Changelog

* Merged output file now incorporates USID formatting for dataset storage and internal HDF5 file heirarchy. See the [USID documentation](https://pycroscopy.github.io/USID/index.html) for more information on the USID format.
* Generalized data import tools to allow for addition of arbitrarily many datasets to a single USID-formatted data file.

## Installation

The module was written using the Anaconda distribution of Python 3.6 (5/22/2019). As such, most of the module's dependencies are included in the standard Anaconda library. A full list of dependencies is given below. 

In addition to the standard Anaconda library, the module requires the installation of the cv2, h5py, and pyUSID packages. h5py and pyUSID may be installed via pip: 

'''python
pip install h5py, pyUSID
'''

cv2 cannot currently be installed through the PyPI, and must instead be installed through conda:

'''
conda install -c conda-forge opencv 
'''

#### Requirements

##### Included with Anaconda

* Numpy
* Matplotlib

##### Not included with Anaconda

* h5py
* pyUSID
* cv2

## Usage

Co-registration starts with identification of fiduciary markers within MALDI dataset.

* Open the "BadBrains_classWrapped" iPython Notebook
* Work through the notebook, 
