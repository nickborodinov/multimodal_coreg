# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:16:14 2019

@author: s4k
"""

data_fname = r''

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
os.chdir('..')
import CoReg

cr = CoReg.CoReg(data_fname)
hf = cr.open_merged(data_fname)

maldi_grp = hf['MALDI']
sims_grp = hf['SIMS']
maldi_data = maldi_grp['intensity']
sims_data = sims_grp['intensity']
affine_matrix = maldi_grp['affine'][:]
maldi_mz_bar = maldi_grp['m_over_z'][:]
sims_mz_bar = sims_grp['m_over_z'][:]

maldi_keys = [k for k in maldi_grp.keys()]
sims_keys = [k for k in sims_grp.keys()]
print('MALDI keys:', maldi_keys)
print('MALDI data shape:', maldi_data.shape)
print('Affine matrix:', affine_matrix)
print('SIMS keys:', sims_keys)
print('SIMS data shape:', sims_data.shape)

%matplotlib inline

fig, ax = plt.subplots(2)
maldi_test_mz = maldi_mz_bar[int(len(maldi_mz_bar)/2)]
sims_test_mz = sims_mz_bar[int(len(sims_mz_bar)/2)]
maldi_sample_image = cr.get_maldi_image(maldi_test_mz)
sims_sample_image = cr.get_sims_image(sims_test_mz)
cr_sample_image = cr.get_coregistered_image(maldi_test_mz, sims_test_mz)