# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:26:27 2019

@author: s4k
"""

import pickle
import h5py
import cv2
import numpy as np
import pyUSID as usid

class DataModel:
    
    def __init__(self):
        self.nmf = None                 #NMF model for reconstruction
        self.cca = None                 #CCA model for reconstruction
        self.src_data = None            #h5 dataset corresponding to input dataset
        self.dst_data = None            #USIDataset corresponding to output dataset
        return
    
    def _set_src_(self, src_grp, gridsize):
        self.src_data = src_grp['Raw_data']
        self.src_cal = (src_grp.attrs['SF'], src_grp.attrs['SF'])
        self.src_gridsize = (gridsize, gridsize)
        return
    
    def _set_dst_(self, dst_grp):
        if 'Main' not in dst_grp:
            raise ValueError('EC001')
        self.dst_data = usid.USIDataset(dst_grp['Main'])
        self.dst_bar = dst_grp['Spectroscopic_Values'][:]
        self.dst_shape = tuple(self.dst_data.pos_dim_sizes)
        return
    
    def _set_nmf_(self, nmf, train_peaklist):
        self.nmf = nmf
        self.train_peaks = train_peaklist
        return

class Translator:
    
    def __init__(self):
        self.sims_fname = None
        self.model = None
        return
    
    def tof_to_mass(self, tof):
        SF, K0 = self.model.src_cal
        return ((tof+K0+71*40)/SF)**2
    
    def _assign_sims_datafile_(self, fname):
        self.sims_fname = fname
        return
    
    def _assign_model_(self, cca_fname, nmf_fname):
        
        
        return
    
    def translate(self):
        
        def find_grid(blocksize=1000000):
            grid_bounds = []
            offset = (0,0)
            bound = (offset[0]+self.model.src_gridsize[0], offset[1]+self.model.src_gridsize[1])
            bi = 0
            low = 0
            high = blocksize
            while low <= len(self.model.src_data):
                try: block = self.model.src_data[low:high, 0:2]
                except IndexError: block = self.model.src_data[low:, 0:2]
                x = block[:, 0] - offset[0]
                y = block[:, 1] - offset[1]
                coords = set(zip(x, y))
                for idx, (a, b) in enumerate(coords):
                    if a > bound[0]:
                        grid_bounds.append((bi * blocksize) + idx)
                        offset[0] += self.model.src_gridsize[0]
                        bound = (offset[0]+self.model.src_gridsize[0], offset[1]+self.model.src_gridsize[1])
                    elif b > bound[1]:
                        grid_bounds.append((bi * blocksize) + idx)
                        offset[1] += self.model.src_gridsize[1]
                        bound = (offset[0]+self.model.src_gridsize[0], offset[1]+self.model.src_gridsize[1])
            return grid_bounds
        
        def build_spectra(low, high):
            spectra = {}
            data = self.model.data[low:high]
            
        
        self.model.grid_bounds = find_grid()
        return
                
            
        
    
    
