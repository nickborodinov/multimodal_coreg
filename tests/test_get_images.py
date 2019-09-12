# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:44:18 2019

@author: s4k
"""
import os
os.chdir(r'C:\Users\s4k\Documents\Python\sims-maldi_co-registration')
from CoReg import CoReg

merged_fname = r"F:\dump dir\merged.h5"
cr = CoReg(merged_fname)

maldi_target_masses = [726.0435, 736.1267,752.8608]
sims_target_masses = [41.15,55.2,91.26]
target_mass_pairs = [(maldi_target_masses[i], sims_target_masses[i]) for i in range(len(maldi_target_masses))]

maldi_maps = [cr.get_maldi_image(maldi_mass) for (maldi_mass, sims_mass) in target_mass_pairs]
sims_maps = [cr.get_sims_image(sims_mass) for (maldi_mass, sims_mass) in target_mass_pairs]
cr_maps = [cr.get_coregistered_image(maldi_mass, sims_mass) for (maldi_mass, sims_mass) in target_mass_pairs]