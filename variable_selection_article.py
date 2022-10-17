#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:51:23 2022

@author: frem
"""

import sys
import numpy as np
import clumpy

#%%
palette = clumpy.load_palette("data/palette.qml")
lul2003 = clumpy.open_layer("data/luc_2003.tif",
                            kind='land_use',
                            dtype=np.int16)

lul2009 = clumpy.open_layer("data/luc_2009.tif",
                            kind='land_use',
                            dtype=np.int16)

#%%
lul2009.display(center=(1000,1000), 
                window=(100,100), 
                palette=palette)

#%%
dem = clumpy.open_layer("data/dem.tif",
                        kind='feature')
slope = clumpy.open_layer("data/slope.tif",
                          kind='feature',
                          bounded='left')

#%%
u = 6
list_v = [2,6,7]
J = lul2003.get_J(u)
J, V = lul2009.get_V(J, final_states=list_v)
Z = lul2003.get_X(J, features=[dem, slope, 2, 3, 7])
idx = Z[:,0]>0
Z = Z[idx]
V = V[idx]

n, d = Z.shape

features_names = ['1- elevation (m)',
                  '2- slope (°)',
                  '3- dist. to urban (m)',
                  '4- dist. to eco. act. (m)',
                  '5- dist. to forest (m)']

print(np.unique(V, return_counts=True))

#%%
import pandas as pd
from matplotlib import pyplot as plt

#%%
import time
epsilon = 0.08
cramer_mrmr = clumpy.feature_selection.CramerMRMR(initial_state=u,
                                                  V_gof_min = 0.1, 
                                                  V_toi_max = 0.2,
                                                  epsilon=epsilon,
                                                  alpha=0.9,
                                                  approx='mean',
                                                  kde_method=False,
                                                  features_names=features_names,
                                                  k_shift=1)
st = time.time()
cramer_mrmr.fit(Z=Z, 
                transited_pixels=V==2)
t = time.time()-st

cramer_mrmr_kde = clumpy.feature_selection.CramerMRMR(initial_state=u,
                                                      V_gof_min = 0.1, 
                                                      V_toi_max = 0.2,
                                                      epsilon=epsilon,
                                                      alpha=0.9,
                                                      approx='mean',
                                                      kde_method=True,
                                                      kde_params={'kernel':'gaussian'},
                                                      features_names=features_names,
                                                      k_shift=1)
cramer_mrmr_kde.fit(Z=Z, 
                    transited_pixels=V==2,
                    bounds=None)
print(cramer_mrmr_kde._cols_support)

#%%
cramer_mrmr.plot(path_prefix='plot_agr_urb_without_kde', extension='pdf')

cramer_mrmr_kde.plot(path_prefix='plot_agr_urb', extension='pdf')

#%%
tab=cramer_mrmr_kde.tex_table(features_names)
print(tab)

#%%
cramer_mrmr_kde = clumpy.feature_selection.CramerMRMR(initial_state=u,
                                                  V_gof_min = 0.1, 
                                                  V_toi_max = 0.2,
                                                  epsilon=0.15,
                                                  alpha=0.9,
                                                  approx='mean',
                                                  kde_method=True,
                                                  kde_params={'kernel':'gaussian'},
                                                  features_names=features_names,
                                                  k_shift=1)
cramer_mrmr_kde.fit(Z, transited_pixels=V==7)
print(cramer_mrmr_kde._cols_support)
#%%
cramer_mrmr_kde.plot(path_prefix='plot_agr_for', extension='pdf')

#%%
tab=cramer_mrmr_kde.tex_table(features_names, k_shift=1)
print(tab)

#%%
cramer_mrmr_kde_2 = clumpy.feature_selection.CramerMRMR(initial_state=u,
                                                      V_gof_min = 0.1, 
                                                      V_toi_max = 0.2,
                                                      epsilon=0.08,
                                                      alpha=0.9,
                                                      approx='mean',
                                                      kde_method=True,
                                                      kde_params={'kernel':'gaussian'})
cramer_mrmr_kde_7 = clumpy.feature_selection.CramerMRMR(initial_state=u,
                                                      V_gof_min = 0.12, 
                                                      V_toi_max = 0.18,
                                                      epsilon=0.15,
                                                      alpha=0.9,
                                                      approx='mean',
                                                      kde_method=True,
                                                      kde_params={'kernel':'gaussian'})
fs = clumpy.feature_selection.FeatureSelectors({2:cramer_mrmr_kde_2,
                                                7:cramer_mrmr_kde_7})
st = time.time()
fs.fit(Z, V, bounds=None)
print(time.time()-st)

#%%
Z_fs = fs.transform(Z)

#%%
plot_4gof(cramer_mrmr_kde_7, path=None)

#%%
tab = cramer_mrmr_kde_2.tex_table(features_names, k_shift=1)
print(tab)

#%%
tab = cramer_mrmr_kde_7.tex_table(features_names, k_shift=1)
print(tab)

