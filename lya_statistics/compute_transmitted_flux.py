import os, sys
import numpy as np
import h5py as h5
root_dir = os.path.dirname(os.getcwd()) + '/'
subDirectories = [x[0] for x in os.walk(root_dir)]
sys.path.extend(subDirectories)
from tools import *
from load_data import Load_Skewers_File, load_analysis_data
from spectra_functions import Compute_Skewers_Transmitted_Flux
from flux_power_spectrum import Compute_Flux_Power_Spectrum


base_dir = '/gpfs/alpine/ast175/proj-shared/using_cholla/' 
input_dir  = base_dir + 'simulations/256_50Mpc/skewers_files/'
output_dir = base_dir + 'simulations/256_50Mpc/transmitted_flux_files/'
create_directory( output_dir )


axis_list = [ 'x', 'y', 'z' ]
n_skewers_list = [ 'all', 'all', 'all']
skewer_ids_list = [ 'all', 'all', 'all']
field_list = [  'HI_density', 'los_velocity', 'temperature' ]

file_id = 30
skewer_dataset = Load_Skewers_File( file_id, input_dir, axis_list=axis_list, fields_to_load=field_list )

#Box parameters
Lbox = skewer_dataset['Lbox']#kpc/h
box = {'Lbox':[ Lbox, Lbox, Lbox ] }

# Cosmology parameters
cosmology = {}
cosmology['H0'] = skewer_dataset['H0']
cosmology['Omega_M'] = skewer_dataset['Omega_M']
cosmology['Omega_L'] = skewer_dataset['Omega_L']
cosmology['current_z'] = skewer_dataset['current_z']

skewers_data = { field:skewer_dataset[field] for field in field_list }
data_Flux = Compute_Skewers_Transmitted_Flux( skewers_data, cosmology, box  )


#Compute the flux power spectrum
data_ps = Compute_Flux_Power_Spectrum( data_Flux, d_log_k=0.1 )
k_vals = data_ps['k_vals']
skewers_ps = data_ps['skewers_ps']
ps_mean = data_ps['mean']

