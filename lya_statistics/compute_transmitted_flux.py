import os, sys
import numpy as np
import h5py as h5
root_dir = os.getcwd() + '/'
subDirectories = [x[0] for x in os.walk(root_dir)]
sys.path.extend(subDirectories)
from tools import *
from load_skewers import load_skewers_multiple_axis
from spectra_functions import Compute_Skewers_Transmitted_Flux
from flux_power_spectrum import Compute_Flux_Power_Spectrum


base_dir = '/gpfs/alpine/ast175/proj-shared/using_cholla/' 
input_dir  = base_dir + 'simulations/256_50Mpc/skewers_files/'
output_dir = base_dir + 'simulations/256_50Mpc/transmitted_flux_files/'
create_directory( output_dir )

# Box parameters
Lbox = 50000.0 #kpc/h
box = {'Lbox':[ Lbox, Lbox, Lbox ] }

axis_list = [ 'x', 'y', 'z' ]
n_skewers_list = [ 'all', 'all', 'all']
skewer_ids_list = [ 'all', 'all', 'all']
field_list = [  'HI_density', 'los_velocity', 'temperature' ]


skewer_dataset = Load_Skewers_File( file_indx, input_dir, axis_list=axis_list, fields_to_load=field_list )
current_z = skewer_dataset['current_z']

