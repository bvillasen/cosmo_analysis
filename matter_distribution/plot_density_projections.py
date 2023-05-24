import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
root_dir = os.path.dirname(os.getcwd()) + '/'
subDirectories = [x[0] for x in os.walk(root_dir)]
sys.path.extend(subDirectories)
from tools import *
from load_data import load_snapshot_data_distributed
from power_spectrum_functions import get_power_spectrum


Lbox = 50000.0    #kpc/h
n_cells = 512
L_Mpc = int(Lbox/1000)

base_dir = f'/home/bvillase/tests/cosmo_sims/{n_cells}_{L_Mpc}Mpc/' 
input_dir  = base_dir + 'snapshot_files/'
output_dir = base_dir + 'figures/'
create_directory( output_dir ) 


data_type = 'particles'
fields = [ 'density' ]

box_size = [ Lbox, Lbox, Lbox ]
grid_size = [ n_cells, n_cells, n_cells ] #Size of the simulation grid
precision = np.float64

# snapshots = range(5
fields = [ 'density' ]


snap_id = 339
snap_data = load_snapshot_data_distributed( data_type, fields, snap_id, input_dir,  box_size, grid_size, precision  )
z = snap_data['Current_z']
density = snap_data['density']
dens_mean = density.mean()
proj = (density**2).sum() / density.sum() 

