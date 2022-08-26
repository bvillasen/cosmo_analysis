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


base_dir = '/gpfs/alpine/ast175/proj-shared/using_cholla/simulations/' 
input_dir  = base_dir + '256_50Mpc/snapshot_files/'
output_dir = base_dir + 'figures/'
create_directory( output_dir ) 

input_dir = '/gpfs/alpine/ast175/scratch/bvilasen/cosmo_sims/256_50Mpc/snapshot_files/'


data_type = 'particles'
fields = [ 'density' ]

Lbox = 50000.0    #kpc/h
n_cells = 256
box_size = [ Lbox, Lbox, Lbox ]
grid_size = [ n_cells, n_cells, n_cells ] #Size of the simulation grid
precision = np.float64

n_bins = 15
L_Mpc = Lbox * 1e-3    #Mpc/h
nx, ny, nz = grid_size
dx, dy, dz = Lbox/nx, Lbox/ny, Lbox/nz

snapshots = range(5)
fields = [ 'density' ]

data_type = 'particles'

ps_dm = {}
for snap_id in snapshots:
  snap_data = load_snapshot_data_distributed( data_type, fields,  snap_id, input_dir,  box_size, grid_size, precision  )
  z = snap_data['Current_z']
  density = snap_data['density']
  # power_spectrum, k_vals, n_in_bin = get_power_spectrum( density, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_bins )
  # power_spectrum_all[snap_id][data_type][data_id] = { 'z': z, 'k_vals': k_vals, 'power_spectrum':power_spectrum }

