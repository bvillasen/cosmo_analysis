import sys, os
import numpy as np
root_dir = os.path.dirname(os.getcwd()) + '/'
sys.path.append( root_dir + 'tools')
from tools import print_line_flush, print_progress
from io_tools import load_cholla_snapshot_distributed
from spectra_functions import Compute_Skewers_Transmitted_Flux

# The input directory where the snapshots are located 
data_directory = '/lustre/user/bvillase/benchmarks/cholla/cosmology/1024_50Mpc'
input_directory = data_directory + '/snapshot_files/'

# The snapshot id
snapshot_id = 1

# The type of data to load. ['hydro', 'particles', 'gravity']
data_type = 'hydro'

# The fields to load 

# Load 4x4 area across the z-axis, that will be converted into 16 skewers
subgrid = [[100,104], [56,60], [0,-1]]
precision = np.float64

# Since the skewers are along the z axis, we need momentum_z to compute LOS velocity
fields_to_load = ['density', 'HI_density', 'temperature',  'momentum_z', ]

# Load the sub-volume data
data_hydro = load_cholla_snapshot_distributed(data_type, fields_to_load,  snapshot_id, input_directory, 
                                             precision=precision, subgrid=subgrid )

#  Load the dark matter density
fields_to_load = ['density']
data_dm = load_cholla_snapshot_distributed('particles', fields_to_load,  snapshot_id, input_directory, 
                                            precision=precision, subgrid=subgrid )
dm_density = data_dm['density']


# Compute LOS velocity
data_hydro['los_velocity'] = data_hydro['momentum_z'] / data_hydro['density'] 

# Cosmology parameters
cosmology = {}
cosmology['H0']        = data_hydro['H0']
cosmology['Omega_M']   = data_hydro['Omega_M']
cosmology['Omega_L']   = data_hydro['Omega_L']
cosmology['current_z'] = data_hydro['Current_z']

#Box parameters
Lbox = data_hydro['domain'] #kpc/h
box = {'Lbox':Lbox }


field_list = [  'HI_density', 'los_velocity', 'temperature' ]
# Flatten the 3D sub-volume into a 2D array of skewers
data_skewers = {}
for field in field_list:
  data_field = data_hydro[field]
  nx, ny, nz = data_field.shape
  n_skewers = nx * ny
  skewers_length = nz
  field_skewers = np.zeros( [n_skewers, skewers_length], dtype=precision )
  for i in range(nx):
    for j in range(ny):
      field_skewers[i*ny+j,:] = data_field[i,j,:]
  data_skewers[field] = field_skewers


data_Flux = Compute_Skewers_Transmitted_Flux( data_skewers, cosmology, box  )
skewers_vel_Hubble = data_Flux['vel_Hubble'] 
skewers_transmitted_flux = data_Flux['skewers_Flux']
skewers_tau = data_Flux['skewers_tau']



