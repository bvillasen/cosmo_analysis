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


base_dir = '/gpfs/alpine/ast175/proj-shared/using_cholla/' 
input_dir  = base_dir + 'simulations/256_50Mpc/snapshot_files/'
output_dir = base_dir + 'figures/'
create_directory( output_dir ) 


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
dx, dy, dz = L_Mpc/nx, L_Mpc/ny, L_Mpc/nz

snapshots = range(1)
fields = [ 'density' ]

# Compute DM power spectrum
data_type = 'particles'
ps_dm = {}
for snap_id in snapshots:
  snap_data = load_snapshot_data_distributed( data_type, fields,  snap_id, input_dir,  box_size, grid_size, precision  )
  z = snap_data['Current_z']
  density = snap_data['density']
  power_spectrum, k_vals, n_in_bin = get_power_spectrum( density, L_Mpc, nx, ny, nz, dx, dy, dz,  n_kSamples=n_bins )
  ps_dm[snap_id] = { 'z': z, 'k_vals': k_vals, 'power_spectrum':power_spectrum }


# Compute Hydro power spectrum
data_type = 'hydro'
ps_hydro = {}
for snap_id in snapshots:
  snap_data = load_snapshot_data_distributed( data_type, fields,  snap_id, input_dir,  box_size, grid_size, precision  )
  z = snap_data['Current_z']
  density = snap_data['density']
  power_spectrum, k_vals, n_in_bin = get_power_spectrum( density, L_Mpc, nx, ny, nz, dx, dy, dz,  n_kSamples=n_bins )
  ps_hydro[snap_id] = { 'z': z, 'k_vals': k_vals, 'power_spectrum':power_spectrum }




figure_width = 4
text_color = 'black'  
nrows = 1
ncols = 2
fig_height = nrows * figure_width
fig_width = ncols * figure_width
fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height) )
plt.subplots_adjust( hspace = 0.0, wspace=0.16)

for i in range(2):
  if i == 0: ps_data_all = ps_dm
  if i == 1: ps_data_all = ps_hydro
    
  ax = ax_l[i]
  
  for snap_id in ps_data_all:
    ps_data = ps_data_all[snap_id]
    z = ps_data['z']
    k_vals = ps_data['k_vals']
    ps = ps_data['power_spectrum']
    
    label = r'$z=$' +f'{z:.1f}'
    ax.plot( k_vals, ps, label=label )
  
  ax.set_xscale('log')
  ax.set_yscale('log')

  ax.set_xlabel(r'$k$  [$h\, \mathrm{Mpc^{-1}}$]')
  ax.set_ylabel(r'$P(k)$')

  ax.legend(frameon=False, loc=3, fontsize=8)

figure_name  = output_dir + 'power_spectrum.png'
fig.savefig( figure_name, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor() )
print( f'Saved Figure: {figure_name}' )



