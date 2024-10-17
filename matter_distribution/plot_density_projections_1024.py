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
from cosmology import Cosmology
from constants_cosmo import Myear
from colors import *

# Initialize Cosmology
z_start = 100000
cosmo = Cosmology( z_start=z_start, get_a=True )
cosmo_z = cosmo.z_vals
cosmo_t = cosmo.t_vals / Myear / 1000 #Gyear

Lbox = 50000.0    #kpc/h
n_cells = 1024
L_Mpc = int(Lbox/1000)

base_dir = f'/home/bvillase/tests/cosmo_sims/{n_cells}_{L_Mpc}Mpc/' 
input_dir  = base_dir + 'snapshot_files/'
proj_dir = base_dir + 'projections/'
output_dir = base_dir + 'figures/'
create_directory( proj_dir ) 
create_directory( output_dir ) 


data_type = 'particles'
fields = [ 'density' ]

box_size = [ Lbox, Lbox, Lbox ]
grid_size = [ n_cells, n_cells, n_cells ] #Size of the simulation grid
n_slice = 400
subgrid = [[0,n_slice], [0,n_cells], [0,n_cells]]
precision = np.float32

max_factor = 1e5
dens_max = None
vmin, vmax = np.inf, -np.inf

snapshots = [ 2, 5, 10,  17]

force_load_snap = False

figs_data = {}
for snap_id, n_snap in enumerate(snapshots):
  file_name = proj_dir + f'projection_{n_snap}.h5'
  if os.path.isfile(file_name) and not force_load_snap:
    print( f'Loading file: {file_name}')
    file = h5.File( file_name, 'r' )
    z = file.attrs['z']
    proj = file['proj'][...]
    file.close()
  
  else:   
    snap_data = load_snapshot_data_distributed( data_type, fields, n_snap, input_dir,  box_size, grid_size, precision, subgrid=subgrid  )
    z = snap_data['Current_z']
    density = snap_data['density']
    slice = density
    dens_mean = density.mean()
    if dens_max is None: dens_max = max_factor * dens_mean
    density[density > dens_max] = dens_max
    proj2 = (slice**2).sum( axis=0 )
    proj  = slice.sum( axis=0 )
    proj2 = proj2 / proj
    file = h5.File( file_name, 'w' )
    file.attrs['z'] = z
    file.create_dataset( 'proj', data=proj )
    file.create_dataset( 'proj2', data=proj2 )
    file.close()
    print( f'Saved file: {file_name}')
    
  projection =  proj
  projection = np.log10(projection)
  vmax = max( vmax, projection.max() )
  vmin = min( vmin, projection.min() )
  t = np.interp(z, cosmo_z[::-1], cosmo_t[::-1])
  figs_data[snap_id] = {'proj':projection, 't':t}


print( f'z: {z:.2f}  t: {t:.2f} GYr')

font_size = 16
legend_font_size = 12

label_size = 18
figure_text_size = 16
legend_font_size = 12
tick_label_size_major = 15
tick_label_size_minor = 13
tick_size_major = 5
tick_size_minor = 3
tick_width_major = 1.5
tick_width_minor = 1
border_width = 1
text_color = 'k'

color_map = cmap_deep_r
color_map = cmap_davos

colormaps = [ cmap_deep_r, cmap_davos, cmap_oslo, cmap_lapaz, cmap_ice, cmap_turku, cmap_lajolla_r ]
# colormaps = [ cmap_turku  ]


figure_width = 6




for cmap_id, color_map in enumerate(colormaps):

  color_map = colormaps[cmap_id]

  nrows, ncols = 1, 4
  fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figure_width*ncols,figure_width*nrows))
  plt.subplots_adjust( hspace = 0.03, wspace=0.03)


  for fig_id in figs_data:

    i = fig_id // ncols
    j = fig_id % ncols
    # ax = ax_l[i][j]
    ax = ax_l[j]

    fig_data = figs_data[fig_id]
    projection = fig_data['proj']
    t = fig_data['t']
    vmin = projection.min() 
    ax.imshow( projection, cmap=color_map, extent=[0, 50, 0, 50], vmin=vmin, vmax=vmax )
    # ax.imshow( projection, cmap=color_map, extent=[0, 50, 0, 50] )

    label = f'Time: {t:.1f} Gyr'
    text_pos_x = 0.05
    text_pos_y = 0.93
    bbox_props = dict(boxstyle="round", fc="gray", ec="0.5", alpha=0.5)
    ax.text(text_pos_x, text_pos_y, label, horizontalalignment='left',  verticalalignment='center', transform=ax.transAxes, fontsize=figure_text_size, color='white', bbox=bbox_props) 
    

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

  figure_name = output_dir + f'dm_density_{cmap_id}.png'
  fig.savefig( figure_name, bbox_inches='tight', dpi=500, facecolor=fig.get_facecolor() )
  print( f'Saved Figure: {figure_name}' )
