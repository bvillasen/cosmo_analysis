import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import subprocess
#Extend path to inclide local modules
root_dir = os.path.dirname(os.getcwd())
sub_directories = [x[0] for x in os.walk(root_dir)]
sys.path.extend(sub_directories)
from tools import *
from ics_particles import generate_ics_particles
from ics_grid import expand_data_grid_to_cholla
from internal_energy import get_internal_energy
from load_enzo_ics_file import Load_File_Attrs, Load_Gas_Field, Load_Particles_Field


ics_hydro, ics_particles = False, False
args = sys.argv
n_args = len(args)
if n_args == 1:
  print( 'Missing type: hydro or particles')
  exit(-1)

types = args[1:]
if 'hydro' in types:     ics_hydro = True
if 'particles' in types: ics_particles = True



# # Box Size
# Lbox = 50000.0    #kpc/h
# n_points = 256
# n_boxes  = 8


# Box Size
Lbox = 50000.0    #kpc/h
n_points = 512
n_boxes  = 16


L_Mpc = int( Lbox / 1000)

input_dir = f'/gpfs/alpine/ast175/proj-shared/using_cholla/ics/ics_music/ics_{n_points}_{L_Mpc}Mpc_enzo/'
output_dir = f'/gpfs/alpine/ast175/proj-shared/using_cholla/ics/ics_music/ics_{n_points}_{L_Mpc}Mpc/'

create_directory( output_dir )
output_dir += f'ics_{n_boxes}_z100/'
create_directory( output_dir )
print(f'Input Dir: {input_dir}' )
print(f'Output Dir: {output_dir}' )

temperature = 231.44931976   #k
H_fraction = 0.75984603480
He_fraction = 1 - H_fraction
mmw = 1 / ( H_fraction + He_fraction/4 )
file_attrs = Load_File_Attrs( input_dir, Lbox=Lbox, type=types[0] )

data_ics = { 'dm':{}, 'gas':{} }
data_ics['current_a'] = file_attrs['a_start']
data_ics['current_z'] = file_attrs['z_start']

if ics_hydro:
  gas_density = Load_Gas_Field( 'density', input_dir, attrs=file_attrs )
  gas_vel_x = Load_Gas_Field( 'vel_x', input_dir, attrs=file_attrs )
  gas_vel_y = Load_Gas_Field( 'vel_y', input_dir, attrs=file_attrs )
  gas_vel_z = Load_Gas_Field( 'vel_z', input_dir, attrs=file_attrs )
  print( 'Computing GasEnergy')
  gas_U = get_internal_energy( temperature, mu=mmw ) * gas_density
  print( 'Computing Energy')
  gas_E = 0.5*gas_density*( gas_vel_x*gas_vel_x + gas_vel_y*gas_vel_y + gas_vel_z*gas_vel_z ) + gas_U

  data_ics['gas']['density'] = gas_density
  data_ics['gas']['momentum_x'] = gas_density * gas_vel_x
  data_ics['gas']['momentum_y'] = gas_density * gas_vel_y
  data_ics['gas']['momentum_z'] = gas_density * gas_vel_z
  data_ics['gas']['GasEnergy'] = gas_U
  data_ics['gas']['Energy'] = gas_E


if ics_particles:
  p_pos_x = Load_Particles_Field( 'pos_x', input_dir, attrs=file_attrs )
  p_pos_y = Load_Particles_Field( 'pos_y', input_dir, attrs=file_attrs )
  p_pos_z = Load_Particles_Field( 'pos_z', input_dir, attrs=file_attrs )
  p_vel_x = Load_Particles_Field( 'vel_x', input_dir, attrs=file_attrs )
  p_vel_y = Load_Particles_Field( 'vel_y', input_dir, attrs=file_attrs )
  p_vel_z = Load_Particles_Field( 'vel_z', input_dir, attrs=file_attrs )
  print( f"Particle Mass: {file_attrs['dm_particle_mass']}" )
  data_ics['dm']['p_mass'] = file_attrs['dm_particle_mass']
  data_ics['dm']['pos_x'] = p_pos_x
  data_ics['dm']['pos_y'] = p_pos_y
  data_ics['dm']['pos_z'] = p_pos_z
  data_ics['dm']['vel_x'] = p_vel_x
  data_ics['dm']['vel_y'] = p_vel_y
  data_ics['dm']['vel_z'] = p_vel_z


if n_boxes == 1: proc_grid  = [ 1, 1, 1 ]
if n_boxes == 2: proc_grid  = [ 2, 1, 1 ]
if n_boxes == 8: proc_grid  = [ 2, 2, 2 ]
if n_boxes == 16: proc_grid = [ 4, 2, 2 ]
if n_boxes == 32: proc_grid = [ 4, 4, 2 ]
if n_boxes == 64: proc_grid = [ 4, 4, 4 ]
if n_boxes == 128: proc_grid = [ 8, 4, 4 ]
if n_boxes == 512: proc_grid = [ 8, 8, 8 ]
if n_boxes == 1024: proc_grid = [ 16, 8, 8 ]
if n_boxes == 2048: proc_grid = [ 16, 16, 8 ]

n_snapshot = 0
box_size = [ Lbox, Lbox, Lbox ]
grid_size = [ n_points, n_points, n_points ]
output_base_name = '{0}_particles.h5'.format( n_snapshot )
if ics_particles: generate_ics_particles(data_ics, output_dir, output_base_name, proc_grid, box_size, grid_size)

output_base_name = '{0}.h5'.format( n_snapshot )
if ics_hydro: expand_data_grid_to_cholla( proc_grid, data_ics['gas'], output_dir, output_base_name, loop_complete_files=True )
