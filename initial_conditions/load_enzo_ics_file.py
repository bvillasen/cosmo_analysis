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
from constants_cosmo import G_COSMO, Gcosmo
from internal_energy import get_temperature, get_internal_energy

correction_factor = 1.0000563343966022


def Load_File_Attrs( input_dir, Lbox=None, type='hydro'):
  if type == 'hydro': file_name='GridDensity'
  if type == 'particles': file_name='ParticleDisplacements_x'
  file_name = input_dir + file_name
  file = h5.File( file_name, 'r' )
  attrs = file.attrs
  n_cells = attrs['Dimensions']
  n_particles = n_cells.prod()
  h = attrs['h0']
  dx = attrs['dx'] * h  #Mpc/h
  box_size = n_cells * dx 
  if (box_size == box_size[0]).sum() != 3:
    print( 'ERROR: Box is not a cube')
    return None
  box_size = box_size[0]  
  H0 = h * 100
  a_start = attrs['a_start']
  z_start = 1 / a_start - 1
  Omega_b = attrs['omega_b']
  Omega_m = attrs['omega_m']
  Omega_l = attrs['omega_v']
  file.close()
  rho_crit = 3*(H0*1e-3)**2/(8*np.pi* Gcosmo) / h / h * correction_factor # h^2 Msun kpc^-3 
  rho_mean = rho_crit * Omega_m
  rho_mean_dm  = rho_crit * ( Omega_m - Omega_b )
  rho_mean_gas = rho_crit * Omega_b
  dm_particle_mass =  rho_mean_dm * (Lbox)** 3  / n_particles #Msun/h
  L_unit = Lbox / H0 / (1+z_start) 
  rho_0 = 3 * Omega_m * (100* H0*1e-3 )**2 / ( 8 * np.pi * Gcosmo )
  time_unit = 1/np.sqrt(  4 * np.pi * Gcosmo * rho_0 * (1+z_start)**3 )
  vel_unit = L_unit / time_unit 
  attrs = { 'dx':dx, 'h':h, 'box_size':box_size, 'H0':H0, 'a_start':a_start, 'Omega_b':Omega_b,
            'Omega_m': Omega_m, 'Omega_l':Omega_l, 'rho_crit':rho_crit, 'rho_mean_dm':rho_mean_dm,
            'rho_mean_gas':rho_mean_gas, 'dm_particle_mass':dm_particle_mass, 'vel_unit':vel_unit,
            'n_points':n_cells[0], 'z_start':z_start, 'Lbox':Lbox }
  return attrs
  


field_keys_gas = {'density':'GridDensity', 'vel_x':'GridVelocities_x', 'vel_y':'GridVelocities_y', 'vel_z':'GridVelocities_z', }
field_keys_particles = { 'pos_x':'ParticleDisplacements_x', 'pos_y':'ParticleDisplacements_y', 'pos_z':'ParticleDisplacements_z', 'vel_x':'ParticleVelocities_x', 'vel_y':'ParticleVelocities_y', 'vel_z':'ParticleVelocities_z' }

def Load_Gas_Field( field_name, input_dir, attrs=None ):
  if attrs is None: attrs = Load_File_Attrs( input_dir )
  
  field_key = field_keys_gas[field_name]
  file_name = input_dir + field_key
  print( f'Loading File: {file_name}' )
  file = h5.File( file_name, 'r' )
  field = file[field_key][...][0].T
  file.close()
  if field_name == 'density': field *= attrs['Omega_m'] / attrs['Omega_b'] * attrs['rho_mean_gas']  
  if field_name in [ 'vel_x', 'vel_y', 'vel_z' ]: field *= attrs['vel_unit'] 
  return field


def Load_Particles_Field( field_name, input_dir, attrs=None ):
  if attrs is None: attrs = Load_File_Attrs( input_dir )
  
  field_key = field_keys_particles[field_name]
  file_name = input_dir + field_key
  print( f'Loading File: {file_name}' )
  file = h5.File( file_name, 'r' )
  field = file[field_key][...][0].T
  file.close()
  
  if field_name in [ 'pos_x', 'pos_y', 'pos_z' ]:
    n_points = attrs['n_points']
    delta_x = 1 / n_points
    ones_1d = np.ones( n_points )
    p_pos   = np.meshgrid( ones_1d, ones_1d, ones_1d )[0]
    for slice_id in range(n_points):
      if field_name == 'pos_x': p_pos[slice_id, :, :] *= (slice_id + 0.5) * delta_x
      if field_name == 'pos_y': p_pos[:, slice_id, :] *= (slice_id + 0.5) * delta_x
      if field_name == 'pos_z': p_pos[:, :, slice_id] *= (slice_id + 0.5) * delta_x
    p_pos += field
    p_pos = p_pos.flatten() 
    p_pos[p_pos < 0] += 1 
    p_pos[p_pos > 1] -= 1
    p_pos *= attrs['Lbox']
    return p_pos
    
  if field_name in  [ 'vel_x', 'vel_y', 'vel_z' ]: 
    p_vel = field.flatten() * attrs['vel_unit']
    return p_vel
    