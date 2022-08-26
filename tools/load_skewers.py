import os, sys
import h5py as h5
import numpy as np


def load_skewers_single_axis(  n_skewers, skewer_axis,  nSnap, input_dir, set_random_seed=False, print_out=True, ids_to_load=None, load_HeII=False ):
  inFileName = input_dir + f'skewers_{skewer_axis}_{nSnap}.h5'
  inFile = h5.File( inFileName, 'r' )
  n_total = inFile.attrs['n']
  # if print_out: print( f'Availbale squewers: {n_total}')
  current_z = inFile.attrs['current_z']
  
  if print_out: print(f"Loading {n_skewers} skewers {skewer_axis} axis from {n_total} available")
  
  if type(ids_to_load) == np.ndarray:
    skewer_ids = ids_to_load
    if print_out: print( f' Loading: {skewer_axis} {skewer_ids}')
    if n_skewers != len(skewer_ids):
      print("ERROR: List of skewer ids sont match n_skewres"      )
      exit(-1)
  elif ids_to_load == 'random':  
    if set_random_seed:   
      if print_out: print( 'WANING: Fixed random seed to load skewers')
      np.random.seed(12345)
    # skewer_ids = np.random.randint(0, n_total, n_skewers)
    skewer_ids = np.random.randint(0, n_total, n_skewers).astype(np.float)
  elif ids_to_load == 'all':
    skewer_ids = np.arange( n_total )
  else:
    print('ERROR: Not supported ids to load ')
    return None 


  skewers_dens, skewers_temp, skewers_HI, skewers_vel, skewers_HeII = [], [], [], [], []
  for skewer_id in skewer_ids:
    if str(skewer_id) not in inFile.keys(): skewer_id = float(skewer_id)
    skewer_data = inFile[str(skewer_id)]
    density = skewer_data['density'][...]
    HI_density = skewer_data['HI_density'][...]
    temperature = skewer_data['temperature'][...]
    velocity = skewer_data['velocity'][...]
    skewers_dens.append( density )
    skewers_HI.append( HI_density )
    skewers_temp.append( temperature )
    skewers_vel.append(velocity)
    if load_HeII:
      HeII_density = skewer_data['HeII_density'][...]
      skewers_HeII.append( HeII_density )

  inFile.close() 
  data_out  = {}
  data_out['current_z'] = current_z
  data_out['n_skewers'] = n_skewers
  data_out['density']     = np.array( skewers_dens )
  data_out['HI_density']  = np.array( skewers_HI )
  data_out['temperature'] = np.array( skewers_temp )
  data_out['velocity']    = np.array( skewers_vel )
  if load_HeII: data_out['HeII_density']  = np.array( skewers_HeII )
  return data_out


def load_skewers_multiple_axis( axis_list, n_skewers_list, nSnap, input_dir, set_random_seed=False, print_out=True, ids_to_load_list=None, load_HeII=False):
  n_axis = len(axis_list)

  dens_list, HI_list, temp_list, vel_list, HeII_list = [], [], [], [], []

  for i in range( n_axis ):
    skewer_axis = axis_list[i]
    n_skewers = n_skewers_list[i]
    if ids_to_load_list != None:  ids_to_load = ids_to_load_list[i]
    else: ids_to_load = None
    data_skewers = load_skewers_single_axis( n_skewers, skewer_axis,  nSnap, input_dir, set_random_seed=set_random_seed, print_out=print_out, ids_to_load=ids_to_load, load_HeII=load_HeII )
    current_z = data_skewers['current_z']
    skewers_dens = data_skewers['density']
    skewers_HI = data_skewers['HI_density']
    skewers_temp = data_skewers['temperature']
    skewers_vel = data_skewers['velocity']
    if load_HeII: skewers_HeII = data_skewers['HeII_density']
    dens_list.append( skewers_dens )
    HI_list.append( skewers_HI )
    temp_list.append( skewers_temp )
    vel_list.append( skewers_vel )
    if load_HeII:  HeII_list.append( skewers_HeII )
  dens_all = np.concatenate( dens_list )
  dens_list = []
  HI_all = np.concatenate( HI_list )
  HI_list = []
  temp_all = np.concatenate( temp_list )
  temp_list = []
  vel_all = np.concatenate( vel_list )
  vel_list = []
  if load_HeII:
    HeII_all = np.concatenate( HeII_list )
    HeII_list = []
  n_skewers = len( dens_all)
  data_skewers = {}
  data_skewers['current_z'] = current_z
  data_skewers['n_skewers'] = n_skewers
  data_skewers['density'] = dens_all
  data_skewers['HI_density'] = HI_all
  data_skewers['los_velocity'] = vel_all
  data_skewers['temperature'] = temp_all
  if load_HeII: data_skewers['HeII_density'] = HeII_all  
  return data_skewers


# dataDir = '/data/groups/comp-astro/bruno/'
# input_dir = dataDir + 'cosmo_sims/2048_hydro_50Mpc/skewers_pchw18/'
# 
# nSnap = 169
# 
# skewer_axis = 'x'
# n_skewers = 100
# data_skewers = load_skewers_single_axis(n_skewers, skewer_axis,  nSnap, input_dir, set_random_seed=True, print_out=True )
# current_z = data_skewers['current_z']
# 
# 
# skewer_id = 0
# skewer_density     = data_skewers['density'][skewer_id]
# skewer_HI_density  = data_skewers['HI_density'][skewer_id]
# skewer_temperature = data_skewers['temperature'][skewer_id]
# skewer_velocity    = data_skewers['velocity'][skewer_id]
