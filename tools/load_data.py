import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import pickle


def Load_Skewers_File( n_file, input_dir, axis_list=[ 'x', 'y', 'z' ], fields_to_load=[ 'HI_density', 'temperature', 'los_velocity' ] ):
  file_name = input_dir + f'{n_file}_skewers.h5'
  file = h5.File( file_name, 'r' )
  data_out = { key:file.attrs[key][0] for key in file.attrs }
  for field in fields_to_load:
    skewers = []
    for axis in axis_list:
      skewers_axis = file[f'skewers_{axis}'][field][...]
      skewers.append( skewers_axis )
    skewers = np.concatenate( skewers )
    data_out[field] = skewers
  data_out[f'vel_Hubble'] = file['skewers_x']['vel_Hubble'][...]
  file.close()
  return data_out


def Load_Skewers_File_old( n_file, input_dir, chem_type = 'HI', axis_list = [ 'x', 'y', 'z' ] ):
  file_name = input_dir + f'{n_file}_skewers.h5'
  file = h5.File( file_name, 'r' )
  data_out = { key:file.attrs[key][0] for key in file.attrs }
  skewers = []
  for axis in axis_list:
    skewers_axis = file[f'skewers_{axis}'][f'los_transmitted_flux_{chem_type}'][...]
    skewers.append( skewers_axis )
  skewers = np.concatenate( skewers )
  data_out[f'skewers_flux_{chem_type}'] = skewers
  data_out[f'vel_Hubble'] = file['skewers_x']['vel_Hubble'][...]
  file.close()
  return data_out
  

def load_analysis_data( n_file, input_dir, phase_diagram=True, lya_statistics=True, load_skewer=False, load_fit=False, load_flux_Pk=True, mcmc_fit_dir=None):
  file_name = input_dir + f'{n_file}_analysis.h5'
  file = h5.File( file_name, 'r' ) 

  data_out = {}
  attrs = file.attrs
  # print( attrs.keys() )
  
  data_out['box'] = {}
  data_out['box']['Lbox'] = attrs['Lbox']
  
  
  data_out['cosmology'] = {}
  data_out['cosmology']['H0'] = attrs['H0'][0]
  data_out['cosmology']['Omega_L'] = attrs['Omega_L'][0]
  data_out['cosmology']['Omega_M'] = attrs['Omega_M'][0]
  data_out['cosmology']['current_z'] = attrs['current_z'][0]
  data_out['cosmology']['current_a'] = attrs['current_a'][0]
  
  if phase_diagram or load_fit:  data_out['phase_diagram'] = {}  
  
  if phase_diagram:
    phase_diagram = file['phase_diagram']
    for key in phase_diagram.attrs:
      data_out['phase_diagram'][key] = phase_diagram.attrs[key][0]
    data_out['phase_diagram']['data'] = phase_diagram['data'][...] 
    
  if load_fit:
    fit_dir_name = 'fit_mcmc'
    if mcmc_fit_dir: fit_dir_name = mcmc_fit_dir
    fit_dir = input_dir + f'{fit_dir_name}/'
    # print(f'Loading Fit: {fit_dir}' )
    fit_file_name = fit_dir + f'fit_{n_file}.pkl'
    fit_file = open(fit_file_name, 'rb')
    mcmc_stats = pickle.load(fit_file)
    mcmc_T0 = mcmc_stats['T0']['mean']
    mcmc_T0_sigma = mcmc_stats['T0']['standard deviation']
    mcmc_gamma = mcmc_stats['gamma']['mean']
    mcmc_gamma_sigma = mcmc_stats['gamma']['standard deviation']
    T0 = 10**mcmc_T0
    delta_T0 =  T0 * np.log(10) * mcmc_T0_sigma
    gamma = mcmc_gamma
    delta_gamma = mcmc_gamma_sigma
    data_out['phase_diagram']['fit'] = {}
    data_out['phase_diagram']['fit']['T0'] = T0
    data_out['phase_diagram']['fit']['delta_T0'] = delta_T0
    data_out['phase_diagram']['fit']['gamma'] = gamma
    data_out['phase_diagram']['fit']['delta_gamma'] = delta_gamma
    
  if lya_statistics:
    data_out['lya_statistics'] = {}
    lya_statistics = file['lya_statistics']
    data_out['lya_statistics']['n_skewers'] = lya_statistics.attrs['n_skewers'][0]
    F_mean = lya_statistics.attrs['Flux_mean_HI'][0]
    F_mean = max( F_mean, 1e-60 )
    data_out['lya_statistics']['Flux_mean'] = F_mean
    data_out['lya_statistics']['tau'] = -np.log( F_mean )
    F_mean = lya_statistics.attrs['Flux_mean_HeII'][0]
    F_mean = max( F_mean, 1e-60 )
    data_out['lya_statistics']['Flux_mean_HeII'] = F_mean
    data_out['lya_statistics']['tau_HeII'] = -np.log( F_mean )
    
    if load_flux_Pk:
      data_out['lya_statistics']['power_spectrum'] = {}
      if 'power_spectrum' in lya_statistics:
        ps_data = lya_statistics['power_spectrum']
        k_vals  = ps_data['k_vals'][...]
        ps_mean = ps_data['p(k)'][...]
        indices = ps_mean > 0
        data_out['lya_statistics']['power_spectrum']['k_vals']  = k_vals[indices]
        data_out['lya_statistics']['power_spectrum']['ps_mean'] = ps_mean[indices]
      else:
        data_out['lya_statistics']['power_spectrum']['k_vals']  = None
        data_out['lya_statistics']['power_spectrum']['ps_mean'] = None

    if load_skewer:
      skewer = lya_statistics['skewer']
      data_out['lya_statistics']['skewer'] = {}
      data_out['lya_statistics']['skewer']['HI_density']       = skewer['HI_density'][...]
      data_out['lya_statistics']['skewer']['velocity']         = skewer['velocity'][...]
      data_out['lya_statistics']['skewer']['temperature']      = skewer['temperature'][...]
      data_out['lya_statistics']['skewer']['optical_depth']    = skewer['optical_depth'][...]
      data_out['lya_statistics']['skewer']['vel_Hubble']       = skewer['vel_Hubble'][...]
      data_out['lya_statistics']['skewer']['transmitted_flux'] = skewer['transmitted_flux'][...]
  file.close()
  return data_out

def get_domain_block( proc_grid, box_size, grid_size ):
  np_x, np_y, np_z = proc_grid
  Lx, Ly, Lz = box_size
  nx_g, ny_g, nz_g = grid_size
  dx, dy, dz = Lx/np_x, Ly/np_y, Lz/np_z
  nx_l, ny_l, nz_l = nx_g//np_x, ny_g//np_y, nz_g//np_z,

  nprocs = np_x * np_y * np_z
  domain = {}
  domain['global'] = {}
  domain['global']['dx'] = dx
  domain['global']['dy'] = dy
  domain['global']['dz'] = dz
  for k in range(np_z):
    for j in range(np_y):
      for i in range(np_x):
        pId = i + j*np_x + k*np_x*np_y
        domain[pId] = { 'box':{}, 'grid':{} }
        xMin, xMax = i*dx, (i+1)*dx
        yMin, yMax = j*dy, (j+1)*dy
        zMin, zMax = k*dz, (k+1)*dz
        domain[pId]['box']['x'] = [xMin, xMax]
        domain[pId]['box']['y'] = [yMin, yMax]
        domain[pId]['box']['z'] = [zMin, zMax]
        domain[pId]['box']['dx'] = dx
        domain[pId]['box']['dy'] = dy
        domain[pId]['box']['dz'] = dz
        domain[pId]['box']['center_x'] = ( xMin + xMax )/2.
        domain[pId]['box']['center_y'] = ( yMin + yMax )/2.
        domain[pId]['box']['center_z'] = ( zMin + zMax )/2.
        gxMin, gxMax = i*nx_l, (i+1)*nx_l
        gyMin, gyMax = j*ny_l, (j+1)*ny_l
        gzMin, gzMax = k*nz_l, (k+1)*nz_l
        domain[pId]['grid']['x'] = [gxMin, gxMax]
        domain[pId]['grid']['y'] = [gyMin, gyMax]
        domain[pId]['grid']['z'] = [gzMin, gzMax]
  return domain
  

def select_procid( proc_id, subgrid, domain, ids, ax ):
  domain_l, domain_r = domain
  subgrid_l, subgrid_r = subgrid
  if domain_l <= subgrid_l and domain_r > subgrid_l:
    ids.append(proc_id)
  if domain_l >= subgrid_l and domain_r <= subgrid_r:
    ids.append(proc_id)
  if domain_l < subgrid_r and domain_r >= subgrid_r:
    ids.append(proc_id)




def select_ids_to_load( subgrid, domain, proc_grid ):
  subgrid_x, subgrid_y, subgrid_z = subgrid
  nprocs = proc_grid[0] * proc_grid[1] * proc_grid[2]
  ids_x, ids_y, ids_z = [], [], []
  for proc_id in range(nprocs):
    domain_local = domain[proc_id]
    domain_x = domain_local['grid']['x']
    domain_y = domain_local['grid']['y']
    domain_z = domain_local['grid']['z']
    select_procid( proc_id, subgrid_x, domain_x, ids_x, 'x' )
    select_procid( proc_id, subgrid_y, domain_y, ids_y, 'y' )
    select_procid( proc_id, subgrid_z, domain_z, ids_z, 'z' )
  set_x = set(ids_x)
  set_y = set(ids_y)
  set_z = set(ids_z)
  set_ids = (set_x.intersection(set_y)).intersection(set_z )
  return list(set_ids)



def load_snapshot_data_distributed( data_type, fields,  nSnap, inDir,  box_size, grid_size,    precision, subgrid=None, proc_grid=None, show_progess=True, get_statistics=False, print_fields=False ):
  
  if show_progess:
    print( f'Loading {data_type} Snapshot: {nSnap}  ')
    print( f' Input Directory:  {inDir}  ')
    
  name_base = 'h5'
  # Load Header Data
  if data_type == 'particles': inFileName = '{0}_particles.{1}.0'.format(nSnap, name_base )
  if data_type == 'hydro': inFileName = '{0}.{1}.0'.format(nSnap, name_base )
  
  inFile = h5.File( inDir + inFileName, 'r')
  available_fields = inFile.keys()
  head = inFile.attrs
  if not proc_grid:  proc_grid = head['nprocs']
  if not subgrid:  subgrid = [ [0, grid_size[0]], [0, grid_size[1]], [0, grid_size[2]] ]
    
  if show_progess:
    print( f' proc_grid: {proc_grid}' )
    print( f' grid_size: {grid_size}' )
    print( f' subgrid:   {subgrid}' )
    if 'current_z' in head: print(' current_z: {0}'.format( head['current_z'][0] ) )
    elif 'Current_z' in head: print(' current_z: {0}'.format( head['Current_z'][0] ) )
  inFile.close()
    
  # Get the doamin domain_decomposition
  domain = get_domain_block( proc_grid, box_size, grid_size )
  
  # Find the ids to load 
  ids_to_load = select_ids_to_load( subgrid, domain, proc_grid )

  # print(("Loading Snapshot: {0}".format(nSnap)))
  #Find the boundaries of the volume to load
  domains = { 'x':{'l':[], 'r':[]}, 'y':{'l':[], 'r':[]}, 'z':{'l':[], 'r':[]}, }
  for id in ids_to_load:
    for ax in list(domains.keys()):
      d_l, d_r = domain[id]['grid'][ax]
      domains[ax]['l'].append(d_l)
      domains[ax]['r'].append(d_r)
  boundaries = {}
  for ax in list(domains.keys()):
    boundaries[ax] = [ min(domains[ax]['l']),  max(domains[ax]['r']) ]

  # Get the size of the volume to load
  nx = int(boundaries['x'][1] - boundaries['x'][0])    
  ny = int(boundaries['y'][1] - boundaries['y'][0])    
  nz = int(boundaries['z'][1] - boundaries['z'][0])    


  dims_all = [ nx, ny, nz ]
  data_out = {}
  if get_statistics: data_out['statistics'] = {}
  for field in fields:
    data_particels = False
    if field in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'particle_IDs']: data_particels = True 
    if not data_particels: data_all = np.zeros( dims_all, dtype=precision )
    else: data_all = []
    added_header = False
    n_to_load = len(ids_to_load)
    if get_statistics:
      data_out['statistics'][field] = {}
      min_val, max_val = np.inf, -np.inf
    for i, nBox in enumerate(ids_to_load):
      if data_type == 'particles': inFileName = '{0}_particles.{1}.{2}'.format(nSnap, name_base, nBox)
      if data_type == 'hydro': inFileName = '{0}.{1}.{2}'.format(nSnap, name_base, nBox)
    
      inFile = h5.File( inDir + inFileName, 'r')
      available_fields = inFile.keys()
      head = inFile.attrs
      if added_header == False:
        # print( ' Loading: ' + inDir + inFileName )
        if print_fields: print( f' Available Fields:  {available_fields}')
        for h_key in list(head.keys()):
          if h_key in ['dims', 'dims_local', 'offset', 'bounds', 'domain', 'dx', ]: continue
          data_out[h_key] = head[h_key][0]
        added_header = True
    
      if show_progess:
        terminalString  = '\r Loading File: {0}/{1}   {2}  {3}'.format(i, n_to_load, data_type, field)
        sys.stdout. write(terminalString)
        sys.stdout.flush() 
    
      if not data_particels:
        procStart_x, procStart_y, procStart_z = head['offset']
        procEnd_x, procEnd_y, procEnd_z = head['offset'] + head['dims_local']
        # Substract the offsets
        procStart_x -= boundaries['x'][0]
        procEnd_x   -= boundaries['x'][0]
        procStart_y -= boundaries['y'][0]
        procEnd_y   -= boundaries['y'][0]
        procStart_z -= boundaries['z'][0]
        procEnd_z   -= boundaries['z'][0]
        procStart_x, procEnd_x = int(procStart_x), int(procEnd_x)
        procStart_y, procEnd_y = int(procStart_y), int(procEnd_y)
        procStart_z, procEnd_z = int(procStart_z), int(procEnd_z)
        data_local = inFile[field][...]
        data_all[ procStart_x:procEnd_x, procStart_y:procEnd_y, procStart_z:procEnd_z] = data_local
        if get_statistics:
          min_val = min( min_val, data_local.min())
          max_val = max( min_val, data_local.max())
      else:
        data_local = inFile[field][...]
        data_all.append( data_local )
    
    if not data_particels:
      # Trim off the excess data on the boundaries:
      trim_x_l = subgrid[0][0] - boundaries['x'][0]
      trim_x_r = boundaries['x'][1] - subgrid[0][1]  
      trim_y_l = subgrid[1][0] - boundaries['y'][0]
      trim_y_r = boundaries['y'][1] - subgrid[1][1]  
      trim_z_l = subgrid[2][0] - boundaries['z'][0]
      trim_z_r = boundaries['z'][1] - subgrid[2][1]  
      trim_x_l, trim_x_r = int(trim_x_l), int(trim_x_r) 
      trim_y_l, trim_y_r = int(trim_y_l), int(trim_y_r) 
      trim_z_l, trim_z_r = int(trim_z_l), int(trim_z_r) 
      data_output = data_all[trim_x_l:nx-trim_x_r, trim_y_l:ny-trim_y_r, trim_z_l:nz-trim_z_r,  ]
      data_out[field] = data_output
      if get_statistics:
        data_out['statistics'][field]['min'] = min_val
        data_out['statistics'][field]['max'] = max_val
        
    else:
      data_all = np.concatenate( data_all )
      data_out[field] = data_all
      # if field == 'particle_IDs': data_out[field] = data_out[field].astype( np.int ) 
    if show_progess: print("")
  return data_out


def load_cholla_snapshot_file( nSnap, inDir, cool=False, dm=True, cosmo=True, hydro=True, file_name=None  ):
  
  if file_name != None:
    gridFileName = inDir + file_name
  else:
    partFileName = inDir + 'particles_{0:03}.h5'.format(nSnap)
    gridFileName = inDir + 'grid_{0:03}.h5'.format(nSnap)

  outDir = {'dm':{}, 'gas':{} }
  if hydro:  
    data_grid = h5.File( gridFileName, 'r' )
    fields_data = list(data_grid.keys())
    for key in list(data_grid.attrs.keys()): outDir[key] = data_grid.attrs[key]
    fields_grid = fields_data
    for field in fields_grid:
      if field not in fields_data: continue
      outDir['gas'][field] = data_grid[field]

  if dm:
    data_part = h5.File( partFileName, 'r' )
    fields_data = list(data_part.keys())
    fields_part = [ 'density',  'grav_potential', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'particle_IDs' ]
    # current_z = data_part.attrs['current_z']
    # current_a = data_part.attrs['current_a']
    # outDir['current_a'] = current_a
    # outDir['current_z'] = current_z
    for key in list(data_part.attrs.keys()): outDir[key] = data_part.attrs[key]
    if cosmo:
      current_z = data_part.attrs['current_z']
      print(("Loading Cholla Snapshot: {0}       current_z: {1}".format( nSnap, current_z) ))
    for field in fields_part:
      if field not in fields_data: continue
      # print field
      outDir['dm'][field] = data_part[field]

  return outDir


# #Example for Loading  Snapshot Data (Below)
# 
# data_dir = '/data/groups/comp-astro/bruno/'
# input_dir = data_dir + 'cosmo_sims/halo_tests/256_hydro_50Mpc/output_files/'
# 
# 
# precision = np.float64
# Lbox = 50000.0    #kpc/h
# n_cells = 256
# box_size = [ Lbox, Lbox, Lbox ]
# grid_size = [ n_cells, n_cells, n_cells ] #Size of the simulation grid
# 
# n_snapshot = 0
# 
# #Load Gas data
# fields = [ 'density' ]
# data_gas = load_snapshot_data_distributed( 'hydro', fields, n_snapshot, input_dir, box_size, grid_size,  precision, show_progess=True )
# current_z = data_gas['Current_z']  #redshift
# density_gas = data_gas['density']  # h^2 Msun / kpc^3
# 
# #Load DM data
# fields = [ 'density', 'pos_x', 'pos_y', 'pos_z', 'particle_IDs' ]
# data_dm = load_snapshot_data_distributed( 'particles', fields, n_snapshot, input_dir, box_size, grid_size,  precision, show_progess=True, print_fields=True )
# particle_mass = data_dm['particle_mass'] #h^-1 Msun 
# density_dm = data_dm['density']          # h^2 Msun / kpc^3
# pos_x = data_dm['pos_x']                 #h^-1 kpc
# pos_y = data_dm['pos_y']                 #h^-1 kpc
# pos_z = data_dm['pos_z']                 #h^-1 kpc
# p_ids = data_dm['particle_IDs']      

