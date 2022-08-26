import os, sys, time
import h5py as h5
import numpy as np
from load_data import get_domain_block
root_dir = os.path.dirname(os.getcwd())
tools_dir = root_dir + 'tools'
sys.path.append( tools_dir )
from tools import check_if_file_exists, print_line_flush

def get_yt_field( field, data, current_a, h ):
  print( f'Loading YT Field: {field}')
  if field == 'mass':  data_field = data[('all', 'particle_mass')].in_units('msun')*h
  if field == 'pos_x': data_field = data[('all', 'particle_position_x')].in_units('kpc')/current_a*h
  if field == 'pos_y': data_field = data[('all', 'particle_position_y')].in_units('kpc')/current_a*h
  if field == 'pos_z': data_field = data[('all', 'particle_position_z')].in_units('kpc')/current_a*h
  if field == 'vel_x': data_field = data[('all', 'particle_velocity_x')].in_units('km/s')
  if field == 'vel_y': data_field = data[('all', 'particle_velocity_y')].in_units('km/s')
  if field == 'vel_z': data_field = data[('all', 'particle_velocity_z')].in_units('km/s')
  return data_field
  
def generate_ics_particles_distributed_single_field( field, particles_domain_indices, proc_grid, grid_size, output_dir, ds, data  ):
  h = ds.hubble_constant
  current_z = np.float(ds.current_redshift)
  current_a = 1./(current_z + 1)

  n_procs = proc_grid[0]*proc_grid[1]*proc_grid[2]
  data_field = get_yt_field( field, data, current_a, h )
  if field == 'mass': 
    particle_mass = data_field[0].v
    print(f' Particle Mass: {particle_mass:.3e} Msun/h')
  n_local_all = []
  for proc_id in range(n_procs):
    indices_local = np.where( particles_domain_indices == proc_id )[0]
    n_local = len( indices_local ) 
    n_local_all.append( n_local )
    data_local = data_field[indices_local]
    out_file_name = output_dir + f'temp_{field}_{proc_id}.h5'
    out_file = h5.File( out_file_name, 'w' )
    if field == 'mass':  out_file.attrs['particle_mass'] = particle_mass    
    out_file.attrs['current_a'] = current_a
    out_file.attrs['current_z'] = current_z
    out_file.attrs['n_particles_local'] = n_local
    out_file.create_dataset( field, data=data_local )
    out_file.close()
    print( f' proc_id:{proc_id}  n_local: {n_local}  File: {out_file_name}' )
  print( f'Total Particles Saved: {sum(n_local_all)}  Grid_size: {np.prod(grid_size)}' )
  time.sleep(2)
  data_field, data_local = None, None


def Merge_Particles_Fileds( field_list, proc_grid, grid_size, output_dir, output_base_name = 'particles.h5', n_snapshot=0 ):

  n_procs = proc_grid[0]*proc_grid[1]*proc_grid[2]
  n_total = []
  for proc_id in range(n_procs):
    print('\nproc_id: ', proc_id)
    
    # Create the final output file
    out_file_name = f'{output_dir}{n_snapshot}_{output_base_name}.{proc_id}'
    out_file = h5.File( out_file_name, 'w' )
    
    n_local_all = []
    # field = 'mass'
    for field in field_list:
      #Load the field data
      file_name = f'{output_dir}temp_{field}_{proc_id}.h5'
      # print(f' Loading Field: {field}     File: {file_name}' )
      in_file = h5.File( file_name, 'r' )
      data_field = in_file[field][...]
      n_local = in_file.attrs['n_particles_local']
      n_local_all.append( n_local )
    
      if field == 'mass':
        n_total.append( in_file.attrs['n_particles_local'] )
        for key in list(in_file.attrs.keys()):
          out_file.attrs[key] = in_file.attrs[key]
        print(f' Saved Attrs')
        for key in list(out_file.attrs.keys()):
          print( f' {key}: {out_file.attrs[key]} ')
    
      print(f' Writing Field: {field}   n_local: {n_local}   ' )
      out_file.create_dataset( field, data=data_field )
      in_file.close()
    
    if np.min(n_local_all) != np.max(n_local_all): 
      print('ERROR: n_local mismatch')
      extit()
    
    out_file.close()
    print('Saved File: ', out_file_name)
    time.sleep(0.1)
  print( f'\nTotal Partilces Saved: {np.sum(n_total)}   Grid_size: {np.prod(grid_size)} ') 


def Get_PID_Indices_axis( key_pos, domain, data, ds, outputDir, func='load', type_int=np.int16  ):
  
  file_name = outputDir + f'temp_indices_{key_pos}.h5'
  if func == 'load':
    file_exists = check_if_file_exists( file_name )
    if file_exists:
       print( f'Loading File: {file_name} ')
       file_temp_indx = h5.File( file_name , 'r')
       pid_indxs = file_temp_indx['pid_indxs'][...]
       return pid_indxs
    else:
      func = 'save'
  
  keys_domain = { 'pos_x':['x', 'dx'], 'pos_y':['y', 'dy'], 'pos_z':['z', 'dz'] }
  keys_data = { 'pos_x':'particle_position_x', 'pos_y':'particle_position_y', 'pos_z':'particle_position_z' }

  key_domain, key_delta = keys_domain[key_pos]
  delta = domain['global'][key_delta]

  key_data = keys_data[key_pos]
  h = ds.hubble_constant
  current_z = np.float(ds.current_redshift)
  current_a = 1./(current_z + 1)
  pos = data[('all', key_data)].in_units('kpc').v/current_a*h
  n_total = len(pos)
  print(' Getting Indices: {0}'.format(key_domain))
  print('  N total: {0}'.format(n_total))
  pid_indxs = ( pos / delta ).astype( type_int )

  if func == 'save':
    # Temporal file to save the indices
    file_temp_indx = h5.File( file_name , 'w')
    file_temp_indx.create_dataset( 'pid_indxs', data=pid_indxs )
    file_temp_indx.close()
    print('  Saved Indices: {0}'.format(key_domain))
  return pid_indxs



def Compute_Particles_Domain_Indices( box_size, grid_size, proc_grid, data, ds, output_dir, type_int=np.int16 ):
  
  file_name = output_dir + 'temp_indices_global.h5'
  file_exists = check_if_file_exists( file_name )
  if file_exists:
    print( f'Loading File: {file_name} ')
    infile = h5.File( file_name, 'r' )
    indices_global = infile['indices_global'][...]
    infile.close()
    return indices_global
  
  # Get the domain decomposition
  domain =  get_domain_block( proc_grid, box_size, grid_size )

  pos_keys = [ 'pos_x', 'pos_y', 'pos_z' ]
  indices_x = Get_PID_Indices_axis( 'pos_x', domain, data, ds, output_dir, func='load', type_int=type_int  )
  indices_y = Get_PID_Indices_axis( 'pos_y', domain, data, ds, output_dir, func='load', type_int=type_int  )
  indices_z = Get_PID_Indices_axis( 'pos_z', domain, data, ds, output_dir, func='load', type_int=type_int  )

  print('Computing Particles Global Indices ')
  indices_global = indices_x + indices_y * proc_grid[0] + indices_z*proc_grid[0]*proc_grid[1]
  #Free the indices memory
  indices_x, indices_y, indices_z = None, None, None
  outfile = h5.File( file_name, 'w' )
  outfile.create_dataset( 'indices_global', data=indices_global )
  outfile.close()
  print (f'Saved File: {file_name} ')
  return indices_global




def generate_ics_particles( data_in, outDir, outputBaseName, proc_grid, box_size, grid_size ):
  
  print( '\nGenerating ICs: Particles' )
  domain = get_domain_block( proc_grid, box_size, grid_size )

  current_a = data_in['current_a']
  current_z = data_in['current_z']
  # box_size = data_in['box_size']

  data = data_in['dm']
  pos_x = data['pos_x'][...]
  pos_y = data['pos_y'][...]
  pos_z = data['pos_z'][...]
  vel_x = data['vel_x'][...]
  vel_y = data['vel_y'][...]
  vel_z = data['vel_z'][...]
  # mass = data['mass'][...]
  # particle_mass = mass[0]
  particle_mass = data['p_mass']
  nPart = pos_x.shape[0]
  ids = np.arange(nPart).astype(np.int64)
  print('N total particles: ', nPart)

  dx = domain[0]['box']['dx']
  dy = domain[0]['box']['dy']
  dz = domain[0]['box']['dz']
  
  # print(( dx, dy, dz))
  print('Selecting local indices')
  index_x = ( pos_x / dx ).astype(np.int)
  index_y = ( pos_y / dy ).astype(np.int)
  index_z = ( pos_z / dz ).astype(np.int)
  indexs = index_x + index_y * proc_grid[0] + index_z*proc_grid[0]*proc_grid[1]


  n_local_all = []
  nprocs = proc_grid[0] * proc_grid[1] * proc_grid[2]
  for pId in range(nprocs):

    outputFileName = outDir + outputBaseName + ".{0}".format(pId)
    if nprocs == 1: outputFileName = outDir + outputBaseName 
    # print(' Writing h5 file: ', outputFileName)
    outFile = h5.File( outputFileName, 'w')
    outFile.attrs['current_a'] = current_a
    outFile.attrs['current_z'] = current_z
    outFile.attrs['particle_mass'] = particle_mass

    indx = np.where(indexs == pId)[0]
    n_local = len(indx)
    print_line_flush(f'Writing file: {outputBaseName}.{pId}  n_local: {n_local}')    
    n_local_all.append(n_local)
    pos_x_l = pos_x[indx]
    pos_y_l = pos_y[indx]
    pos_z_l = pos_z[indx]
    vel_x_l = vel_x[indx]
    vel_y_l = vel_y[indx]
    vel_z_l = vel_z[indx]
    # mass_l = mass[indx]
    ids_l = ids[indx]
    # print('  n_local: ', n_local)
    # print('  Current_a: ', current_a)
    outFile.attrs['n_particles_local'] = n_local
    # outFile.attrs['N_DM_file'] = np.float(nPart)
    # outFile.create_dataset( 'mass', data=mass_l )
    outFile.create_dataset( 'pos_x', data=pos_x_l.astype(np.float64) )
    outFile.create_dataset( 'pos_y', data=pos_y_l.astype(np.float64) )
    outFile.create_dataset( 'pos_z', data=pos_z_l.astype(np.float64) )
    outFile.create_dataset( 'vel_x', data=vel_x_l.astype(np.float64)  )
    outFile.create_dataset( 'vel_y', data=vel_y_l.astype(np.float64)  )
    outFile.create_dataset( 'vel_z', data=vel_z_l.astype(np.float64)  )
    outFile.create_dataset( 'particle_IDs', data=ids_l.astype(np.int64)  )

    outFile.close()
    print('')
  print( f'Total Particles Saved: {sum(n_local_all)} / {nPart}' )
