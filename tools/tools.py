import os, sys
from os import listdir
from os.path import isfile, join
import numpy as np
import h5py as h5
import time

system = None
system = os.getenv('SYSTEM_NAME')
if not system:
  print( 'Can not find the system name')
  exit(-1)
# print( f'System: {system}')

if system == 'Eagle':    data_dir = '/media/bruno/ssd_bruno/data/'
# if system == 'Eagle':    data_dir = '/home/bruno/Desktop/data/'
if system == 'Tornado':  data_dir = '/home/bruno/Desktop/ssd_0/data/'
if system == 'Shamrock': data_dir = '/raid/bruno/data/'
if system == 'Lux':      data_dir = '/data/groups/comp-astro/bruno/'
if system == 'xps':      data_dir = '/home/bruno/Desktop/data/'
if system == 'Mac_mini': data_dir = '/Users/bruno/Desktop/data/'
if system == 'MacBook':  data_dir = '/Users/bruno/Desktop/data/'
# if system == 'Summit':   data_dir = '/gpfs/alpine/ast169/scratch/bvilasen/'
if system == 'Summit':   data_dir = '/gpfs/alpine/ast175/scratch/bvilasen/'

if system == 'Lux':      home_dir = '/home/brvillas/'
if system == 'Summit':   home_dir = '/ccs/home/bvilasen/'
if system == 'Tornado':  home_dir = '/home/bruno/'
if system == 'Shamrock': home_dir = '/home/bruno/'
if system == 'xps':      home_dir = '/home/bruno/'
if system == 'Mac_mini': home_dir = '/Users/bruno/'
if system == 'MacBook':  home_dir = '/Users/bruno/'

projects_dir = home_dir + 'Desktop/Dropbox/projects/'
# projects_dir = home_dir + 'Desktop/Dropbox/projects/'



def select_closest_index( val, vals, tol=1e-3 ):
  diff = np.abs( vals - val )
  if diff.min() > tol: return None, None
  indxs = np.where( diff == diff.min() )[0]
  if len(indxs) > 1: print( f'WARNING:L More than one closest indx found: val:{val}  closest:{vals[indxs]}')
  indx = indxs[0]
  return indx, vals[indx]
  
def select_interval( interval, x_vals ):
  x_min, x_max = interval
  indices = ( x_vals >= x_min ) * ( x_vals <= x_max )
  indices = np.where( indices == True )[0]
  return indices



def Select_Indices( x_to_select, x_vals, tolerance=1e-3 ):
  indices = []
  for x in x_to_select:
    diff = np.abs( x_vals - x )
    diff_min = diff.min()
    if diff_min > tolerance:
      print( f'ERROR: No index found for {x}, min difference is : {diff_min}')
      return None
    indx = np.where( diff == diff_min )[0]
    if len( indx ) > 1:
      print( f'WARNING:Multiple values found for {x}, found: {x_vals[indx]}. Only selected: {x_vals[indx[0]]}   ' )
    indx = indx[0]
    indices.append( indx )
  indices = np.array( indices )
  selected_vals = x_vals[indices]
  n_to_select = len( x_to_select )
  selected_diff = 0
  for i in range(n_to_select):
    selected_diff += np.abs( x_vals[indices[i]] - x_to_select[i] )
  if selected_diff > (n_to_select * tolerance) :
    print( f'ERROR: Selected values dosent match the values to select' )
    return None
  return indices
  

def Combine_List_Pair( a, b ):
  output = []
  for a_i in a:
    for b_i in b:
      if type(b_i) == list:
        add_in = [a_i] + b_i
      else:
        add_in = [ a_i, b_i ]
      output.append( add_in )
  return output

def Get_Parameters_Combination( param_vals ):
  n_param = len( param_vals )
  indices_list = []
  for i in range(n_param):
    param_id = n_param - 1 - i
    n_vals =  len(param_vals[param_id]) 
    indices_list.append( [ x for x in range(n_vals)] )
  param_indx_grid = indices_list[0]
  for i in range( n_param-1 ):
    param_indx_grid = Combine_List_Pair( indices_list[i+1], param_indx_grid )
  param_combinations = []
  for param_indices in param_indx_grid:
    p_vals = [ ]
    for p_id, p_indx in enumerate(param_indices):
      p_vals.append( param_vals[p_id][p_indx] )
    param_combinations.append( p_vals )
  return param_combinations
  
  
def print_progress( i, n, time_start, extra_line="" ):
  import time
  time_now = time.time()
  time = time_now - time_start
  if i == 0: remaining = time *  n
  else: remaining = time * ( n - i ) / i

  hrs = remaining // 3600
  min = (remaining - hrs*3600) // 60
  sec = remaining - hrs*3600 - min*60
  etr = f'{hrs:02.0f}:{min:02.0f}:{sec:02.0f}'
  progres = f'{extra_line}Progress:   {i}/{n}   {i/n*100:.1f}%   ETR: {etr} '
  print_line_flush (progres )

def Get_Free_Memory( print_out=False):
  import psutil
  mem = psutil.virtual_memory()
  free = mem.free / 1e9
  if print_out: print( f'Free Memory: {free:.1f} GB' )
  return free 
  
def check_if_file_exists( file_name ):
  return os.path.isfile( file_name )
  

def Load_Pickle_Directory( input_name, print_out=True ):
  import pickle
  if print_out: print( f'Loading File: {input_name}')
  dir = pickle.load( open( input_name, 'rb' ) )
  return dir
  
def Write_Pickle_Directory( dir, output_name ):
  import pickle 
  f = open( output_name, 'wb' )
  pickle.dump( dir, f)
  print ( f'Saved File: {output_name}' )


def split_array_mpi( array, rank, n_procs, adjacent=False ):
  n_index_total = len(array)
  n_proc_indices = (n_index_total-1) // n_procs + 1
  indices_to_generate = np.array([ rank + i*n_procs for i in range(n_proc_indices) ])
  if adjacent: indices_to_generate = np.array([ i + rank*n_proc_indices for i in range(n_proc_indices) ])
  else: indices_to_generate = np.array([ rank + i*n_procs for i in range(n_proc_indices) ])
  indices_to_generate = indices_to_generate[ indices_to_generate < n_index_total ]
  return array[indices_to_generate]

def split_indices( indices, rank, n_procs, adjacent=False ):
  n_index_total = len(indices)
  n_proc_indices = (n_index_total-1) // n_procs + 1
  indices_to_generate = np.array([ rank + i*n_procs for i in range(n_proc_indices) ])
  if adjacent: indices_to_generate = np.array([ i + rank*n_proc_indices for i in range(n_proc_indices) ])
  else: indices_to_generate = np.array([ rank + i*n_procs for i in range(n_proc_indices) ])
  indices_to_generate = indices_to_generate[ indices_to_generate < n_index_total ]
  return indices_to_generate

def extend_path( dir=None ):
  if not dir: dir = os.getcwd()
  subDirectories = [x[0] for x in os.walk(dir) if x[0].find('.git')<0 ]
  sys.path.extend(subDirectories)


def print_mpi( text, rank, size,  mpi_comm):
  for i in range(size):
    if rank == i: print( text )
    time.sleep( 0.01 )
    mpi_comm.Barrier()

def print_line_flush( terminalString ):
  terminalString = '\r' + terminalString
  sys.stdout. write(terminalString)
  sys.stdout.flush() 


def create_directory( dir, print_out=True, print_status=False ):
  if print_out: print(("Creating Directory: {0}".format(dir) ))
  indx = dir[:-1].rfind('/' )
  inDir = dir[:indx]
  dirName = dir[indx:].replace('/','')
  dir_list = next(os.walk(inDir))[1]
  if dirName in dir_list: 
    if print_status: print( " Directory exists")
  else:
    os.mkdir( dir )
    if print_status: print( " Directory created")


def get_files_names( inDir, fileKey='',  type=None ):
  if not type: dataFiles = [f for f in listdir(inDir) if isfile(join(inDir, f)) ]
  if type=='nyx': dataFiles = [f for f in listdir(inDir) if (f.find(fileKey) >= 0 )  ]
  if type == 'cholla': dataFiles = [f for f in listdir(inDir) if (isfile(join(inDir, f)) and (f.find(fileKey) >= 0 ) ) ]
  dataFiles = np.sort( dataFiles )
  nFiles = len( dataFiles )
  # index_stride = int(dataFiles[1][len(fileKey):]) - int(dataFiles[0][len(fileKey):])
  if not type: return dataFiles
  if type == 'nyx': return dataFiles, nFiles
  if type == 'cholla': return dataFiles, nFiles
