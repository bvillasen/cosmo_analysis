import numpy as np
from constants_cosmo import Mpc, Myear, Gcosmo, Msun, kpc
from rk4 import RK4_step


class Cosmology:
  
  def __init__(self, z_start=100, get_a=False ):
    # Initializa Planck 2018 parameters
    self.H0 = 67.66
    self.Omega_M = 0.3111
    self.Omega_L = 0.6889
    self.Omega_b = 0.0497
    self.h = self.H0 / 100.
    self.rho_crit =  3*(self.H0*1e-3)**2/(8*np.pi* Gcosmo) * Msun / (kpc*100)**3 #kg cm^-3
    self.rho_gas_mean = self.rho_crit * self.Omega_b / Msun * (kpc*100)**3  / self.h**2  #kg cm^-3
    self.z_start = z_start
    self.n_points = 1000000
    self.z_array = None
    self.a_array = None
    self.t_array = None
    
    if get_a: self.integrate_scale_factor()
    
  def get_Hubble( self, current_a ):
      a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  )  
      H = a_dot / current_a
      return H
  
  def get_dt( self, current_a, delta_a ):
    a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  ) * 1000 / Mpc
    dt = delta_a / a_dot
    return dt  
    
  def get_delta_a( self, current_a, dt ):
    a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  ) * 1000 / Mpc
    delta_a = dt * a_dot 
    return delta_a  
    
  def a_deriv( self, time, current_a, kargs=None ):
    a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  ) * 1000 / Mpc
    return a_dot
    
  def get_current_a( self, time ):
    current_a = np.interp( time, self.t_vals, self.a_vals )
    return current_a
    
  def integrate_scale_factor(self):
    print( 'Integrating Scale Factor')
    z_vals = [self.z_start]
    a_vals = [1./(self.z_start+1) ]
    t_vals = [0]
    dt = 0.1 * Myear
    while a_vals[-1] < 1.0:
      a = RK4_step( self.a_deriv, t_vals[-1], a_vals[-1], dt )
      a_vals.append( a )
      z_vals.append( 1/a - 1 )
      t_vals.append( t_vals[-1] + dt)
    self.t_vals = np.array( t_vals )
    self.a_vals = np.array( a_vals )
    self.z_vals = np.array( z_vals )
    
    
      