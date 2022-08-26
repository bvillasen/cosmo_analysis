# gravitational constant, kpc km^2 s^-2 Msun^-1
G_COSMO =  4.300927161e-06



eV_to_ergs = 1.60218e-12


#Boltazman constant
# K_b = 1.38064852e-23 #m2 kg s-2 K-1
K_b = 1.380649e-16 #cm2 g s-2 K-1  (ergs/K)


#Mass of proton
M_p = 1.6726219e-27 #kg
M_p_cgs = M_p * 1000 
M_e = 9.10938356e-31 #kg
e_charge = 1.60217662e-19 # Coulombs 

c = 299792000.458 # velocity of light in m/sec
# pc = 3.086e13  #km
pc = 3.0857e16  #m
# pc = 3.08567758128e16  #m
kpc = 1e3 * pc
Mpc = 1e6 * pc
Msun = 1.98847e30  #kg
# Msun = 1.9889200011445836e30  #kg
Myear = 365 * 24 * 3600 * 1e6


Gconst = 6.6740831e-11 #m3  s-2 kg-1
# Gconst = 6.6743015e-11 #m3  s-2 kg-1

Gcosmo = Gconst * ( 1./kpc) * 1./1000 * 1./1000 * Msun  # kpc km^2 s^-2 Msun^-1