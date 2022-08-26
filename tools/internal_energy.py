import sys
import numpy as np

from constants_cgs import *


def get_temperature( u, gamma=5./3, mu=None ):
  temp = (gamma - 1) * M_p / K_b * u * 1e10
  if mu is not None : temp *= mu
  return temp

def get_internal_energy( temp, gamma=5./3, mu=1 ):
  u = temp / (gamma - 1) * K_b / M_p * 1e-10 / mu
  return u

