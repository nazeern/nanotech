# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:06:20 2023

@author: Czaja
"""
import numpy as np

#Nanotube geometry
CNT_radius = 5.0E-09 #m
CNT_height = 2.5E-04 #m

#Nanotube forest geometry
nanostructure_width = 3.0E-06 #m
nanostructure_length = 2.54E-02 #m
gap_between_nanostructures = 2.0E-06 #m

#Chip geometry
chip_length = 2.54E-02 #1 inch
chip_width = 2.54E-02 #1 inch

#Dielectric properties
epsilon_r = 1000 #average dielectric constant between two forests
E_breakdown = 1.2E+09 # dielectric breakdown E-field [V/m]

#Substrate geometry
Si_thickness = 3.0E-04 #m
SiO2_thickness = 2.1E-06 #m
Metal_1_thickness = 1.0E-07 #m
Metal_2_thickness = 1.0E-08 #m
Catalyst_thickness = 1.0E-08 #m
CNT_layer_thickness = CNT_height

#Physical constants (SI units)
e = 1.602e-19  #electron charge
z = 1 #electrons/surface particle
C = 1.0E-15
C_0 = 1.0E-12
e_r = 78.49 #dielctric constant
e_0 = 8.854E-12 #vacuum permittivity
k_b= 1.38E-23 #Boltzmann const
T = 298.1 #room temperature
V_zeta = 5.0E-02
N_a = 6E+23

def calculate_edl_capacitance_over_a():
    return np.sqrt((2 * z**2 * e**2 * C_0 * e_r * e_0)/(k_b * T)) * np.cosh(z * e * V_zeta / (2 * k_b * T))

def calculate_d_edl():
    return np.sqrt((k_b * T * e_0 * e_r) / (2 * (z * e)**2 * N_a * C_0))

def calculate_edl_capacitance(d_edl):
    return e_0 * e_r / d_edl

C_edl_over_A = calculate_edl_capacitance_over_a()

C_edl = calculate_edl_capacitance(calculate_d_edl())


