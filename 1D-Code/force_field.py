#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:48:50 2022

@author: castrojo
"""

import numpy as np
from modules import Units


data = np.loadtxt("spline_coeffs.dat", comments="#")
x_knots = data[:, 0]
c0, c1, c2, c3 = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
n_intervals = len(x_knots) - 1
x0 = x_knots[0]
xL = x_knots[-1]
L = xL - x0


class Force_Field(object):
    def __init__(self, potential, mass,):
        self.potential = potential
        self.mass = mass
        
    def ff(self, q, Nbeads, potential):
        potential_choice = str(self.potential)
        if potential_choice == 'harmonic':
            forces,potential_energies = self.harmonic(q,Nbeads)
        if potential_choice == 'double_well_voth':
            forces, potential_energies = self.double_well_voth(q,Nbeads)
        if potential_choice == 'oscillating_voth':
            forces, potential_energies = self.oscillating_voth(q,Nbeads)

        return forces, potential_energies
    
    def noneq_force_function(self, q, Nbeads, lambda_t):
        #print(lambda_t)
        potential_0 = self.potential[0].split()
        potential_1 = self.potential[1].split()
        
        forces_0,potential_energies_0 = self.ff(q, Nbeads, potential_0)
        forces_1,potential_energies_1 = self.ff(q, Nbeads, potential_1)
        
        forces = (lambda_t)*forces_1 + (1 - lambda_t)*forces_0
        potential_energies = (lambda_t)*potential_energies_1 + (1 - lambda_t)*potential_energies_0
        
        pot = (1/Nbeads)*np.sum(potential_energies)
        #print(pot)
        return forces, potential_energies, pot 

    def eq_force_function(self, q, Nbeads):
        potential = self.potential[0].split()
        
        forces, potential_energies = self.ff( q, Nbeads, potential)        

        pot = 1/Nbeads*np.sum(potential_energies)
        #print('forces ', forces)
        #print('potens ', potential_energies)
        # print('pot ', pot)
        return forces, potential_energies, pot

    
    def harmonic(self, q, Nbeads):
        param = 0.5
        forces = np.zeros(Nbeads, float)
        potential_energies = np.zeros(Nbeads, float)

        K = self.mass * (param**2)

        for k in range(Nbeads):
            bead_position = q[k]
            forces[k] = -K * bead_position
            potential_energies[k] = 0.5 * K * (bead_position**2)

        return forces, potential_energies

    def double_well_voth(self, q, Nbeads):
        # from JCTC 10, 3634 (2014)
        forces = np.zeros(Nbeads, float)
        potential_energies = np.zeros(Nbeads, float)

        a = -0.03123777    # hartree/bohr^2
        b = 0.00023614737  # hartree/bohr^3
        c = 0.031240952    # hartree/bohr^4

        for k in range(Nbeads):
            bead_position = q[k]
            forces[k] = -2 * a * bead_position \
                        -3 * b * bead_position**2 \
                        -4 * c * bead_position**3

            potential_energies[k] = (
                a * bead_position**2 +
                b * bead_position**3 +
                c * bead_position**4
            )

        return forces, potential_energies
    
    def oscillating_voth(self, q, Nbeads):
        forces = np.zeros(Nbeads, float)
        Venergies = np.zeros(Nbeads, float)

        for k in range(Nbeads):
            x = q[k]

            i = np.searchsorted(x_knots, x, side="right") - 1
            i = int(np.clip(i, 0, n_intervals - 1))
            dx = x - x_knots[i]

            V  = ((c0[i] * dx + c1[i]) * dx + c2[i]) * dx + c3[i]
            dV = (3.0 * c0[i] * dx + 2.0 * c1[i]) * dx + c2[i]

            Venergies[k] = V
            forces[k] = -dV

        return forces, Venergies



        
    def harmonic_PE(self,potential,temperature):
        param = float(potential[1])
        V_ho = (param/4)*(1/np.tanh(param/(2*Units(temperature,'KbT','hartree'))))
        print('Analytical PE at ' + str(temperature) + 'K : ',V_ho)
        return(V_ho)
    
    def get_potential_func(self):
        potential_choice = self.potential

        if potential_choice == 'double_well_voth':
            a = -0.03123777
            b = 0.00023614737
            c = 0.031240952
            return lambda x: a * x**2 + b * x**3 + c * x**4

        elif potential_choice == 'harmonic':
            K = self.mass * (0.5**2)
            return lambda x: 0.5 * K * x**2

        elif potential_choice == 'oscillating_voth':
            n_intervals = len(x_knots) - 1

            def V(x):
                x = np.atleast_1d(x)
                Vout = np.zeros_like(x)

                for idx, val in enumerate(x):
                    i = np.searchsorted(x_knots, val, side="right") - 1
                    i = int(np.clip(i, 0, n_intervals - 1))
                    dx = val - x_knots[i]
                    Vout[idx] = ((c0[i] * dx + c1[i]) * dx + c2[i]) * dx + c3[i]

                return Vout if len(Vout) > 1 else Vout[0]

            return V

        else:
            raise ValueError(f"Unbekanntes Potential: {potential_choice}")