#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:42:47 2022

@author: castrojo
"""
import numpy as np
from modules import *
from modules import Units

#TODO: remove all references to Nparticles from the whole code

class Integrator(object):
    def __init__(self,temperature,time_step,force_field,C,mode=None):
        self.temperature = temperature
        self.time_step = time_step
        self.force_field = force_field
        self.C = C
        self.mode = mode
        self.mass = self.force_field.mass
        
        
    def PIMD_step(self, p, q, f, Nbeads, lambda_t=None):
        # Update momenta on first half-step
        p += 0.5 * f * self.time_step  # p: (Nbeads,)

        # Transformation to normal modes
        pn = self.C @ p
        qn = self.C @ q
        pn_prop = np.zeros_like(pn)
        qn_prop = np.zeros_like(qn)

        # Propagation in normal modes
        for k in range(Nbeads):
            Xk = Normal_Propagator(k, self.mass, Nbeads, self.temperature, self.time_step)
            vec_k = np.array([pn[k], qn[k]])
            vec_k_prop = Xk @ vec_k
            pn_prop[k] = vec_k_prop[0]
            qn_prop[k] = vec_k_prop[1]

        # Back-transformation to Cartesian coordinates
        p = self.C.T @ pn_prop
        q = self.C.T @ qn_prop

        # Calculate new forces and potential energies using new positions
        f, potential_energies, external_pot = self.force_field.eq_force_function(q, Nbeads)

        # Update momenta on second half-step
        p += 0.5 * f * self.time_step

        return p, q, f, potential_energies, external_pot

    

    def compute_energetics(self,q,p,qc,f,potential_energies,external_pot,Nbeads):        
        internal_pot = Internal_Potential(q, self.temperature,self.mass)
        pot = external_pot + internal_pot 

        #Calculate the temperature of the system and the Kinetic energy
        #system_T = Units(KbT,'hartree','KbT')      
        kin = (0.5*self.mass*np.sum(p**2))/Nbeads 
        #system_T = 2*E_kin
        KbT = np.sum(self.mass*(p**2))/Nbeads**2
        system_T = KbT       
        kin = (0.5*self.mass*np.sum(p**2))/Nbeads
                
        #system_T = Units(system_T,'KbT','hartree')  # Convert temperature to Kelvin
        #Classical total energy
        etot = kin+pot
        
        #QM estimators   
        qc, E_tot,E_kin,E_pot = E_Estimators(self.temperature, f, q, qc, potential_energies)
        return(pot,kin,etot,E_pot,E_kin,E_tot,system_T,internal_pot)
    
    
    def update_system(self,system,q,p,qc,f,system_T,pot,kin,etot,E_pot,E_kin,E_tot,internal_pot):

        #Update the system        
        system.set_velocities(p)        
        system.set_positions(q)
        system.set_qc(qc)
        system.set_forces(f)
        #TESTESTSET
        system.set_measured_temperature(system_T)
        
        #Update Classical energies
        system.set_pot(pot)
        system.set_kin(kin)
        system.set_tot(etot)
        
        #Update Energy estimators
        system.set_E_pot(E_pot)
        system.set_E_kin(E_kin)
        system.set_E_tot(E_tot)
        system.set_int_pot(internal_pot)   
        
            
    def step_NVE(self,system,lambda_t = None):
        q = system.get_positions()
        f = system.get_forces()
        Nbeads = system.Nbeads
        p = system.get_velocities()*self.mass #units of momenta
         
        #PIMD propagation step
        p,q,f,potential_energies,external_pot = self.PIMD_step(p,q,f,Nbeads,lambda_t)
          
        p = p/self.mass #Back to units of velocity

        #Position of the centroid
        qc = np.sum(q,axis=0)/Nbeads
        
        #Compute classic energies, estimator energies and temperature of the system
        pot,kin,etot,E_pot,E_kin,E_tot,system_T,internal_pot = self.compute_energetics(q,p,qc,f,potential_energies,external_pot,Nbeads)
        
        self.update_system(system, q, p, qc, f, system_T, pot, kin, etot, E_pot, E_kin, E_tot, internal_pot)
        

    def step_NVT_L(self, system, thermostat, lambda_t=None):
        q = system.get_positions()        # (Nbeads,)
        f = system.get_forces()          # (Nbeads,)
        Nbeads = system.Nbeads
        p = system.get_velocities() * self.mass  # (Nbeads,)

        # Thermalization of momenta - 1st half-step
        pn = self.C @ p
        pn_t = thermostat.PILE_L(pn, self.time_step, self.mass)
        p = self.C.T @ pn_t

        # PIMD propagation step
        p, q, f, potential_energies, external_pot = self.PIMD_step(p, q, f, Nbeads, lambda_t)

        # Thermalization of momenta - 2nd half-step
        
        pn = self.C @ p
        pn_t = thermostat.PILE_L(pn, self.time_step, self.mass)
        #T_nm = np.sum((pn_t**2) / self.mass) / Nbeads**2
        #T_nm = np.sum((pn_t**2) / self.mass) / Nbeads
        #system.set_measured_temperature(T_nm)
        p = self.C.T @ pn_t
        
        #TEST
        # Back to velocity units
        #p = p / self.mass

        # Centroid
        qc = np.sum(q) / Nbeads
        p = p / self.mass
        # Energetics
        pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = self.compute_energetics(
            q, p, qc, f, potential_energies, external_pot, Nbeads
        )
        
        self.update_system(system, q, p, qc, f, system_T, pot, kin, etot, E_pot, E_kin, E_tot, internal_pot)



    def step_NVT_G(self, system, thermostat, lambda_t=None):
        q = system.get_positions()                # (Nbeads,)
        f = system.get_forces()                  # (Nbeads,)
        Nbeads = system.Nbeads
        p = system.get_velocities() * self.mass  # (Nbeads,)

        # Compute K and General noise terms
        pn = self.C @ p
        pn_0 = pn[0]
        K = pn_0**2 / (2 * self.mass)

        # Thermalization of momenta - 1st half-step
        pn_t = thermostat.PILE_G(pn, self.time_step, self.mass, K)
        p = self.C.T @ pn_t

        # PIMD propagation step
        p, q, f, potential_energies, external_pot = self.PIMD_step(p, q, f, Nbeads, lambda_t)

        # Recompute K after propagation
        pn = self.C @ p
        pn_0 = pn[0]
        K = pn_0**2 / (2 * self.mass)

        # Thermalization of momenta - 2nd half-step
        pn_t = thermostat.PILE_G(pn, self.time_step, self.mass, K)
        p = self.C.T @ pn_t

        # Convert back to velocity units
        p = p / self.mass

        # Centroid
        qc = np.sum(q) / Nbeads

        # Energetics
        pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = self.compute_energetics(
            q, p, qc, f, potential_energies, external_pot, Nbeads
        )

        self.update_system(system, q, p, qc, f, system_T, pot, kin, etot, E_pot, E_kin, E_tot, internal_pot)

    
    
    def step_Andersen(self, system, thermostat, lambda_t=None):
        q = system.get_positions()                      # (Nbeads,)
        f = system.get_forces()                        # (Nbeads,)
        Nbeads = system.Nbeads
        p = system.get_velocities() * self.mass        # (Nbeads,)

        # Thermalize momenta on the first half-step
        p = thermostat.Andersen(p, self.time_step, self.mass)

        # PIMD propagation step
        p, q, f, potential_energies, external_pot = self.PIMD_step(p, q, f, Nbeads, lambda_t)

        # Thermalize momenta on the second half-step
        p = thermostat.Andersen(p, self.time_step, self.mass)

        p = p / self.mass  # Back to velocity units
        qc = np.sum(q) / Nbeads

        pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = self.compute_energetics(
            q, p, qc, f, potential_energies, external_pot, Nbeads
        )

        self.update_system(system, q, p, qc, f, system_T, pot, kin, etot, E_pot, E_kin, E_tot, internal_pot)

        
        
    def step_Langevin(self, system, thermostat, lambda_t=None):
        q = system.get_positions()
        f = system.get_forces()
        Nbeads = system.Nbeads
        p = system.get_velocities() * self.mass

        # Thermalize momenta on the first half-step
        p = thermostat.Langevin(p, self.time_step, self.mass)

        # PIMD propagation step
        p, q, f, potential_energies, external_pot = self.PIMD_step(p, q, f, Nbeads, lambda_t)

        # Thermalize momenta on the second half-step
        p = thermostat.Langevin(p, self.time_step, self.mass)

        p = p / self.mass
        qc = np.sum(q) / Nbeads

        pot, kin, etot, E_pot, E_kin, E_tot, system_T, internal_pot = self.compute_energetics(
            q, p, qc, f, potential_energies, external_pot, Nbeads
        )

        self.update_system(system, q, p, qc, f, system_T, pot, kin, etot, E_pot, E_kin, E_tot, internal_pot)
 