#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:18:17 2022

@author: castrojo
"""

import numpy as np
from integrator import Integrator
from modules import *
from file_recorder import Recorder
from velocity_read import *

class Simulation(object):
    #Constructor
    def __init__(self,system,temperature,time_step,force_field,thermostat,mode,name_file,index,rank,size):

        self.initializer(system,temperature,time_step,force_field,thermostat,mode,name_file,index,rank,size)

    def initializer(self,system,temperature,time_step,force_field,thermostat,mode,name_file,index,rank,size):
        self.system = system
        self.temperature = temperature
        self.Nbeads = system.Nbeads
        self.time_step = time_step
        self.force_field = force_field
        self.thermostat = thermostat
        self.mode = mode
        self.index = index
        self.name_file = name_file      
        beta = 1/temperature
        if self.Nbeads == 1:
            self.chi = ChiTracker(force_field.get_potential_func(), beta, xmin=-0.4, xmax=20.4, nbins=200)
            #self.chi = ChiTracker(force_field.get_potential_func(), beta, xmin=-1.5, xmax=1.5, nbins=200)
        else:
            self.chi = ChiTrackerPIMD(force_field.get_potential_func(), beta, xmin=-0.4, xmax=20.4, nbins=200, P=self.Nbeads, m=self.force_field.mass)
            #self.chi = ChiTrackerPIMD(force_field.get_potential_func(), beta, xmin=-1.5, xmax=1.5, nbins=200, P=self.Nbeads, m=self.force_field.mass)
        mass = self.force_field.mass
        #Read positions from file
        
        #Read from a single file
        if mode == 'in':
            pos_file_list = np.array([name_file + '_pos.in'])
        
        #Read from Nbeads files
        
        if mode == 'inN' or mode == 'noneq':
            pos_file_list = np.empty(self.Nbeads,dtype=object)
            for i in range(self.Nbeads):
                pos_file_list[i] = name_file + '_pos_' + str(i) +'.in'            
        
        positions = self.system.build_q(pos_file_list,1)
        self.system.set_positions(positions)
        #init_qc = positions
        init_qc = self.system.build_qc(pos_file_list,1)        
        #Generate velocities from MB-distribution
        
        #HIER 
        #std = np.sqrt(self.temperature / self.force_field.mass)
        std = (Units(self.temperature,'KbT','hartree')/mass)**0.5
        mean = 0.0
        MB_velocities = np.random.normal(mean, std, self.Nbeads)
        self.system.set_velocities(MB_velocities)
                    
        #Initialize the ring polymer of each particle
        for i in range(self.Nbeads):
            system.set_bead_velocity(MB_velocities[np.random.randint(0, len(MB_velocities))], i)
        
        init_q = self.system.get_positions()
        init_p = self.system.get_velocities()
                    
        if mode == 'noneq':
            init_f,init_potential_energies,init_pot = self.force_field.noneq_force_function(init_q,self.Nbeads,lambda_t=0)
        else:
            init_f,init_potential_energies,init_pot = self.force_field.eq_force_function(init_q,self.Nbeads)
        
        
        #init_kin = 0.5 * mass * np.sum(init_p**2) / self.Nbeads
        #T_eff = 2 * init_kin
        #HIER
        KbT = np.sum(mass*(init_p**2))/self.Nbeads**2##
        init_system_T = Units(KbT,'hartree','KbT')
        init_kin = 0.5*mass*np.sum(init_p**2)/self.Nbeads##
        #print((init_qc))
        init_qc, init_E_tot, init_E_kin, init_E_pot = E_Estimators(temperature, init_f, init_q, init_qc, init_potential_energies)
        #self.system.set_velocities(p)##
        self.system.set_forces(init_f)
        self.system.set_pot(init_pot)
        self.system.set_kin(init_kin)
        self.system.set_tot(init_pot+init_kin)
        self.system.set_E_pot(init_E_pot)
        self.system.set_E_kin(init_E_kin)
        self.system.set_E_tot(init_E_tot)
        self.system.set_potential_energies(init_potential_energies)
        self.system.set_target_temperature(temperature)          
        self.system.set_measured_temperature(init_system_T)             

        # Initialize the file recorder only at the start
        if index % size == rank: 
            self.rec = Recorder(self.system, self.name_file, index, rank, size)
        else:
            self.rec = None

    def run_PIMD(self,nsteps,index,step):
        #Normal modes transformation matrix
        C = C_matrix(self.Nbeads)
        counter = 0
        time_counter = step*self.time_step
        #get the initial forces
        #Call the integrator

        #self.temperature = self.thermostat.temperature #JNM 27.12.25
        integrator = Integrator(self.temperature,self.time_step,self.force_field,C,self.mode)

        #if self.rec is not None:
        #    current_step = step + counter
        #    time = current_step * self.time_step
        #self.rec.write(self.system,counter,time)

        while counter < nsteps:

            if self.Nbeads > 1:
                self.chi.update(self.system.q,rep_id=index)
            else:
                #index statt self.index
                self.chi.update(self.system.q[0],rep_id=index)
            counter += 1
            lambda_t = counter/nsteps
            #print(lambda_t)
            #print('input',self.system.get_velocities())
            if str(self.thermostat.thermostat_type) == 'PILE_L':
                integrator.step_NVT_L(self.system,self.thermostat,lambda_t)
                
            elif str(self.thermostat.thermostat_type) == 'PILE_G':
                integrator.step_NVT_G(self.system,self.thermostat,lambda_t)
            
            elif str(self.thermostat.thermostat_type) == 'Andersen':
                integrator.step_Andersen(self.system,self.thermostat,lambda_t)

            elif str(self.thermostat.thermostat_type) == 'Langevin':
                integrator.step_Langevin(self.system,self.thermostat,lambda_t) 

            else:
                integrator.step_NVE(self.system,lambda_t)
          
            #print('ouput',self.system.get_velocities())                    
            time_counter += self.time_step
            #print(self.system.qc)
            #print(self.system.Temperature)
            if self.rec is not None:
                current_step = step + counter
                time = current_step * self.time_step
                self.rec.write(self.system,counter,time)
        


    def reassign_values(self, system, q, p, temperature):

        self.system = system
        self.system.set_positions(q)
        self.system.set_velocities(p)


        f, potential_energies, pot = self.force_field.eq_force_function(q, self.Nbeads)


        kin = 0.5 * self.force_field.mass * np.sum(p ** 2) / self.Nbeads

        qc = 0.0  
        #qc, E_tot, E_kin, E_pot = E_Estimators(temperature, f, q, qc, potential_energies)
        qc, E_tot, E_kin, E_pot = E_Estimators(self.temperature, f, q, qc, potential_energies)
        #TESTETST
        #T_eff = 2 * kin
        self.system.set_target_temperature(temperature)
        self.system.set_forces(f)
        self.system.set_pot(pot)
        self.system.set_kin(kin)
        self.system.set_tot(pot + kin)
        self.system.set_E_pot(E_pot)
        self.system.set_E_kin(E_kin)
        self.system.set_E_tot(E_tot)

  



    def QM_Energy_Estimator(self,Energy_file,nsteps):
        print('Estimator data from: ', Energy_file)
        #print('lptm',nsteps)
        E_TOT_arr = np.zeros(nsteps+1)
        E_KIN_arr = np.zeros(nsteps+1)
        E_POT_arr = np.zeros(nsteps+1)

        with open (Energy_file,'r') as f:
            next(f)
            step = 0
            for line in f:
                p = line.split()
                E_TOT_arr[step] = float(p[1])
                E_KIN_arr[step] = float(p[2])
                E_POT_arr[step] = float(p[3])
                step += 1
        
        E_TOT = np.average(E_TOT_arr)
        E_KIN = np.average(E_KIN_arr)
        E_POT = np.average(E_POT_arr)

        print('Total Energy: ', E_TOT)
        print('Kinetic Energy: ', E_KIN)
        print('Potential Energy: ', E_POT)
    
        return(E_TOT,E_KIN,E_POT)
    


    