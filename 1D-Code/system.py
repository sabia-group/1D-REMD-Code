#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:10:14 2022

@author: castrojo
"""

import numpy as np
import linecache

class System(object):
    '''
    Class to hold the system properties of a PIMD simulation.
    It contains the positions, momenta, forces, potential energies, kinetic energies,
    and other properties of the beads in the ring polymer.
    '''
    
    def __init__(self,Nbeads):

        self.initialize(Nbeads)
    
    def initialize(self,Nbeads):

        self.Nbeads = Nbeads

        #Positions and momenta of the beads
        self.q = np.zeros(Nbeads,float)
        self.p = np.zeros(Nbeads,float)

        #forces of the beads
        self.f = np.zeros(Nbeads,float)
        
        #potential energies of the beads
        self.potential_energies = np.zeros(Nbeads,float)

        #kinetic energies of the beads
        self.kinetic_energies = np.zeros(Nbeads,float)

        #Classical Energies
        self.kin = 0
        self.pot = 0
        self.tot = 0        

        #Energy estimators
        self.E_kin = 0
        self.E_pot = 0
        self.E_tot = 0
        
        
        #Internal potential between beads
        self.int_pot = 0
        
        #Temperature
        self.Measured_Temperature = 0
        self.Target_Temperature = 0 
        
        #Positions and momenta of the centroids
        self.qc = np.zeros((1),float)
       #print('alv',np.shape(self.qc))


    #For simulation initialization only
    
    def set_bead_position(self, new_position, bead_number):
        self.q[bead_number] = new_position

    def set_bead_velocity(self, new_velocity, bead_number):
        self.p[bead_number] = new_velocity
        
    def set_centroid_q(self,new_qc):
        self.qc = new_qc
    
    #Setting
    def set_positions(self,new_positions):
        self.q = new_positions
        
    def set_velocities(self,new_velocities):
        self.p = new_velocities
    
    def set_qc(self,new_qc):
        self.qc = new_qc
    
    def set_forces(self,new_forces):
        self.f = new_forces
    
    def set_potential_energies(self,new_potential_energies):
        self.potential_energies = new_potential_energies
        
    def set_kinetic_energies(self,new_kinetic_energies):
        self.kinetic_energies = new_kinetic_energies
    
    def set_kin(self,new_kin):
        self.kin = new_kin
        
    def set_pot(self,new_pot):
        self.pot = new_pot
        
    def set_tot(self,new_tot):
        self.tot = new_tot
    
    def set_E_kin(self,new_E_kin):
        self.E_kin = new_E_kin
        
    def set_E_pot(self,new_E_pot):
        self.E_pot = new_E_pot
        
    def set_E_tot(self,new_E_tot):
        self.E_tot = new_E_tot
        
    def set_int_pot(self,new_int_pot):
        self.int_pot = new_int_pot
    
    def set_measured_temperature(self,new_Temperature):
        self.Measured_Temperature = new_Temperature

    def set_target_temperature(self,new_Target_Temperature):
        self.Target_Temperature = new_Target_Temperature

    def read_bead_position(self, positions_file, line_num=1):
        #print('Reading positions from: ', positions_file)
        line = linecache.getline(positions_file, line_num)
        value = float(line.strip().split()[0])  # nimmt nur den ersten Eintrag der Zeile
        return value

                
    def build_q(self, bead_pos_list, line_num=1):
        positions = np.empty(self.Nbeads, dtype=float)

        if len(bead_pos_list) == 1:
            # gleiche Position für alle Beads verwenden
            pos = self.read_bead_position(bead_pos_list[0], line_num)
            for i in range(self.Nbeads):
                positions[i] = pos
        else:
            # individuelle Datei für jeden Bead
            for i in range(self.Nbeads):
                positions[i] = self.read_bead_position(bead_pos_list[i], line_num)

        return positions

    
    def build_qc(self, bead_pos_list, line_num=1):
        if len(bead_pos_list) == 1:
            qc = float(self.read_bead_position(bead_pos_list[0], line_num))
        else:
            qcb = np.empty(self.Nbeads, dtype=float)
            for i in range(self.Nbeads):
                qcb[i] = float(self.read_bead_position(bead_pos_list[i], line_num))
            qc = np.mean(qcb)

        return qc
      

        
    #Getting    

    def get_positions(self):
        return self.q
    
    def get_velocities(self):
        return self.p

    def get_forces(self):
        return self.f
    
    def get_centroids(self):
        return self.qc
    
    def get_target_temperature(self):
        return self.Target_Temperature
    
    def get_measured_temperature(self):
        return self.Measured_Temperature