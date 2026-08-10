#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:37:31 2022

@author: castrojo
"""
from os import write
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.integrate import simpson
from scipy.linalg import eigh
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
sns.set(style='whitegrid', font_scale=1.3)
matplotlib.use("MacOSX")
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13


def Units(value,from_,to_):
    
    if from_ == 'angstrom' and to_ == 'bohrradius':
        return(value*1.8897261)
    
    if from_ == 'angstrom/picosecond' and to_ == 'bohrradius/atomictime':
        return(value*4.5710289E-5)
    
    if from_ == 'angstrom/femtosecond' and to_ == 'bohrradius/atomictime':
        return(value*4.5710289E-3)
    
    if from_ == 'ev' and to_ == 'hartree':
        return(value*0.036749322)
    
    if from_ == 'picosecond' and to_ == 'atomictime':
        return(value*41341.373)
    
    if from_ == 'femtosecond' and to_ == 'atomictime':
        return(value*41.341473)
    
    if from_ == 'bohr' and to_ == 'angstrom':
        return(value*0.52917721)
    
    if from_ == 'bohrradius/atomictime' and to_ == 'angstrom/picosecond':
        return(value*21876.913)
    
    if from_ == 'bohrradius/atomictime' and to_ == 'angstrom/femtosecond':
        return(value*21.876913)
    
    if from_ == 'hartree' and to_ == 'ev':
        return(value*27.211386)
    
    if from_ == 'atomictime' and to_ == 'picosecond':
        return(value*2.4188843E-5)
    
    if from_ == 'KbT' and to_ == 'hartree':
        return(value*3.1668116E-6)
    
    if from_ == 'hartree' and to_ == 'KbT':
        return(value*315775.02)

    if from_ == 'kcal/mol' and to_ == 'hartree':
        return(value*0.001593601) 
    
    if from_ == 'hartree' and to_ == 'kcal/mol':
        return(value/0.001593601) 
    
    if from_ == to_:
        return(value)

def C_matrix(Nbeads):
    
    C = np.zeros((Nbeads,Nbeads),float)
    C[0,:] = np.sqrt(1)

    for j in range(Nbeads):
        for k in range(1,(Nbeads//2)+1):
            #print('i-',k)
            C[k][j] = np.sqrt(2.0)*np.cos((2*np.pi*j*k)/float(Nbeads))
        for k in range((Nbeads//2)+1,Nbeads):
            #print('i+',k)
            C[k][j] =  np.sqrt(2.0)*np.sin((2*np.pi*j*k)/float(Nbeads))
        if Nbeads%2 == 0:
            C[Nbeads//2][0:Nbeads:2] = 1.0
            C[Nbeads//2][1:Nbeads:2] = -1.0

    return C/np.sqrt(Nbeads)

    
def Normal_Propagator(k,mass,Nbeads,Temperature,dt):
    #B = 1/(Units(Temperature,'KbT','hartree')) 
    B = 1/Temperature
    Wp = Nbeads/(B)
    Wk = 2*Wp*np.sin(k*np.pi/Nbeads)
    #print('mmv',np.fft.fft([Wk]))
    wk0 = dt/mass

    if Wk == 0:
        NP_matrix = np.array([[np.cos(Wk*dt), -mass*Wk*np.sin(Wk*dt)],[np.round(wk0,15), np.cos(Wk*dt)]])
    else:
        NP_matrix = np.array([[np.cos(Wk*dt), -mass*Wk*np.sin(Wk*dt)],[(1/(mass*Wk))*np.sin(Wk*dt), np.cos(Wk*dt)]])
    return NP_matrix


def read_params(params_file):
    p = open(params_file,'r') 
    parameters = p.readlines()
    
    for line in parameters:
        sym_mode = re.match(r'mode = (.*)',line,re.M|re.I)
        if sym_mode:
            mode = sym_mode.group(1)
            break
        else:
            mode = 'in'
        
    for line in parameters:
        sym_temperature = re.match(r'temperature = (.*)',line,re.M|re.I)
        if sym_temperature:
            temperature = [float(t) for t in sym_temperature.group(1).split()]
            nrep = len(temperature)
            break
        else:
            temperature = [0.0009500]
            nrep = 1

    for line in parameters:
        sym_time_step = re.match(r'time_step = (.*)',line,re.M|re.I)
        if sym_time_step:
            time_step = float(sym_time_step.group(1))
            break
        else:
            time_step = 20.670687

    for line in parameters:
        sym_total_steps = re.match(r'total_steps = (.*)',line,re.M|re.I)
        if sym_total_steps:
            total_steps = int(sym_total_steps.group(1))
            break
        else:
            total_steps = 100000
    
    for line in parameters:
        sym_Nbeads = re.match(r'Nbeads = (.*)',line,re.M|re.I)
        if sym_Nbeads:
            Nbeads = int(sym_Nbeads.group(1))
            break
        else:
            Nbeads = 1
    
    for line in parameters:
        sym_mass = re.match(r'mass = (.*)',line,re.M|re.I)
        if sym_mass:
            mass = float(sym_mass.group(1))
            break
        else:
            mass = 1.0

    #for line in parameters:
    #    sym_nrep = re.match(r'nrep = (.*)',line,re.M|re.I)
    #    if sym_nrep:
    #        nrep = int(sym_nrep.group(1))
    #        break
    #    else:
    #        nrep = 1

    for line in parameters:
        sym_pot = re.match(r'potential = (.*)',line,re.M|re.I)
        if sym_pot:
            potential = str(sym_pot.group(1))
            break
        else:
            potential = 'harmonic'

    for line in parameters:
        sym_type = re.match(r'thermostat = (.*)',line,re.M|re.I)
        if sym_type:
            T_type = str(sym_type.group(1))
            break
        else: 
            T_type = 'PILE-L'

    for line in parameters:
        sym_tau = re.match(r'tau = (.*)',line,re.M|re.I)
        if sym_tau:
            tau = float(sym_tau.group(1))
            break
        else:
            tau = 800

    for line in parameters:
        sym_name = re.match(r'name_file = (.*)',line,re.M|re.I)
        if sym_name:
            name = sym_name.group(1)
            break
        else:
            name = 'My_Simulation'

    for line in parameters:
        sym_exstride = re.match(r'exstride = (.*)',line,re.M|re.I)
        if sym_exstride:
            exstride = int(sym_exstride.group(1))
            break  
        else:
            exstride = 100

    return temperature,time_step,total_steps,Nbeads,mass,potential,T_type,tau,name,mode,nrep,exstride

 
def Internal_Potential(q, temperature, mass):
    Nbeads = np.shape(q)[0]
    beta = 1 / temperature
    #beta = 1 / Units(temperature, 'KbT', 'hartree')
    Wp = Nbeads / beta

    internal_potential = 0.0
    for p in range(Nbeads):
        q_pp = q[p]
        q_p = q[p - 1] 
        internal_potential += 0.5 * mass * (Wp ** 2/Nbeads) * (q_pp - q_p) ** 2

    return internal_potential

            

def E_Estimators(temperature, forces, beads_positions, centroids_positions, potential_energies):
    Nbeads = len(beads_positions)

    #B = 1 / Units(temperature, 'KbT', 'hartree')
    B = 1/temperature
    qc = np.mean(beads_positions)
    FQ = np.sum((beads_positions - qc) * forces)
    E_kin = 0.5 / B - 0.5 / Nbeads * FQ
    E_pot = np.mean(potential_energies)

    E_tot = E_kin + E_pot

    return qc, E_tot, E_kin, E_pot

    
    
def read_my_pimd(file):
    E_TOT = []
    E_KIN = []
    E_POT = []
    time_e = []
    with open (file,'r') as e:
        next(e)
        next(e)
        for line in e:
            t = line.split()
            time_e.append(float(t[0]))
            E_TOT.append(float(t[1]))
            E_KIN.append(float(t[2]))
            E_POT.append(float(t[3]))

    print('My PE estimator: ',np.average(E_POT))

    return(time_e,E_TOT,E_KIN,E_POT)

def read_i_pi(file):
    E_TOT_i = []
    E_KIN_i = []
    E_POT_i = []  
    time_i = []      

    with open (file,'r') as p:
        next(p) 
        next(p)
        next(p)
        next(p)
        next(p)
        next(p)
        next(p)
        next(p)
        for line in p:
            m = line.split()
            time_i.append(float(m[1]))
            E_TOT_i.append(float(m[4]) + float(m[5]))
            E_KIN_i.append(float(m[6]))
            E_POT_i.append(float(m[5]))

    print('ipi PE estimator: ',np.average(E_POT_i))             
            
    return(time_i,E_TOT_i,E_KIN_i,E_POT_i)

    
def sign(value):
    if value > 0:
        return(float(1))
    elif value < 0:
        return(float(-1))
    

class ChiTracker: 
    '''
    Class to track the convergence of the chi estimator in a 1D REMD simulation.
    It computes the chi estimator based on the sampled distribution and the Boltzmann distribution.
    The Wasserstein distance is also computed to measure the convergence of the sampled distribution.
    '''
    def __init__(self, V, beta, xmin, xmax, nbins): 
        self.bin_edges = np.linspace(xmin, xmax, nbins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.dx = self.bin_edges[1] - self.bin_edges[0]
        self.nbins = nbins
        rho_B_unnorm = np.exp(-beta * V(self.bin_centers))
        #Z = np.sum(rho_B_unnorm) * self.dx    
        
        Z = simpson(rho_B_unnorm, self.bin_centers)
        self.rho_B = rho_B_unnorm / Z
        self.C0   = simpson(self.rho_B**2,    self.bin_centers)
        #self.C0 = np.trapezoid(self.rho_B**2, self.bin_centers)
        self.S1 = 0.0
        self.S2 = 0.0
        self.counts = np.zeros(nbins, dtype=int)
        self.t = 0
        self.values = []
        self.wasserstein_vals = []

    def wasserstein(self):
        hist = self.counts / (np.sum(self.counts) * self.dx)
        wd = wasserstein_distance(self.bin_centers, self.bin_centers, hist, self.rho_B)
        return wd

    def update(self, qc_scalar, rep_id=0):
        self.t += 1
        k = np.digitize(qc_scalar, self.bin_edges) - 1
        k = np.clip(k, 0, self.nbins - 1)
        c_old = self.counts[k]

        self.S1 += 2 * c_old + 1
        self.S2 += self.rho_B[k]
        self.counts[k] += 1

        chi_t = np.sqrt((self.S1 / (self.t**2 * self.dx)) - (2.0 / self.t) * self.S2 + self.C0)
        self.values.append(chi_t)

        if len(self.values) % 250000 == 0:
            step_str = f"{len(self.values):08d}step"
            filename = f"Dist/status_dist_{step_str}_rep{rep_id}"
            self.show_status(filename=filename)

    def show_status(self, filename="Dist/status_distribution"):
        plt.figure(figsize=(8, 4))
        hist = self.counts / (np.sum(self.counts) * self.dx)

        plt.plot(self.bin_centers, hist, label="Sampled Histogram")
        plt.plot(self.bin_centers, self.rho_B, label="Boltzmann-Distribution")
        plt.title(f"Step {len(self.values)}")
        plt.xlabel("Position in Bohr")
        plt.ylabel("Probability Density in a.u.")
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(filename + ".png")
        plt.close()

    def get_values(self):
        return np.array(self.values)
    
    def get_wasserstein_values(self):
        return np.array(self.wasserstein_vals)
    
    def check_sampling_coverage(self):
        visited_bins = np.count_nonzero(self.counts)
        coverage = visited_bins / self.nbins
        print(f"Sampling covers {coverage:.1%} of the bins.")

class ChiTrackerPIMD: 
    '''
    Class to track the convergence of the chi estimator in a 1D (RE)PIMD simulation.
    '''
    def __init__(self, V, beta, xmin, xmax, nbins, P, m, hbar=1.0): 
        self.bin_edges = np.linspace(xmin, xmax, nbins+1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.dx = self.bin_edges[1] - self.bin_edges[0]
        self.nbins = nbins

        self.rho_B = self.get_quantum_density(V, beta, m, hbar, n_states=100)
        
        ref_data = np.column_stack((self.bin_centers, self.rho_B))
        np.savetxt(f"Dist/boltzmann_ref_beta{beta:.4f}.dat", ref_data, 
                   header="Position_Bohr  Probability_Density_au")
        print(f"Referenzdaten gespeichert in: Dist/boltzmann_ref_beta{beta:.4f}.dat")

        self.C0    = simpson(self.rho_B**2, self.bin_centers)
        self.P = P
        self.S1 = 0.0
        self.S2 = 0.0
        self.counts = np.zeros(nbins, dtype=int)
        self.t = 0  
        self.values = []

    def update(self, bead_positions,rep_id):
        # bead_positions: Liste oder Array ALLER Bead-Positionen
        bead_positions = np.asarray(bead_positions)
        n_beads = bead_positions.size
        for q in bead_positions:
            k = np.digitize(q, self.bin_edges) - 1
            k = np.clip(k, 0, self.nbins - 1)
            c_old = self.counts[k]
            self.S1 += 2 * c_old + 1
            self.S2 += self.rho_B[k]
            self.counts[k] += 1

        self.t += n_beads  # Korrekt für Gesamtzahl!
        chi_t = np.sqrt((self.S1 / (self.t**2 * self.dx)) - (2.0 / self.t) * self.S2 + self.C0)
        self.values.append(chi_t)

        # Plot every n beads (optional)
        if len(self.values) % 250000 == 0:
            step_str = f"{len(self.values):08d}step"
            filename = f"Dist/status_dist_{step_str}_{self.P}bead_rep{rep_id}"
            self.show_status(filename=filename)
    '''def update(self, bead_positions, rep_id):
        bead_positions = np.asarray(bead_positions)
        
        # NEU: Berechne den Zentroid
        q_c = np.mean(bead_positions)

        # Sortiere q_c in Bin ein
        k = np.digitize(q_c, self.bin_edges) - 1
        k = np.clip(k, 0, self.nbins - 1)
        
        c_old = self.counts[k]
        self.S1 += 2 * c_old + 1
        self.S2 += self.rho_B[k]
        self.counts[k] += 1

        self.t += 1  # Nur 1 Zentroid pro update
        chi_t = np.sqrt((self.S1 / (self.t**2 * self.dx)) - (2.0 / self.t) * self.S2 + self.C0)
        self.values.append(chi_t)

        if len(self.values) % 250000 == 0:
            step_str = f"{len(self.values):08d}step"
            filename = f"status_dist_{step_str}_{self.P}bead_rep{rep_id}"
            self.show_status(filename=filename)'''



    def show_status(self, filename="Dist/status_distribution"):
        plt.figure(figsize=(8, 4))
        hist = self.counts / (np.sum(self.counts) * self.dx)
        print("Norm sampled hist:", simpson(hist, self.bin_centers))
        print("Norm Boltzmann   :", simpson(self.rho_B, self.bin_centers))
        print("Max hist:", np.max(hist))
        print("Max Boltzmann:", np.max(self.rho_B))

        plt.plot(self.bin_centers, hist, label="Sampled Histogram")
        plt.plot(self.bin_centers, self.rho_B, label="Boltzmann-Distribution")
        plt.title(f"Step {len(self.values)}")
        plt.xlabel("Position in Bohr")
        plt.ylabel("Probability Density in a.u.")
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(filename + ".png")
        plt.close()

        hist = self.counts / (np.sum(self.counts) * self.dx)
        current_data = np.column_stack((self.bin_centers, hist))
        np.savetxt(f"{filename}_data.txt", current_data)



    def get_values(self):
        return np.array(self.values)

    def check_sampling_coverage(self):
        visited_bins = np.count_nonzero(self.counts)
        coverage = visited_bins / self.nbins
        print(f"Sampling covers {coverage:.1%} of the bins.")

    def get_quantum_density(self, V, beta, m, hbar=1.0, n_states=1000):
        x = self.bin_centers
        dx = self.dx
        nbins = self.nbins

        Vx = V(x)
        T = self.sinc_dvr_kinetic(nbins, dx, m, hbar)  
        H = T + np.diag(Vx)

        E, psi = np.linalg.eigh(H)
        E = E[:n_states]
        psi = psi[:, :n_states]
        boltz_weights = np.exp(-beta * E)
        Z = np.sum(boltz_weights)
        rho = np.sum((np.abs(psi) ** 2) * boltz_weights, axis=1) / Z
        rho /= simpson(rho, x)  # <--- Normierung auf 1!
        return rho

    @staticmethod
    def sinc_dvr_kinetic(nbins, dx, m, hbar=1.0):
        T = np.zeros((nbins, nbins))
        prefactor = (np.pi * hbar)**2 / (2 * m * dx**2)
        for i in range(nbins):
            for j in range(nbins):
                if i == j:
                    T[i, j] = prefactor * (1.0 / 3.0)
                else:
                    T[i, j] = prefactor * ((-1) ** (i - j)) * (2.0 / (np.pi ** 2)) / ((i - j) ** 2)
        return T