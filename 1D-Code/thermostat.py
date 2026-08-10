#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:04:49 2022

@author: castrojo
"""

from modules import *

class Thermostat(object):
    def __init__(self, target_temperature, thermostat_strenght, thermostat_type=None, seed=420):
        self.temperature = target_temperature
        self.tau = thermostat_strenght
        self.seed = seed
        self.thermostat_type = thermostat_type
        self.rng = np.random.mtrand.RandomState(seed)

    def PILE_L(self, p_nm, time_step, mass):
        Nbeads = len(p_nm)
        B = 1/self.temperature
        Wp = Nbeads / B
        #TEST
        #Wk_vec = np.array([2 / B * np.sin(k * np.pi / Nbeads) for k in range(Nbeads)])
        Wk_vec = np.array([2 * Wp * np.sin(k * np.pi / Nbeads) for k in range(Nbeads)])
        r_k_vec = np.array([1/self.tau if k == 0 else 2 * Wk_vec[k] for k in range(Nbeads)])

        C1_vec = np.exp(-0.5 * time_step * r_k_vec)
        C2_vec = np.sqrt(1 - C1_vec**2)

        E_noise = self.rng.standard_normal(Nbeads)
        new_p_nm = C1_vec * p_nm + np.sqrt(Nbeads * mass / B) * C2_vec * E_noise
        #new_p_nm = C1_vec * p_nm + np.sqrt(mass / B) * C2_vec * E_noise
        #new_p_nm[0] = p_nm[0]

        return new_p_nm    

    def PILE_G(self, p_nm, time_step, mass, K):
        Nbeads = len(p_nm)
        B = 1/(Units(self.temperature,'KbT','hartree')) 
        KbT = 1 / B
        Wp = Nbeads / B

        Wk_vec = np.array([2 * Wp * np.sin(k * np.pi / Nbeads) for k in range(Nbeads)])
        r_k_vec = np.array([1 / self.tau if k == 0 else 2 * Wk_vec[k] for k in range(Nbeads)])

        C1_vec = np.exp(-0.5 * time_step * r_k_vec)
        C2_vec = np.sqrt(1 - C1_vec**2)

        new_p_nm = np.zeros(Nbeads)

        C = np.exp(-time_step * r_k_vec[0] * 0.5)
        r1 = self.rng.standard_normal()
        rg = 0  # optional: use chi2(1) if desired

        alpha2 = C + (Nbeads * ((1 - C) * (r1**2 + rg)) / (2 * K * B)) + \
                 2 * r1 * np.sqrt(Nbeads * C * (1 - C) / (2 * B * K))
        alpha = np.sqrt(alpha2)

        if r1 + np.sqrt(2 * K / (0.5 * B) * C / (1 - C)) < 0:
            alpha *= -1

        new_p_nm[0] = alpha * p_nm[0]

        # Remaining modes (k ≥ 1)
        E_noise = self.rng.standard_normal(Nbeads - 1)
        for k in range(1, Nbeads):
            new_p_nm[k] = C1_vec[k] * p_nm[k] + np.sqrt(Nbeads * mass / B) * C2_vec[k] * E_noise[k - 1]

        return new_p_nm

    def Andersen(self, p, time_step, mass):
        std = (self.temperature / mass)**0.5
        MB_velocities = np.random.normal(0.0, std, 10000) * mass

        p_Andersen = 1.0 - np.exp(-time_step/self.tau) # probability ∈ [0, 1]
        Nbeads = len(p)

        for i in range(Nbeads):
            if np.random.rand() < p_Andersen:
                p[i] = MB_velocities[np.random.randint(0, len(MB_velocities))]

        return p

    def Langevin(self, p, time_step, mass):
        Nbeads = len(p)
        r_k_vec = np.full(Nbeads, 1 / self.tau)
        B = 1/(Units(self.temperature,'KbT','hartree')) 

        C1_vec = np.exp(-0.5 * time_step * r_k_vec)
        C2_vec = np.sqrt(1 - C1_vec**2)

        E_noise = self.rng.standard_normal(Nbeads)
        new_p = C1_vec * p + np.sqrt(Nbeads * mass / B) * C2_vec * E_noise

        return new_p
    

    def set_T(self, new_temperature):
        self.temperature = new_temperature

    
    def G_noise(self, p):
        return self.rng.standard_normal(len(p))


    