#!/usr/bin/env python3

import numpy as np
from simulation import Simulation
from system import System
from force_field import Force_Field
from thermostat import Thermostat
from modules import *
from mpi4py import MPI
from time import time, perf_counter
from file_recorder import Recorder
from modules import Internal_Potential
import random


start_time = time()
#set MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


GLOBAL_SEED = 1804

if __name__ == "__main__":
    temperatures,time_step,total_steps,Nbeads,mass,potential,T_type,tau,name,mode,nrep,exstride = read_params('My_PIMD.params')
    
    system=[]
    force_field=[]
    thermostat=[]
    rep_index = np.array([i for i in range(nrep)])
    simulation=[]
    simulation_time = []

    for i in range(nrep):
        np.random.seed(GLOBAL_SEED+i)
        #Set the system
        system.append(System(Nbeads))
        #Set the force field
        force_field.append(Force_Field(potential,mass))
        #Set Thermostat
        #print("DEBUG temp", temperature+(i*dtemp))
        thermostat.append(Thermostat(temperatures[i],tau,T_type, seed=GLOBAL_SEED + i))
        #set simulation 
        simulation.append(Simulation(system[i],temperatures[i],time_step,force_field[i],thermostat[i],mode,name,i,rank,size))
    if rank == 0:
        recorder_global = Recorder(None, "Ensemble_T", 0, rank, size)

    #Initialize simulation
    print('Initializing simulation: ')
    print('Nbeads: ',Nbeads)
    print('External potential: ',potential)
    print('Thermostat type: ',T_type)
    print('N replicas: ',nrep)
    print(temperatures)

    step=0
    global_attempts = 0
    global_accepted = 0 
    while step < total_steps:
        #excute only replicas that are assgigned to rank 
        for rep in range(rank, nrep, size):
            #print('DEBUG BEFORE step, rep, pot, epot, q, p, f', step, rep, system[rep].E_pot, system[rep].q, system[rep].p,
            #        system[rep].f)
            t0 = perf_counter()
            #repid = rep_index[rep]
            #simulation[rep].run_PIMD(exstride, repid, step)
            simulation[rep].run_PIMD(exstride, rep, step)
            t1 = perf_counter()
            simulation_time.append(t1-t0)

            #print('DEBUG AFTER step, rep, pot, epot, q, p, f', step, rep, system[rep].E_pot, system[rep].q, system[rep].p, system[rep].f)
        #wait until all simulation for given step is done
        comm.Barrier()
        #s
        local_data = []
        for rep in range(rank,nrep,size):
            local_data.append({
                'rep':   rep,
                'E_pot': system[rep].pot, 
                'q':     system[rep].q.copy(),
                'p':     system[rep].p.copy()*force_field[rep].mass
            })
        all_data = comm.gather(local_data, root=0)
        #e
        exchange_info = []
        if rank == 0:
            #S
            rep_index_before = rep_index.copy()

            total_attempts = 0
            total_accepted = 0
            flat = [entry for sublist in all_data for entry in sublist]
            flat.sort(key=lambda d: d['rep'])
            Epots = [d['E_pot'] for d in flat]
            qs    = [d['q']     for d in flat]
            ps    = [d['p']     for d in flat]
            #E
            progress = int(step / total_steps * 50)
            bar = "[" + "=" * progress + " " * (50 - progress) + "]"
            print(f"\r{bar} {step / total_steps:.1%}", end="", flush=True)
            #for offset in [0, 1]:
                #for i in range(offset, nrep - 1, 2):
                 #   j = i + 1
            offset = (step // exstride) % 2
            for i in range(offset, nrep-1, 2):
                j = i+1
                qi = qs[i]
                qj = qs[j]
                T_i = temperatures[i]
                T_j = temperatures[j]
                beta_i = 1.0 / (T_i)
                beta_j = 1.0 / (T_j)
                Vi, Vj = Epots[i], Epots[j] # --> old ext+spring
                m = force_field[0].mass
                #Si_Ti = Internal_Potential(qi,T_i,m)
                Si_Tj = Internal_Potential(qi,T_j,m)
                Sj_Ti = Internal_Potential(qj,T_i,m)
                _, _, Vext_i = force_field[i].eq_force_function(qi, Nbeads)
                _, _, Vext_j = force_field[j].eq_force_function(qj, Nbeads)

                #Vext_i *= 1/Nbeads
                #Vext_j *= 1/ Nbeads
                #Sj_Tj = Internal_Potential(qi,T_j,m)
                #Ui_i = Vi + Si_Ti         # U_Ti(qi)
                #Uj_j = Vj + Sj_Tj         # U_Tj(qj)
                Ui_j = Vext_j + Sj_Ti         # U_Ti(qj)
                Uj_i = Vext_i + Si_Tj         # U_Tj(qi)

                Delta = beta_i*Ui_j + beta_j*Uj_i - beta_i*Vi - beta_j*Vj
                #Delta = (beta_i-beta_j)*(Vj-Vi)

                total_attempts += 1
                randu = np.random.rand()
                arg = -Delta
                arg = min(0.0, arg)
                arg = np.clip(arg,-700.00, 0.0)
                expdelta = np.exp(arg)

                if randu < expdelta:
                    pi_new = ps[j] * np.sqrt(T_i / T_j)
                    pj_new = ps[i] * np.sqrt(T_j / T_i)
                    #pi_new = np.random.normal(0.0, np.sqrt(T_i/m), size=ps[i].shape)
                    #pj_new = np.random.normal(0.0, np.sqrt(T_j/m), size=ps[j].shape)

                    #change 11.02.26
                    exchange_info.append((i, j, qj, pi_new/force_field[i].mass, qi, pj_new/force_field[j].mass))
                    rep_index[i], rep_index[j] = (rep_index[j], rep_index[i])
                    total_accepted += 1
                else:
                    pass
                    #print(f"\n--- Swap between replicas {i} and {j} not successful ---")
            # EXAKT wie im ersten Code: schreibe rep_index_before für alle Substeps
            for s in range(step, step + exstride):
                if s < total_steps:
                    recorder_global.write_replica_index_trajectory(s, time_step, rep_index_before, nrep)

            global_attempts += total_attempts
            global_accepted += total_accepted

        exchange_info = comm.bcast(exchange_info if rank == 0 else None, root=0)
        rep_index = comm.bcast(rep_index if rank==0 else None, root=0)




        for i, j, qj, pi_new, qi, pj_new in exchange_info:
            if i % size == rank:
                simulation[i].reassign_values(system[i], qj, pi_new, temperatures[i])
            if j % size == rank:
                simulation[j].reassign_values(system[j], qi, pj_new, temperatures[j])
            #change 110226
        if rank == 0:
            print()
        step=step+exstride

if rank == 0:
    start_i = step - exstride if step >= exstride else 1  
    for i in range(start_i, step + 1):
        recorder_global.write_replica_index_trajectory(i, time_step, rep_index_before,nrep)

for rep in range(nrep):
    if rep % size == rank:
        simulation[rep].chi.check_sampling_coverage()
        np.savetxt(f"chi_convergence_rep{rep}_N{simulation[rep].system.Nbeads}.txt", simulation[rep].chi.get_values())
        #print(simulation[rep].chi.get_wasserstein_values())

if rank == 0:
    end_time = time()
    total_runtime = end_time - start_time
    print(np.mean(simulation_time))
    print(f"Gesamtlaufzeit: {total_runtime:.2f} Sekunden")



if rank == 0:
    if global_attempts > 0:
        global_acc_ratio = global_accepted / global_attempts
    else:
        global_acc_ratio = 0.0
    del recorder_global

    print(f"\n Globale Akzeptanzrate über alle Steps: {global_acc_ratio:.3f}")


