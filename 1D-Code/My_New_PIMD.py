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

start_time = time()

# Initialize MPI communicator, rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Global random seed for reproducibility
GLOBAL_SEED = 1804

if __name__ == "__main__":
    # Read simulation parameters from input file
    temperatures,time_step,total_steps,Nbeads,mass,potential,T_type,tau,name,mode,nrep,exstride = read_params('My_PIMD.params')

    # Initialize lists for systems, force fields, thermostats and simulations
    system=[]
    force_field=[]
    thermostat=[]
    # Replica index array: tracks which physical replica occupies which temperature slot
    rep_index = np.array([i for i in range(nrep)])
    simulation=[]
    simulation_time = []

    # Helper: compute centroid (mean position over all beads)
    def compute_centroid(q):
        return np.mean(q, axis=0)

    # Set up one system, force field, thermostat and simulation object per replica
    for i in range(nrep):
        np.random.seed(GLOBAL_SEED+i)
        # Set the system (initializes bead positions and momenta)
        system.append(System(Nbeads))
        # Set the force field (potential energy and forces)
        force_field.append(Force_Field(potential,mass))
        # Set the thermostat
        thermostat.append(Thermostat(temperatures[i],tau,T_type, seed=GLOBAL_SEED + i))
        # Set up the simulation object for this replica
        simulation.append(Simulation(system[i],temperatures[i],time_step,force_field[i],thermostat[i],mode,name,i,rank,size))

    # Root rank initializes the global replica index recorder
    if rank == 0:
        recorder_global = Recorder(None, "Ensemble_T", 0, rank, size)

    # Print simulation setup summary
    print('Initializing simulation: ')
    print('Nbeads: ',Nbeads)
    print('External potential: ',potential)
    print('Thermostat type: ',T_type)
    print('N replicas: ',nrep)
    print(temperatures)

    step=0

    # Main simulation loop
    while step < total_steps:
        # Each MPI rank runs the PIMD propagation for its assigned replicas
        for rep in range(rank, nrep, size):
            t0 = perf_counter()
            # Propagate replica for exstride steps (PIMD integrator + thermostat)
            simulation[rep].run_PIMD(exstride, rep, step)
            t1 = perf_counter()
            simulation_time.append(t1-t0)

        # Synchronize all MPI ranks before attempting replica exchange
        comm.Barrier()

        # Collect local replica data (potential energy, positions, momenta)
        local_data = []
        for rep in range(rank,nrep,size):
            local_data.append({
                'rep':   rep,
                'E_pot': system[rep].pot,
                'q':     system[rep].q.copy(),
                'p':     system[rep].p.copy()*force_field[rep].mass  # momenta --> p is actually only velocity 
            })

        # Gather all replica data on root rank
        all_data = comm.gather(local_data, root=0)

        exchange_info = []

        if rank == 0:
            # Save replica index state before exchange for trajectory recording
            rep_index_before = rep_index.copy()

            # Flatten and sort gathered data by replica index
            flat = [entry for sublist in all_data for entry in sublist]
            flat.sort(key=lambda d: d['rep'])
            Epots = [d['E_pot'] for d in flat]
            qs    = [d['q']     for d in flat]
            ps    = [d['p']     for d in flat]

            # Print progress bar
            progress = int(step / total_steps * 50)
            bar = "[" + "=" * progress + " " * (50 - progress) + "]"
            print(f"\r{bar} {step / total_steps:.1%}", end="", flush=True)

            # Even-odd nearest-neighbor exchange scheme:
            # alternates between even (0,1),(2,3),... and odd (1,2),(3,4),... pairs
            offset = (step // exstride) % 2
            for i in range(offset, nrep-1, 2):
                j = i+1

                m = force_field[0].mass
                T_i = temperatures[i]
                T_j = temperatures[j]
                # Inverse temperatures (in units where kB=1)
                beta_i = 1.0 / (T_i)
                beta_j = 1.0 / (T_j)

                # Kinetic energy helper (unused in acceptance criterion due to rescaling)
                def K(p, m):
                    return np.sum(p*p) / (2.0*m)

                Ki = K(ps[i], m)
                Kj = K(ps[j], m)

                qi = qs[i]
                qj = qs[j]

                # Compute centroids of both replicas
                ci = compute_centroid(qi)
                cj = compute_centroid(qj)

                # Rescale bead fluctuations around the centroid according to the new temperatures
                # This is the key step of frREPIMD: q' = q_c + sqrt(T_new/T_old) * (q - q_c)
                qj_new = cj + (T_j/T_i)**0.5 * (qj - cj)
                qi_new = ci + (T_i/T_j)**0.5 * (qi - ci)

                # Rescale momenta to match the new temperatures
                pi_new = ps[j] * np.sqrt(T_i / T_j)
                pj_new = ps[i] * np.sqrt(T_j / T_i)

                # Evaluate potential energies at original coordinates
                _, _, Vi = force_field[i].eq_force_function(qi,Nbeads)
                _, _, Vj = force_field[j].eq_force_function(qj,Nbeads)

                # Evaluate potential energies at rescaled coordinates
                _, _, Vi_new = force_field[i].eq_force_function(qj_new, Nbeads)
                _, _, Vj_new = force_field[j].eq_force_function(qi_new, Nbeads)

                # Verify spring term invariance under coordinate rescaling (debug check)
                #term_vorher = beta_i * Internal_Potential(qi, T_i, m) + beta_j * Internal_Potential(qj, T_j, m)
                #term_nachher = beta_i * Internal_Potential(qj_new, T_i, m) + beta_j * Internal_Potential(qi_new, T_j, m)

                # Acceptance criterion: only potential energy differences contribute
                # (spring terms cancel by construction of the coordinate rescaling)
                Delta = beta_i*(Vi_new - Vi) + beta_j*(Vj_new - Vj)

                # Metropolis acceptance: accept with probability min(1, exp(-Delta))
                arg = -Delta
                arg = min(0.0, arg)
                weight = np.exp(arg)
                randu = np.random.rand()

                if randu < min(1.0, weight):

                    # Store accepted exchange: new coordinates and rescaled momenta
                    exchange_info.append((i, j, qj_new, pi_new/force_field[i].mass, qi_new, pj_new/force_field[j].mass))
                    # Update replica index to track which replica is in which temperature slot
                    rep_index[i], rep_index[j] = (rep_index[j], rep_index[i])

        if rank == 0:
            # Write replica index trajectory for all steps in this stride window
            for s in range(step, step + exstride):
                if s < total_steps:
                    recorder_global.write_replica_index_trajectory(s, time_step, rep_index_before, nrep)

        # Broadcast accepted exchange info and updated replica index to all ranks
        exchange_info = comm.bcast(exchange_info if rank == 0 else None, root=0)
        rep_index = comm.bcast(rep_index if rank==0 else None, root=0)

        # Apply accepted exchanges: reassign coordinates and momenta on each rank
        for i, j, qj, pi_new, qi, pj_new in exchange_info:
            if i % size == rank:
                # Slot i receives rescaled coordinates from slot j
                simulation[i].reassign_values(system[i], qj, pi_new, temperatures[i])
            if j % size == rank:
                # Slot j receives rescaled coordinates from slot i
                simulation[j].reassign_values(system[j], qi, pj_new, temperatures[j])

        if rank == 0:
            print()

        step=step+exstride

# Write final replica index entries for the last stride window
if rank == 0:
    start_i = step - exstride if step >= exstride else 1
    for i in range(start_i, step + 1):
        recorder_global.write_replica_index_trajectory(i, time_step, rep_index_before,nrep)

# Save chi convergence metric for each replica
for rep in range(nrep):
    if rep % size == rank:
        simulation[rep].chi.check_sampling_coverage()
        np.savetxt(f"chi_convergence_rep{rep}_N{simulation[rep].system.Nbeads}.txt", simulation[rep].chi.get_values())

# Print simulation time statistics
if rank == 0:
    end_time = time()
    total_runtime = end_time - start_time
    print(np.mean(simulation_time))
    print(f"Gesamtlaufzeit: {total_runtime:.2f} Sekunden")
