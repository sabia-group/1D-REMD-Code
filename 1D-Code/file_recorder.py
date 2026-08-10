import numpy as np

class Recorder(object):
    #Constructor
    
    def __init__(self,system,name_file,index,rank,size):
        self.initialize(system,name_file,index,rank,size)
    def __del__(self):
        if hasattr(self, "T_file"):
            self.T_file.close()
    def initialize(self, system, name_file, index, rank, size):
        self.system = system
        self.rank = rank
        self.size = size
        self.Nbeads = system.Nbeads if system is not None else 0
        self.name = name_file

        if index % size == rank and system is not None:
            self.files = []
            prop_fname = f"{self.name}_properties_{self.Nbeads}_{index}.out"
            prop_file = open(prop_fname, 'w')
            prop_file.write(
                '{0:15s} {1:12s} {2:12s} {3:12s} {4:12s} {5:12s} {6:20s} {7:20s}\n'.format(
                    '#simulation time', 'qc', 'Total Energy',
                    'Kinetic_CV', 'Potential', 'Kinetic_MD',
                    'MeasuredTemperature', 'TargetTemperature'
                )
            )
            self.files.append(prop_file)
        else:
            self.files = []
                                                     
                                                                                          
                                                                                         
        
    def write(self, system, step, elapsed_time):
        if not self.files:
            return
        self.files[-1].write(
            '{0:15.15f} {1:17.17f} {2:17.17f} {3:17.17f} {4:17.17f} {5:17.17f} {6:17.17f} {7:17.17f}\n'.format(
                elapsed_time,
                float(system.qc),
                system.E_tot,
                system.E_kin,
                system.E_pot,
                system.kin,
                system.Measured_Temperature,    # richtiges Attribut
                system.Target_Temperature       # richtiges Attribut
            )
        )

    '''def write_replica_index_trajectory(self,step, time_step, rep_index, nrep):
        if not self.rank == 0:
            return

        if not hasattr(self, "R_file"):
            self.R_file = open("replica_index.out", "w")
            header = "#time " + " ".join([f"rep_T{i}" for i in range(nrep)]) + "\n"
            self.R_file.write(header)

        time = time_step * step
        rep_line = f"{time:.5f} " + " ".join([f"{rep_index[i]}" for i in range(nrep)]) + "\n"
        self.R_file.write(rep_line)'''

    def write_replica_index_trajectory(self, step, time_step, rep_index, nrep=None):
        if self.rank != 0:
            return

        rep_index = np.asarray(rep_index).ravel()   # <- garantiert 1D
        actual_nrep = rep_index.size

        if not hasattr(self, "R_file"):
            self.R_file = open("replica_index.out", "w", buffering=1)  # line-buffered
            header = "#time " + " ".join([f"rep_T{i}" for i in range(actual_nrep)]) + "\n"
            self.R_file.write(header)

        t = time_step * step
        rep_line = f"{t:.5f} " + " ".join(rep_index.astype(int).astype(str)) + "\n"
        self.R_file.write(rep_line)


              
             