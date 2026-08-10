# 1D-REMD-Code
Run REMD simulations with one particle in simple 1D model systems 

Code is based on previous work of Jorge Castro and Mariana Rossi. Some extensions regarding replica exchange has been made by Jan Mohr. 

Description

This GitHub repository contains a Python implementation of a 1-dimensional Path Integral Molecular Dynamics (PIMD) that allows you to simulate the quantum dynamics of a single particle with an arbitrary mass moving in a 1D potential. Further it is also possible to run mpi-based Replica Exchange MD and also a combination of Replica Exchange with PIMD. Most Notably it contains a version of fluctation-rescaling Replica Exchange, that rescales the bead fluctuations around the centroid such that the acceptance criterion is no longer dependent on the spring term.  

Simulation Parameter can be set in "My_PIMD.params". The initial position of the particle can be set in "my_simulation_pos.in". Excecution of "My_New_PIMD.py" via mpi or serially runs replica exchange including fluctuation rescaling. "My_PIMD.py" runs simple PIMD or Replica Exchange PIMD. All parameters are in atomic units. 

Contributing

We welcome contributions to improve and extend the functionality of the 1D PIMD code and post-processing tools. If you find any issues or have ideas for enhancements, feel free to open an issue or submit a pull request.
