#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:26:49 2022

@author: castrojo
"""
import numpy as np

def velocity_reader(name_file, Nbeads, Ndim):
    files = np.empty(Nbeads, dtype=object)
    for i in range(Nbeads):
        if Nbeads >= 10 and i <= 9:
            files[i] = name_file + '0' + str(i) + '.xyz'
        else:
            files[i] = name_file + str(i) + '.xyz'
    print(files)

    p = np.zeros((Nbeads, Ndim), float)
    for i in range(Nbeads):
        with open(files[i], 'r') as f:
            count = 1
            for line in f:
                if count == 3:
                    l = line.split()
                    vel = float(l[1])
                    p[i, :] = vel
                    break
                else:
                    count += 1

    return p

        