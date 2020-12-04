# --- Python 3.8 ---
'''
@File    :   accuracy_timing.py
@Time    :   2020/12/03
@Author  :   Josh Anibal
@Desc    :   creates a figure for the solve time for a given accuracy
'''


# === Standard Python modules ===
import pickle
from time import time
import warnings

# === External Python modules ===
import numpy as np
import scipy.sparse.linalg
from scipy.sparse import SparseEfficiencyWarning
import matplotlib.pyplot as plt
import matplotlib
from mpi4py import MPI

from sysclass import SubSys

import niceplots

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

tols = np.logspace(-1, -9, 9)
mat_sizes = np.zeros(5)
times = np.zeros((mat_sizes.size, tols.size))

for ii, lvl in enumerate(range(0,5)):
    ASmat = pickle.load(open(f"AS_mat_L{lvl}.p", "rb"))
    ASmat.setdiag(-2e-0)

    n =  ASmat.shape[0]
    mat_sizes[ii] = n
    print(f"solving {lvl} {ASmat.shape}")

    np.random.seed(0)
    RHS = np.random.rand(n)

    x = np.zeros(n)
    for jj, tol in enumerate(tols):
        srt_time = time()
        x, exit_code = scipy.sparse.linalg.gmres(ASmat, RHS, restart=200, tol=tol)
        times[ii, jj] = time()-srt_time
        print(tol, exit_code, time()-srt_time)

    plt.loglog(tols, times[ii], label=f'n={n}')

niceplots.all()
plt.xlabel('Solution tolerance')
plt.ylabel('Wall time [s]')
plt.show()
