# --- Python 3.8 ---
'''
@File    :   run_scirpt.py
@Time    :   2020/12/03
@Author  :   Josh Anibal
@Desc    :   main run scripts, runs analysis and does timings
'''


# === Standard Python modules ===
import pickle
from time import time
import warnings
import argparse

# === External Python modules ===
import numpy as np
import scipy.sparse.linalg
from scipy.sparse import SparseEfficiencyWarning
import matplotlib.pyplot as plt
from mpi4py import MPI

from parareal import parareal, seq_iter
from sysclass import SubSys

# np.set_printoptions(threshold=np.inf)
warnings.simplefilter("ignore", SparseEfficiencyWarning)


# np.set_printoptions(threshold=np.inf)
warnings.simplefilter("ignore", SparseEfficiencyWarning)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N_sys = 4
P = comm.Get_size()
rank = comm.rank

# --- split the subsys ---
n_sys = N_sys // P
for ii in range(N_sys % P):
    if rank == ii:
        n_sys += 1


print(rank, n_sys)

# --- get local subsys matrices ---
# determine the local size of matrix
systems = []
n_loc = 0
for ii in range(n_sys):
    ASmat = pickle.load(open("AS_mat_L2.p", "rb"))
    ASmat.setdiag(-2e-0)

    systems.append(SubSys(ASmat))

    n_loc += ASmat.shape[0]
    print(f"{rank}: adding {ASmat.shape}", n_loc)



# --- get RHS ---
b_vec = np.zeros(n_loc)
if rank == P - 1:
    np.random.seed(0)
    sys = systems[-1]
    sys.set_seeds(np.random.rand(sys.size))


# solve in reverse mode


err, times = parareal(systems, comm, P, 1e-6, 1e-9)

# err = output - output[-1]
err_norm = np.linalg.norm(err, axis=1)
rel_err_norm = err_norm/ err_norm[0]

rel_err_norms = comm.gather(rel_err_norm, root=0)

if comm.rank == 0:
    print(err)
    print(rel_err_norms)
    # --- plot the relative errs ---
    for rel_err_norm in rel_err_norms:
        plt.semilogy(times, rel_err_norm, '-o')
        # plt.title(str(8-comm.rank))

    plt.show()

# tot_start_time = time()
# seq_iter(systems, comm, P,  tol=1e-9, mode='fine')
# if rank == 0:

#     print("============")
#     print(  np.linalg.norm(systems[0].d_input_fine), time() - tot_start_time)
#     print("============")
