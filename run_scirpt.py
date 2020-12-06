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

N_sys = 36
P = comm.Get_size()
rank = comm.rank

# # --- split the subsys ---
# n_sys = N_sys // P
# for ii in range(N_sys % P):
#     if rank == ii:
#         n_sys += 1

ranks = np.array(comm.allgather(rank))
ranks += 1
ranks = ranks[::-1]

n_sys = int(N_sys*(ranks[rank])/np.sum(ranks, dtype=int))


# remaining
rem_sys = N_sys - np.sum(comm.allgather(n_sys), dtype=int)
for ii in range(rem_sys):
    if rank == P - ii:
        n_sys += 1


print(rank, n_sys)

# --- get local subsys matrices ---
# determine the local size of matrix
systems = []
n_loc = 0
for ii in range(n_sys):
    ASmat = pickle.load(open("AS_mat_L1.p", "rb"))
    ASmat.setdiag(-2e0)

    systems.append(SubSys(ASmat))

    n_loc += ASmat.shape[0]
    # print(f"{rank}: adding {ASmat.shape}", n_loc)



# --- get RHS ---
b_vec = np.zeros(n_loc)
if rank == P - 1:
    np.random.seed(0)
    sys = systems[-1]
    sys.set_seeds(np.random.rand(sys.size))


# solve in reverse mode


output, times = parareal(systems, comm, P, 1e-5, 1e-9)

if rank == 0:

    print("============")
    print(  np.linalg.norm(systems[0].d_input))
    print("============")

# err = output - output[-1]

err = np.zeros(output.shape)
for ii, arr in enumerate(output):
    err[ii] = systems[0].get_err(arr)

err_norm = np.linalg.norm(err, axis=1)
rel_err_norm = err_norm/ err_norm[0]

rel_err_norms = comm.gather(rel_err_norm, root=0)
times_list = comm.gather(times, root=0)

if comm.rank == 0:
    print(times)
    # --- plot the relative errs ---
    for rel_err_norm, t  in zip(rel_err_norms, times_list):
        plt.semilogy(t, rel_err_norm, '-o')
        # plt.title(str(8-comm.rank))
    plt.xlabel('time [s]')
    plt.ylabel('error')
    plt.show()


# tot_start_time = time()
# seq_iter(systems, comm, P, 0, tol=1e-9, mode='fine')

# if rank == 0:

#     print("============")
#     print(  np.linalg.norm(systems[0].d_input_fine), time() - tot_start_time)
#     print("============")