# --- Python 3.8 ---
"""
@File    :   parareal.py
@Time    :   2020/11/28
@Author  :   Josh Anibal
@Desc    :   backward-propogation with parareal
"""


# === Standard Python modules ===
import pickle
from time import time
import warnings

# === External Python modules ===
import numpy as np
import scipy.sparse.linalg
from scipy.sparse import SparseEfficiencyWarning
import matplotlib.pyplot as plt
from mpi4py import MPI

from sysclass import SubSys


# solve in reverse mode
def solve_sys_fine(systems, comm, tol=1e-10):

    for ii in range(len(systems) - 1, -1, -1):
        srt_time = time()
        # systems[ii].d_input_fine, exit_code = systems[ii].solve_liner_rev(tol, systems[ii].d_input_fine)
        systems[ii].d_input_fine, exit_code = systems[ii].solve_liner_rev(tol, systems[ii].d_input)


        if ii > 0:
            systems[ii - 1].d_output = systems[ii].d_input_fine

        print(comm.rank,"fine",  ii, 'd_out', np.linalg.norm(systems[ii].d_output),'d_in',   np.linalg.norm(systems[ii].d_input_fine), tol, time()-srt_time)


def solve_sys_parareal_iter(systems, comm, tol=1e-3):

    for ii in range(len(systems) - 1, -1, -1):

        tmp, exit_code = systems[ii].solve_liner_rev(tol, np.zeros(systems[ii].size))
        srt_time = time()

        systems[ii].d_input_coarse_p1, exit_code = systems[ii].solve_liner_rev(tol, systems[ii].d_input_coarse)

        print("coarse1", comm.rank, ii, 'c_kp1', np.linalg.norm(tmp), 'c_k', np.linalg.norm(systems[ii].d_input_coarse), 'f_k', np.linalg.norm(systems[ii].d_input_fine), 'y_k', np.linalg.norm(systems[ii].d_input), tol, time()-srt_time)

        if ii > 0:
            systems[ii - 1].d_output = systems[ii].d_input_coarse_p1

    for ii in range(len(systems) - 1, -1, -1):
        systems[ii].d_input = systems[ii].d_input_coarse_p1 - systems[ii].d_input_coarse + systems[ii].d_input_fine

        systems[ii].d_input_coarse = systems[ii].d_input_coarse_p1

        if ii > 0:
            systems[ii - 1].d_output = systems[ii].d_input

        # print("coarse2", comm.rank, ii, 'c_kp1', np.linalg.norm(systems[ii].d_input_coarse_p1), 'c_k', np.linalg.norm(systems[ii].d_input_coarse), 'f_k', np.linalg.norm(systems[ii].d_input_fine), 'y_k', np.linalg.norm(systems[ii].d_input), tol)




def seq_iter(systems, comm, P, outter_iter, tol=1e-3, mode="coarse"):

    # for ii in range(P-outter_iter, -1, -1):
    # if comm.rank == ii:

    if comm.rank <= (P - outter_iter- 1):
        if comm.rank == (P - 1- outter_iter):
            pass
        else:
            # we need to know somthing about the sparsity structure of the global matrix
            # TODO generalize recv
            print("recv...")
            comm.Recv([systems[-1].d_output, MPI.DOUBLE], source=comm.rank + 1, tag=77)
            print(comm.rank, "recv!", np.linalg.norm(systems[-1].d_output))

        if mode == "coarse":
            solve_sys_parareal_iter(systems, comm,  tol=tol)
        elif mode == "fine":
            solve_sys_fine(systems, comm, tol)

        if comm.rank != 0:
            if mode == "coarse":
                comm.Send([systems[0].d_input, MPI.DOUBLE], dest=comm.rank - 1, tag=77)
                print(comm.rank, "coarse sent ", np.linalg.norm(systems[0].d_input), systems[0].d_input.shape)
            elif mode == "fine":
                comm.Send([systems[0].d_input_fine, MPI.DOUBLE], dest=comm.rank - 1, tag=77)
                print(comm.rank, "fine sent ", systems[0].d_input_fine.size, systems[0].d_input_fine.shape)
    else:
        print(comm.rank, "skipping!")

def parareal(systems, comm, P,  tol_cor, tol_fine):
    times = np.zeros(P+2)
    output = np.zeros((P+2, systems[0].d_input.size))

    tot_start_time = time()
    times[0] = time() - tot_start_time
    output[0] = systems[0].d_input

    # --- coarse sequential solve to init---
    print(comm.rank, "------- seq update --------", time() - tot_start_time)

    seq_iter(systems, comm, P, 0,  tol_cor)
    times[1] = time() - tot_start_time
    output[1] = systems[0].d_input

    for s in systems:
        s.d_input_fine = s.d_input

    # comm.barrier()


    for ii in range(P):

        # --- fine parallel solve ---
        print(comm.rank, "----------parallel --------", time() - tot_start_time)
        solve_sys_fine(systems, comm,  tol_fine)

        # comm.barrier()
        # --- corrected coarse sequential update ---
        print(comm.rank, "------- seq update --------", time() - tot_start_time)
        seq_iter(systems, comm, P, ii, tol_cor)

        # comm.barrier()
        times[ii+2] = time() - tot_start_time
        output[ii+2] = systems[0].d_input


    return output, times
