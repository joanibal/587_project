# --- Python 3.8 ---
'''
@File    :   sysclass.py
@Time    :   2020/12/02
@Author  :   Josh Anibal
@Desc    :   holds the class definition for the subsystem problems
'''

# === Standard Python modules ===

# === External Python modules ===
import numpy as np
import scipy.sparse.linalg

# === Extension modules ===



class SubSys():

    def __init__(self,mat):

        self.mat = mat
        self.size = mat.shape[0]
        self.shape = mat.shape

        # --- derivative seeds ---
        self.d_output_seed = np.zeros(self.size)

        self.d_output = np.zeros(self.size)
        self.d_input = np.zeros(self.size)

        self.d_input_coarse = np.zeros(self.size)
        self.d_input_coarse_p1 = np.zeros(self.size)
        self.d_input_fine = np.zeros(self.size)


    def set_seeds(self, seed):
        self.d_output_seed = seed

    def solve_liner_rev(self, tol, x0):
        RHS = self.d_output + self.d_output_seed

        x, exit_code = scipy.sparse.linalg.gmres(self.mat, RHS, x0=x0,  restart=200, tol=tol)

        # self.d_input = x
        return x, exit_code


    def get_err(self):
        RHS = self.d_output + self.d_output_seed

        return self.mat.dot(self.d_input) - RHS


