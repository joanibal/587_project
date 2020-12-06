#!/bin/bash

# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=587_project
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1g
#SBATCH --time=00:30:00
#SBATCH --account=eecs587f20_class
#SBATCH --partition=standard
# Run your program
# (">" redirects the print output of your program,
#  in this case to "mpi_demo_output.txt")
evon py3
mpirun -np 1 python run_scirpt.py > run_1.out
mpirun -np 2 python run_scirpt.py > run_2.out
mpirun -np 4 python run_scirpt.py > run_4.out
mpirun -np 8 python run_scirpt.py > run_8.out
mpirun -np 18 python run_scirpt.py > run_18.out
mpirun -np 36 python run_scirpt.py > run_36.out



