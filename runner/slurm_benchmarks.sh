#!/bin/bash
#
# This script will run a batch job on the mdao cluster
# to perform benchmarks.
#
# Usage: $0 RUN_NAME
#
#     RUN_NAME : the name of the job (REQUIRED)
#

RUN_NAME=$1
CSV_FILE=$1.csv
CMD="testflo --pre_announce -bvs -d $CSV_FILE benchmark/benchmark_beam.py:BenchBeamNP2.benchmark_beam_np2"

#export MPIRUN_ARGS="-v --display-devel-map --display-allocation"
# --oversubscribe"


sbatch -W -N 1 -J $RUN_NAME <<EOF
#!/bin/bash
# Submit only to the mdao partition:
#SBATCH --partition=mdao
#
# Don't run on mdao0:
#SBATCH --exclude=mdao0
#
# Prevent other jobs from being scheduled on the allocated node(s):
#SBATCH --exclusive
#
# Set the mininum and maximum number of nodes:
#SBATCH --nodes=1-1
#
# CPU affinity:
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=1
#
# Output files:
#SBATCH --output=slurm-%x-%j.out.txt
#SBATCH --error=slurm-%x-%j.err.txt

export OMPI_MCA_mpi_warn_on_fork=0
ulimit -s 10240

# If the MPI library supports PMI2, the hostfile is not needed:
#srun -n 1 --mpi=pmi2 $CMD
$CMD
EOF
