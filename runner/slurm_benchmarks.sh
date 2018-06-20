#!/bin/bash

# This script submits a job via SLURM to perform benchmarks with testflo
#
# Usage: $0 RUN_NAME CSV_FILE
#
#     RUN_NAME : the name of the job (Default: YYMMDD_HHMM)
#     CSV_FILE : the file name for the benchmark data (Default: RUN_NAME.csv)
#

if [ -n "$1" ]; then
    RUN_NAME=$1;
else
    RUN_NAME=`date +%Y%m%d_%H%M%S`
fi

if [ -n "$2" ]; then
    CSV_FILE=$2;
else
    CSV_FILE=$RUN_NAME.csv
fi

# generate job script
cat << EOM >$RUN_NAME.sh
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
# Output files:
#SBATCH --output=slurm-%x-%j.out.txt
#SBATCH --error=slurm-%x-%j.err.txt

export OMPI_MCA_mpi_warn_on_fork=0
ulimit -s 10240

testflo --pre_announce -bvs -d $CSV_FILE
EOM

# submit the job
sbatch -W -J $RUN_NAME $RUN_NAME.sh

