#!/bin/bash
#
# Usage: sub_benchmarks RUN_NAME CSV_FILE NSLOTS
#
#     RUN_NAME : the name of the job (REQUIRED)
#     CSV_FILE : the file name for the benchmark data (REQUIRED)
#     NSLOTS   : the number of processors (optional, default 20)
#

RUN_NAME=$1
CSV_FILE=$2

case $3 in
    ''|*[!0-9]*) NSLOTS=20 ;;
    *)           NSLOTS=$1 ;;
esac

# generate job script
cat << EOM >job
#!/bin/bash
#
# Use the ompi parallel environment, with X processors:
#$ -pe ompi $NSLOTS
#
# The name of the job:
#$ -N $RUN_NAME
#
# Start from the current working directory:
#$ -cwd
#
# Set the shell:
#$ -S /bin/bash
#
# Retain the existing environment (except PATH, see below):
#$ -V
#
# Exclude the interactive node
#$ -l h=!mdaoq.mdao0
#
# Output and error filenames (optional):
#$ -o \$JOB_NAME-\$JOB_ID.output
#$ -e \$JOB_NAME-\$JOB_ID.error
#
# Send mail at the end of the job (optional):
#$ -m e
#
# Address to send mail to (optional):
#$ -M stephen.w.ryan@nasa.gov

# USE_PROC_FILES causes I/O erros when using MPI.Spawn
unset USE_PROC_FILES

#module load openmpi/1.10.2/gnu/64bit

# workaround for dlopen issue
LD_PRELOAD=/hx/software/apps/openmpi/1.10.2/64bit/gnu/lib/libmpi.so


# check MPI/PETSc config
echo ---- PATHs ----
env | grep PATH
echo ---- PETSc ----
env | grep PETSC
echo ----  MPI  ----
which mpirun

mpirun -np 1 testflo -n 1 --mpispawn -bv -d $CSV_FILE
EOM

# submit job
qsub -sync y job
