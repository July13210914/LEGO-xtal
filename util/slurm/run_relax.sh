#!/bin/sh -l
#SBATCH --partition=Apus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/dev/null
MODEL=$SLURM_JOB_NAME
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate legoxtal

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TBB_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OPENBLAS_MAIN_FREE=1

# Set RUN_RELAX to 'yes' by default
# Check if a '--no-relax' argument was provided to disable relax.py
RUN_RELAX="yes"
for arg in "$@"
do
  if [ "$arg" = "--no-relax" ]; then
    RUN_RELAX="no"
  fi
done

# Manually redirect stdout and stderr to log-${MODEL}.txt
# Check if the directory does not exist, and create it if needed
if [ ! -d "${MODEL}" ]; then
    mkdir ${MODEL}
fi
exec > ${MODEL}/log-${MODEL}.txt 2>&1

echo "Running on node: $(hostname)"
# Print active conda environment name
if [ -n "$CONDA_DEFAULT_ENV" ]; then
  echo "Conda env: $CONDA_DEFAULT_ENV"
elif [ -n "$CONDA_PREFIX" ]; then
  echo "Conda env: $(basename "$CONDA_PREFIX")"
else
  env_name=$(conda info --json 2>/dev/null | python -c "import sys,json; j=json.load(sys.stdin) if not sys.stdin.isatty() else {}; print(j.get('active_prefix_name',''))")
  if [ -n "$env_name" ]; then
    echo "Conda env: $env_name"
  else
    echo "Conda env: (none)"
  fi
fi

# Test Julia integration before proceeding
echo "Testing Julia integration..."
python -c "
try:
    from juliacall import Main as jl
    jl.seval('using PythonCall')
    print('✅ Julia integration working')
except Exception as e:
    print('❌ Julia integration failed:', e)
    exit(1)
" 

NCPU=$SLURM_CPUS_PER_TASK
NCPU1=$((NCPU/2))
NCPU2=$((NCPU/4))

# Conditionally run relax.py if RUN_RELAX is set to "yes"
if [ "$RUN_RELAX" = "yes" ]; then
  # Run the relaxation script with the specified number of CPUs
  python 2_relax.py --ncpu ${NCPU} --csv data/sample/${MODEL}.csv --end 100000
else
  echo "Skipping relax.py"
fi

# Stepwise relaxation by MACE 
START_TIME=$(date +%s)
python 3_energy.py --ncpu ${NCPU}  --step 250 --min 1   --max 100  --db ${MODEL}/final.db
python 3_energy.py --ncpu ${NCPU} --step 100 --min 100 --max 200  --db ${MODEL}/final.db
python 3_energy.py --ncpu ${NCPU} --step 100 --min 100 --max 200  --db ${MODEL}/final.db
python 3_energy.py --ncpu ${NCPU} --step 50  --min 200 --max 1000 --db ${MODEL}/final.db
python 3_energy.py --ncpu ${NCPU} --step 50  --min 200 --max 1000 --db ${MODEL}/final.db --metric
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "MACE Energy minimization script completed in $((ELAPSED_TIME / 60)) minutes."
