#!/bin/bash
#SBATCH --job-name=example_job
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00
#SBATCH --account=sadasivan
#SBATCH --partition=mi1004x 
# or use --partition=devel if required
 
 # Run your code. Example given below.
rocprof --hip-trace bin/example_tall_and_skinny_gemm_splitk_fp16 1 2 1 231
rocprof --list-basic
rocprof --list-derived
