#!/bin/bash

#SBATCH --ntasks 48 # The number of cores you need...
#SBATCH -J VR_${sim_name}_XXX-SNAP-STRING-XXX #Give it something meaningful.
#SBATCH --mem=${slurm_memory}
#SBATCH -o logs/vr.%J.out
#SBATCH -e logs/vr.%J.err
#SBATCH -p altair
#SBATCH --exclusive
#SBATCH -t $vr_time_string
#SBATCH --mail-type=ALL # notifications for job done & fail
#SBATCH --mail-user=bahe@strw.leidenuniv.nl #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

source ~/.module_load_swift_altair

# Run the program

./stf -C ./vrconfig.cfg \
      -i snapshots/XXX-SNAP-BASENAME-XXX_XXX-SNAP-STRING-XXX \
      -o vr/halos_XXX-SNAP-STRING-XXX \
      -I 2

# Strip trailing 0s from files if there is only one
if [ ! -f vr/halos_XXX-SNAP-STRING-XXX.properties.1 ]; then
    cd vr
    rename .0 "" *.0
    cd ..
fi
