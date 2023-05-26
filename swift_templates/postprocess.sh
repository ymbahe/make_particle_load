#!/bin/bash

# Script that is automatically called by SWIFT to post-process a snapshot

snap_basename=$1
snap_num=$2

token=`h5dump -a Header/SelectOutput ${snap_basename}_${snap_num}.hdf5 | grep "Snapshot"`

if [[ $token == "" ]]; then
    echo "Not a snapshot, skipping"
else

    cp run_vr_template.sh run_vr_$snap_num.sh
    sed -i "s/XXX-SNAP-STRING-XXX/${snap_num}/g" run_vr_$snap_num.sh
    sed -i "s/XXX-SNAP-BASENAME-XXX/${snap_basename}/g" run_vr_$snap_num.sh
    
    sbatch < run_vr_$snap_num.sh
fi
