#!/bin/bash
#
# Script to download the snapshots and a VELOCIraptor halo catalogue for a
# (very) low resolution dark-matter-only version of the EAGLE 100 Mpc box.
# These can be used as demonstration/test input for generating a zoom
# simulation mask.
#
# In total, this will download 326 MB of data.

wget https://home.strw.leidenuniv.nl/~bahe/zoom-setup-data/EAGLE_L100_3e10.tar.gz
tar -xzvf EAGLE_L100_3e10.tar.gz
