#!/bin/bash

# Run from parent directory by typing <bash scripts/k_means.sh>
s=30
for K in 3 4 5
do 
# for s in 30 50
# do
    python scripts/k_means.py -i data/bitou_test/ -o results/kmeans/hsv_scikit/ -K $K -s $s --overlay --hsv --full 
# done
done