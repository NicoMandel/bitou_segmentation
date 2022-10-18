#!/bin/bash

# Run from parent directory by typing <bash scripts/k_means.sh>
for K in 3 5 8 10
do 
    for s in 30 50 0
    do
        python scripts/k_means.py -i data/bitou_test/ -o results/kmeans/ -K $K -s $s --overlay True
    done
done