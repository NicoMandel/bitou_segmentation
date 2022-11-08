#!/bin/bash

# Run from parent directory by typing <bash scripts/k_means.sh>
# s=30
for K in 3 4 5 6
do 
    for s in 20 30
    do
        python scripts/train_kmeans.py -i data/bitou_test/ -o results/kmeans/classifiers/ -K $K -s $s --full --hsv 
    done
done