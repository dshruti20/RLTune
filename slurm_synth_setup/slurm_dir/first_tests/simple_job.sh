#!/bin/bash
#SBATCH --job-name=simpletest_node2
#SBATCH --output=node2.out
#SBATCH --nodelist=p100-20april-2
#SBATCH --time=00:01:00
#SBATCH --partition=debug

echo "Hello World"
