#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate yu4
python 2102004_modelcheck.py --keyword=ml --phase=phase1
echo finish!
