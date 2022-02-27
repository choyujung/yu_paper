#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate yu4
python stock_1125_version4_nokeyword.py --thema=AI --target=Close --phase=phase3
echo finish!

