#!/bin/bash

python main.py --dataset "AWA2" --model_type "cycle" --lr 1e-4 \
--w_dir "./methods/cycle" --n_critic 2 --epochs 20
