#!/bin/bash

python main.py --dataset "AWA2" --model_type "cvae" --lr 1e-4 \
--w_dir "./methods/v2_dcvae" --n_critic 2 --epochs 400
