#!/bin/bash

python main.py --dataset "SUN" --model_type "vae" --lr 1e-4 \
--w_dir "./methods/v2_vae" --n_critic 2 --epochs 400



