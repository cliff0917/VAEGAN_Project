#!/bin/bash

python main.py --dataset "SUN" --model_type "cvae" --lr 1e-4 \
--w_dir "./methods/method_vae" --n_critic 2 --epochs 400



