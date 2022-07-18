#!/bin/bash
# cvaegan
# python main.py --dataset "AWA2" --model_type "cvaegan" --lr 1e-4 \
# --w_dir "./methods/v4_wgan_cvaegan" --n_critic 2 --epochs 400

# vaegan
python main.py --dataset "AWA2" --model_type "vaegan" --lr 1e-4 \
--w_dir "./methods/v4_wgan_vaegan" --n_critic 2 --epochs 400