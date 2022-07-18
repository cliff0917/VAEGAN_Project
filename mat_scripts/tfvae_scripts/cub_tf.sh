#!/bin/bash


# cvaegan
# python remake_att.py --model_type 'cvae' --dataset "CUB" \
# --E_path "methods/v2_cvaegan/checkpoints/CUB/2022-06-23_13-46-06/E.pth"


# vaegan
# python remake_att.py --model_type 'vae' --dataset "CUB" \
# --E_path "methods/v2_vaegan/checkpoints/CUB/2022-06-24_06-10-48/E.pth"

# wgan cvaegan
# python remake_att.py --model_type 'cvae' --dataset "CUB" \
# --E_path "methods/v3_wgan_cvaegan/checkpoints/CUB/2022-07-04_16-11-31/E.pth"

# v4 wgan cvaegan
# python remake_att.py --model_type 'cvae' --dataset "CUB" \
# --E_path "methods/v4_wgan_cvaegan/checkpoints/CUB/2022-07-05_00-43-53/E.pth"


# v4 wgan vaegan
python remake_att.py --model_type 'vae' --dataset "CUB" \
--E_path "methods/v4_wgan_vaegan/checkpoints/CUB/2022-07-05_01-26-53/E.pth"