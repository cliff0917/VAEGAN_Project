#!/bin/bash


# cvaegan
# python remake_att.py --model_type 'cvae' --dataset "AWA2" \
# --E_path "methods/v2_cvaegan/checkpoints/AWA2/2022-06-23_13-45-58/E.pth"

# vaegan
# python remake_att.py --model_type 'vae' --dataset "AWA2" \
# --E_path "methods/v2_vaegan/checkpoints/AWA2/2022-06-24_06-10-05/E.pth"


# wgan cvaegan
# python remake_att.py --model_type 'cvae' --dataset "AWA2" \
# --E_path "methods/v3_wgan_cvaegan/checkpoints/AWA2/2022-07-04_16-10-36/E.pth"


# v4 wgan cvaegan
# python remake_att.py --model_type 'cvae' --dataset "AWA2" \
# --E_path "methods/v4_wgan_cvaegan/checkpoints/AWA2/2022-07-05_00-43-46/E.pth"

# v4 wgan vaegan
python remake_att.py --model_type 'vae' --dataset "AWA2" \
--E_path "methods/v4_wgan_vaegan/checkpoints/AWA2/2022-07-05_01-26-42/E.pth"