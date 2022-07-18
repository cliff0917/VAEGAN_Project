#!/bin/bash


# cvaegan
# python remake_att.py --model_type 'cvae' --dataset "SUN" \
# --E_path "methods/v2_cvaegan/checkpoints/SUN/2022-06-23_13-46-15/E.pth"

# vaegan
# python remake_att.py --model_type 'vae' --dataset "SUN" \
# --E_path "methods/v2_vaegan/checkpoints/SUN/2022-06-24_06-11-25/E.pth"

# wgan cvaegan
# python remake_att.py --model_type 'cvae' --dataset "SUN" \
# --E_path "methods/v3_wgan_cvaegan/checkpoints/SUN/2022-07-04_16-11-52/E.pth"

# wgan v4 cvaegan
# python remake_att.py --model_type 'cvae' --dataset "SUN" \
# --E_path "methods/v4_wgan_cvaegan/checkpoints/SUN/2022-07-05_00-44-15/E.pth"

# wgan v4 vaegan
python remake_att.py --model_type 'vae' --dataset "SUN" \
--E_path "methods/v4_wgan_vaegan/checkpoints/SUN/2022-07-05_01-27-02/E.pth"