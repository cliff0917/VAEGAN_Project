

# Run experiment

## Setup

## ESZSL


```bash
./vaegan_v4/awa2.sh > ~/results/eszsl/v4_wgan_vaegan_awa

./vaegan_v4/cub.sh > ~/results/eszsl/v4_wgan_vaegan_cub

./vaegan_v4/sun.sh > ~/results/eszsl/v4_wgan_vaegan_sun
```

## CADA

setup environment
```bash
act cadavae
```

```bash
./vaegan_scripts/awa2.sh > ~/results/cada/v4_wgan_vaegan_awa

./vaegan_scripts/cub.sh > ~/results/cada/v4_wgan_vaegan_cub

./vaegan_scripts/sun.sh > ~/results/cada/v4_wgan_vaegan_sun
```

## tfvae

setup environment
```bash
act tfvaegan
```


```bash
./vaegan_scripts/awa2.py > ~/results/tfvae/v4_wgan_vaegan_awa

./vaegan_scripts/cub.py > ~/results/tfvae/v4_wgan_vaegan_cub

./vaegan_scripts/sun.py > ~/results/tfvae/v4_wgan_vaegan_sun
```