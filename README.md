

# Run experiment

## Setup

## Get Encoder

```
step 1: 先將 p-learning 產生的 .mat 放到 other_mats/$dataset/ 中
step 2: 在 ./scripts/origin_scripts/ 中新增 $dataset.sh(可參考其他 .sh 的設定)
step 3: 執行 ./scripts/origin_scripts/$datasets.sh
```

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
