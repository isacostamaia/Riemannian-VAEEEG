# Riemannian VAEEEG
## _Riemannian Variational Autoencoder for EEG decoding_

 - Riemannian VAE (M1) 
 - Stacking of Riemannian VAE (M1) + Supervised VAE (M2) as described in [Kingma et al., 2014](https://doi.org/10.48550/arXiv.1406.5298).
 - Stacking of Riemannian VAE (M1) + Supervised VAE (M2) as described in [Joy et al., 2020](https://doi.org/10.48550/arXiv.2006.10102).


## Installing Dependecies

```sh
conda env create -f environment.yml -n rveaeeeg
conda activate rveaeeeg
```

## Train and Test

 Riemannian VAE (M1) 

```sh
python train_scripts/run_train_m1.py
```


Stacking of Riemannian VAE (M1) + Supervised VAE (M2) as described in [Kingma et al., 2014](https://doi.org/10.48550/arXiv.1406.5298).

Train

```sh
python scripts/run_train_m1m2.py --config configs/m1m2_baseline_mi_zhou.yaml
```

Eval

```sh
python scripts/eval_m1m2.py  --config configs/m1m2_baseline.yaml   --exp_path analysis/experiences/EEG/[EXP_NAME]
```


Stacking of Riemannian VAE (M1) + Supervised VAE (M2) as described in [Joy et al., 2020](https://doi.org/10.48550/arXiv.2006.10102).

```sh
python train_scripts/run_train_m1m2_CCVAE.py
```

