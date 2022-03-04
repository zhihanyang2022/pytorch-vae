# pytorch-vae
Elegant VAE using torch.distributions, trained on MNIST.

By default, the code trains and visualizes a VAE with a latent dimension of 2.

## Visualizations

Generations | Posterior Means (Train Set)
:--------:|:------:|
<img src="https://user-images.githubusercontent.com/43589364/156720883-fc5d522d-2ab4-4ddd-af65-7c842221383a.png" width="400"> | <img src="src/posterior_means.png" width="400">

## How to run

```bash
cd src
```

```bash
python train_mnist.py
```

This script above logs to tensorboard, so you can use tensorboard to visualize training stats (ELBO, KL, reconstruction).

After this script finishes, the model will get saved to `src/saved_model`. 

```bash
python plot_posterior_means.py
```

This script above saves a png to `src`.
