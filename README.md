# pytorch-vae
Elegant VAE using torch.distributions, trained on MNIST.

## Visualizations

Generations | Posterior Means (Train Set)
:--------:|:------:|
<img src="https://user-images.githubusercontent.com/43589364/156720883-fc5d522d-2ab4-4ddd-af65-7c842221383a.png" width="300"> | <img src="src/posterior_means_2d.png" width="300">

## How to run

Change working directory:

```bash
cd src
```

Run training script:

```bash
python train_mnist.py
```

By default, the script above trains a VAE with a latent dimension of 2 for 10 epochs.

This script above logs to tensorboard (i.e., as `.local` in `src/runs`), so you can use tensorboard to visualize training stats (ELBO, KL, reconstruction) and generations on a per-epoch basis.

After this script finishes, the model will get saved to `src/saved_model`. 

This script below saves a png to `src`:

```bash
python plot_posterior_means.py
```

## Dependencies

Please adapt `tensorflow-macos` to your own device.

```
pip install numpy==1.22.2 torch==1.10.2 torchvision==0.11.3 tensorflow-macos==2.8.0 keras==2.8.0 matplotlib==3.5.1
```
