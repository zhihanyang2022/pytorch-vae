import numpy as np
from keras.datasets.mnist import load_data
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import torchvision

from model import ImageDLVM

# load and preprocess dataset

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train + np.random.uniform(low=0, high=1, size=x_train.shape)
x_train = x_train / 256
x_train = x_train.reshape(-1, 28*28)

x_test = x_test + np.random.uniform(low=0, high=1, size=x_test.shape)
x_test = x_test / 256
x_test = x_test.reshape(-1, 28*28)

train_ds = TensorDataset(torch.from_numpy(x_train).float())
train_dl = DataLoader(train_ds, batch_size=32)

# training loop

model = ImageDLVM(z_dim=2, x_dim=28*28, lr=1e-3)

writer = SummaryWriter()

for epoch in range(10):

    elbos, kls, recs = [], [], []

    for xb in train_dl:
        stats_dict = model.fit(xb[0])
        elbos.append(stats_dict["elbo"])
        kls.append(stats_dict["kl"])
        recs.append(stats_dict["rec"])

    writer.add_scalar("Stat/elbo", np.mean(elbos), epoch)
    writer.add_scalar("Stat/kl", np.mean(kls), epoch)
    writer.add_scalar("Stat/rec", np.mean(recs), epoch)

    samples = model.sample(n=100).reshape(-1, 1, 28, 28)
    grid = torchvision.utils.make_grid(samples, nrow=10)
    writer.add_image('Viz/samples', grid, epoch)

model.save(save_dir="./saved_model")

# writer.add_embedding(tag='Viz/embeddings',
#                      mat=model.encode(torch.from_numpy(x_test[:1000]).float()).numpy(),
#                      label_img=torch.from_numpy(x_test[:1000].reshape(1000, 1, 28, 28)).float(),
#                      global_step=epoch)
