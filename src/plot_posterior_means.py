import torch
import numpy as np
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt

from model import ImageDLVM

(x_train, y_train), (x_test, y_test) = load_data()

x_test = x_train + np.random.uniform(low=0, high=1, size=x_train.shape)
x_test = x_test / 256
x_test = x_test.reshape(-1, 28*28)

model = ImageDLVM(z_dim=2, x_dim=28*28, lr=1e-3)

model.load(save_dir="./saved_model")

posterior_means = model.encode(torch.from_numpy(x_test).float()).numpy()

plt.figure(figsize=(4, 4))
for g in np.unique(y_train):
    i = np.where(y_train == g)
    plt.scatter(posterior_means[:, 0][i], posterior_means[:, 1][i], label=g, s=0.5)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
lgnd = plt.legend()
for handle in lgnd.legendHandles:
    handle.set_sizes([20.0])
plt.axhline(0, linewidth=0.5, color='black', linestyle='--')
plt.axvline(0, linewidth=0.5, color='black', linestyle='--')
circle = plt.Circle((0, 0), radius=3, fill=False, linestyle='--', color='black', linewidth=0.5)
plt.gca().add_artist(circle)
plt.title("Means of (Approximate) Posteriors")
plt.xlabel("Arbitrary Axis 1")
plt.ylabel("Arbitrary Axis 2")
plt.savefig("posterior_means.png", bbox_inches='tight', dpi=300)
