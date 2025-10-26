"""Plot one example of each digit (0-9) from MNIST using matplotlib."""

import matplotlib.pyplot as plt

from syna.datasets.mnist import MNIST

dataset = MNIST(download=True, train=True, flatten=False)

# find first occurrence of each digit 0-9
found = {}
for img, label in dataset:
    lbl = int(label)
    if lbl not in found:
        found[lbl] = img
    if len(found) == 10:
        break

# prepare plotting grid 2x5
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()

for digit in range(10):
    ax = axes[digit]
    img = found.get(digit)
    if img is None:
        ax.text(0.5, 0.5, f"No {digit}", ha="center", va="center")
        ax.axis("off")
        continue

    ax.imshow(img[0], cmap="gray")
    ax.set_title(f"{digit}")
    # ax.axis("off")

plt.tight_layout()
plt.show()
