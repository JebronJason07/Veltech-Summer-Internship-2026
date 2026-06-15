# exercise_4_edge_preserving.py

import cv2
import numpy as np
from matplotlib import pyplot as plt

def add_noise(image, noise_level=30):
    """Add noise to image"""
    noise = np.random.normal(0, noise_level, image.shape)
    return np.clip(image + noise, 0, 255).astype("uint8")


# Load image
image = cv2.imread("test_image.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Add noise
noisy = add_noise(gray, 25)

# Apply different edge-preserving methods
methods = {
    "Bilateral (mild)": lambda img: cv2.bilateralFilter(img, 9, 50, 50),
    "Bilateral (medium)": lambda img: cv2.bilateralFilter(img, 15, 75, 75),
    "Bilateral (strong)": lambda img: cv2.bilateralFilter(img, 21, 100, 100),
    "Gaussian": lambda img: cv2.GaussianBlur(img, (9, 9), 0),
    "Median": lambda img: cv2.medianBlur(img, 7),
}

# Convert to color for bilateral
noisy_color = add_noise(image, 25)

results = {"Noisy": noisy}
results_color = {"Noisy": noisy_color}

for name, method in methods.items():

    if "Bilateral" in name:
        results[name] = method(noisy_color)
        results[name] = cv2.cvtColor(results[name], cv2.COLOR_BGR2GRAY)

    else:
        results[name] = method(noisy)

# Create comparison grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

plot_idx = 0

for name, img in results.items():

    row = plot_idx // 3
    col = plot_idx % 3

    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(name)
    axes[row, col].axis('off')

    plot_idx += 1

# Add original for last slot
axes[1, 2].imshow(gray, cmap='gray')
axes[1, 2].set_title("Original (no noise)")
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig("edge_preserving_comparison.png")
plt.show()

print("Observation: Bilateral filter removes noise while keeping edges sharp!")
print(" - Compare Bilateral vs Gaussian on the edges")