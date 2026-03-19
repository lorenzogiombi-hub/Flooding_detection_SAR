import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_opening
import os


DATA_FOLDER_before = "Data/Larkana_2022_06_19"  # Adjust this path to your data folder
DATA_FOLDER_after = "Data/Larkana_2022_08_22"  # Adjust this path to your data folder

for file in os.listdir(DATA_FOLDER_before):                   # get the list of files in the data folder
    if "decibel_gamma0" in file and file.endswith(".tiff"):       # Band 4 = Red light (~665 nm). Plants absorb red light during photosynthesis.
        before_path = os.path.join(DATA_FOLDER_before, file)
for file in os.listdir(DATA_FOLDER_after):                   # get the list of files in the data folder
    if "decibel_gamma0" in file and file.endswith(".tiff"):       # Band 4 = Red light (~665 nm). Plants absorb red light during photosynthesis.
        after_path = os.path.join(DATA_FOLDER_after, file)


# 1. Load BEFORE image
with rasterio.open(before_path) as src:
    before_db = src.read(1).astype(float)
    profile = src.profile

# 1. Load AFTER image
with rasterio.open(after_path) as src:
    after_db = src.read(1).astype(float)


# 2. Convert dB → linear scale
before = 10 ** (before_db / 10)
after = 10 ** (after_db / 10)

# 3. Speckle noise reduction
before_filtered = median_filter(before, size=3)
after_filtered = median_filter(after, size=3)

# 4. Compute ratio
ratio = after_filtered / (before_filtered + 1e-6)


# 5. Detect flood (adaptive threshold)
threshold = 1.   # tune this threshold based on the ratio values (e.g., 0.8, 1.0, 1.2)
flood_mask = ratio < threshold  # areas where backscatter decreased significantly (flooded areas) will have ratio < 1
# print(f"Flood pixels detected: {np.sum(flood_mask)}")


# 6. Clean noise (morphology)
flood_clean = binary_opening(flood_mask, structure=np.ones((3,3))) # The opening of an input image by a structuring element is 
                                                                   # the dilation of the erosion of the image by the structuring element


# 7. Visualization
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(before_db, cmap="gray")
plt.title("Before (dB)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(after_db, cmap="gray")
plt.title("After (dB)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(flood_clean, cmap="Blues")
plt.title("Detected Flood")
plt.axis("off")

plt.show()


# 8. Save result
profile.update(dtype=rasterio.uint8, count=1)

with rasterio.open("flood_map.tiff", "w", **profile) as dst:
    dst.write(flood_clean.astype(rasterio.uint8), 1)

print("Flood map saved as flood_map.tiff")
