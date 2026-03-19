import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_opening
import os
import matplotlib as mpl

# Cosmetic parameters and plotting settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

font_size = 16
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams.update({'lines.linewidth': 1.5})
mpl.rcParams.update({'axes.linewidth': 1.})
mpl.rcParams.update({'axes.labelsize': font_size+1})
mpl.rcParams.update({'xtick.labelsize': font_size})
mpl.rcParams.update({'ytick.labelsize': font_size})
mpl.rcParams.update({'legend.fontsize': 16})


DATA_FOLDER_before = "Data/Larkana_2022_06_19"  # Adjust this path to your data folder
DATA_FOLDER_after = "Data/Larkana_2022_08_22"  # Adjust this path to your data folder

for file in os.listdir(DATA_FOLDER_before):                   # get the list of files in the data folder
    if "decibel_gamma0" in file and file.endswith(".tiff"):       # Band 4 = Red light (~665 nm). Plants absorb red light during photosynthesis.
        before_path = os.path.join(DATA_FOLDER_before, file)
for file in os.listdir(DATA_FOLDER_after):                   # get the list of files in the data folder
    if "decibel_gamma0" in file and file.endswith(".tiff"):       # Band 4 = Red light (~665 nm). Plants absorb red light during photosynthesis.
        after_path = os.path.join(DATA_FOLDER_after, file)

# 1. Load images
# BEFORE image
with rasterio.open(before_path) as src:
    before_db = src.read(1).astype(float)
    profile = src.profile

# AFTER image
with rasterio.open(after_path) as src:
    after_db = src.read(1).astype(float)
#--------------------------------------------------------------------------------------------------

# 2. Compute ratio of backscatter (after/before) in linear scale
# Convert dB → linear scale
before = 10 ** (before_db / 10)
after = 10 ** (after_db / 10)

# 3. Speckle noise reduction (median filter)
before_filtered = median_filter(before, size=3)
after_filtered = median_filter(after, size=3)

# 4. Compute ratio
ratio = after_filtered / (before_filtered + 1e-6)

#--------------------------------------------------------------------------------------------------


# 5. Determine pixels of interest using automatic thresholding (Otsu's method)
from skimage.filters import threshold_otsu

# Flatten valid values: filter out non-finite values (e.g., due to division by zero or masked areas)
valid_pixels = ratio[np.isfinite(ratio)]

# Compute threshold automatically using Otsu's method, which finds the threshold that minimizes intra-class variance in the histogram of pixel values
threshold = threshold_otsu(valid_pixels)
print("Auto threshold:", threshold)

flood_mask = ratio < threshold  # areas where backscatter decreased significantly (flooded areas) will have ratio < 1
# print(f"Flood pixels detected: {np.sum(flood_mask)}")


# 6. Clean noise (morphology)
flood_clean = binary_opening(flood_mask, structure=np.ones((3,3))) # The opening of an input image by a structuring element is 
                                                                   # the dilation of the erosion of the image by the structuring element

#--------------------------------------------------------------------------------------------------


# 7. Visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
axes[0].imshow(before_db, cmap="gray")
axes[0].set_title("Before Flooding (2022.06.19)")
axes[0].axis("off") 
axes[1].imshow(after_db, cmap="gray")
axes[1].set_title("After Flooding (2022.08.22)")
axes[1].axis("off")
axes[2].imshow(flood_clean, cmap="Blues")
axes[2].set_title("Detected Flood")
axes[2].axis("off")
plt.tight_layout()
# plt.show()

fig.savefig("flood_detection.png", dpi=300)


# 8. Save result
profile.update(dtype=rasterio.uint8, count=1)

with rasterio.open("flood_map.tiff", "w", **profile) as dst:
    dst.write(flood_clean.astype(rasterio.uint8), 1)

# print("Flood map saved as flood_map.tiff")


fig_over, ax_over = plt.subplots(figsize=(8, 6))

ax_over.imshow(after_db, cmap="gray")
ax_over.imshow(flood_clean, cmap="Blues", alpha=0.4)

ax_over.set_title("Flood Overlay on SAR Image")
ax_over.axis("off")
fig_over.savefig("flood_overlay.png", dpi=300)

#--------------------------------------------------------------------------------------------------


# 9. Compute flooded area (km²)
# Get pixel size (meters)
pixel_size_x = profile["transform"][0]
pixel_size_y = -profile["transform"][4]
pixel_area_m2 = pixel_size_x * pixel_size_y # area of one pixel in square meters (e.g., 10m x 10m = 100 m²)

# Count flooded pixels
flood_pixels = np.sum(flood_clean)

# Total flooded area
flood_area_m2 = flood_pixels * pixel_area_m2
flood_area_km2 = flood_area_m2 / 1e6

print(f"Flooded area: {flood_area_km2:.2f} km²")





plt.show()
