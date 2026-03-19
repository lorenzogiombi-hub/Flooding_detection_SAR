# Flooded vegetation areas often show:
# VV drops but not as much as VH, because VV is more sensitive to surface roughness and can still reflect off the water surface, 
# while VH is more sensitive to volume scattering from vegetation and will drop significantly when vegetation is submerged.
# VH drops even more 
# So we can use the ratio of VV and VH (after/before) to identify flooded vegetation areas, 
# where the ratio will be significantly lower than 1 due to the drop in backscatter from both VV and VH, but especially from VH.


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

# get the list of files in the data folder
for file in os.listdir(DATA_FOLDER_before):                   
    if "VV_-_decibel_gamma0" in file and file.endswith(".tiff"):       
        vv_before_path = os.path.join(DATA_FOLDER_before, file)
    if "VH_-_decibel_gamma0" in file and file.endswith(".tiff"):       
        vh_before_path = os.path.join(DATA_FOLDER_before, file)
for file in os.listdir(DATA_FOLDER_after):                   
    if "VV_-_decibel_gamma0" in file and file.endswith(".tiff"):       
        vv_after_path = os.path.join(DATA_FOLDER_after, file)
    if "VH_-_decibel_gamma0" in file and file.endswith(".tiff"):       
        vh_after_path = os.path.join(DATA_FOLDER_after, file)

# 1. Load images
# BEFORE image
with rasterio.open(vv_before_path) as src:
    vv_before_db = src.read(1).astype(float)
    profile = src.profile
    bounds = src.bounds
    crs = src.crs
with rasterio.open(vh_before_path) as src:
    vh_before_db = src.read(1).astype(float)

# AFTER image
with rasterio.open(vv_after_path) as src:
    vv_after_db = src.read(1).astype(float)
with rasterio.open(vh_after_path) as src:
    vh_after_db = src.read(1).astype(float)

print("CRS:", crs)
print("Bounds:", bounds)
center_x = (bounds.left + bounds.right) / 2
center_y = (bounds.top + bounds.bottom) / 2

print("Center (UTM):", center_x, center_y)
#--------------------------------------------------------------------------------------------------

# 2. Compute ratio of backscatter (after/before) in linear scale
# Convert dB → linear scale
vv_before = 10 ** (vv_before_db / 10)
vv_after = 10 ** (vv_after_db / 10)
vh_before = 10 ** (vh_before_db / 10)
vh_after = 10 ** (vh_after_db / 10)

# 3. Speckle noise reduction (median filter)
vv_before_filtered = median_filter(vv_before, size=3)
vv_after_filtered = median_filter(vv_after, size=3)
vh_before_filtered = median_filter(vh_before, size=3)
vh_after_filtered = median_filter(vh_after, size=3)

# 4. Compute ratio
vv_ratio = vv_after_filtered / (vv_before_filtered + 1e-6)
vh_ratio = vh_after_filtered / (vh_before_filtered + 1e-6)
# Combine VV and VH
feature = vv_ratio * vh_ratio
feature = (feature - np.nanmean(feature)) / ( np.nanstd(feature))  # Normalize feature before Otsu's method to improve thresholding performance

#--------------------------------------------------------------------------------------------------


# 5. Determine pixels of interest using automatic thresholding (Otsu's method)
from skimage.filters import threshold_otsu

# Flatten valid values: filter out non-finite values (e.g., due to division by zero or masked areas)
# valid_pixels = vv_ratio[np.isfinite(vv_ratio)]
valid_pixels_vvvh = feature[np.isfinite(feature)]

# Compute threshold automatically using Otsu's method, which finds the threshold that minimizes intra-class variance in the histogram of pixel values
threshold_vvvh = threshold_otsu(valid_pixels_vvvh)
threshold_vv = threshold_otsu(vv_ratio[np.isfinite(vv_ratio)])
threshold_vh = threshold_otsu(vh_ratio[np.isfinite(vh_ratio)])
print("Auto threshold:", threshold_vvvh)

# flood_mask = vv_ratio < threshold  # areas where backscatter decreased significantly (flooded areas) will have ratio < 1
flood_mask_vvvh = feature < threshold_vvvh  # areas where backscatter decreased significantly (flooded areas) will have ratio < 1
flood_mask_vv = vv_ratio < threshold_vv
flood_mask_vh = vh_ratio < threshold_vh
# print(f"Flood pixels detected: {np.sum(flood_mask)}")


# 6. Clean noise (morphology)
flood_clean_vvvh = binary_opening(flood_mask_vvvh, structure=np.ones((3,3))) # The opening of an input image by a structuring element is 
                                                                   # the dilation of the erosion of the image by the structuring element
flood_clean_vv = binary_opening(flood_mask_vv, structure=np.ones((3,3)))
flood_clean_vh = binary_opening(flood_mask_vh, structure=np.ones((3,3)))
#--------------------------------------------------------------------------------------------------


# 7. Visualization
fig_vv, axes = plt.subplots(1, 3, figsize=(14, 6))
axes[0].imshow(vv_before_db, cmap="gray")
axes[0].set_title("VV Before Flooding (2022.06.19)")
axes[0].axis("off") 
axes[1].imshow(vv_after_db, cmap="gray")
axes[1].set_title("VV After Flooding (2022.08.22)")
axes[1].axis("off")
axes[2].imshow(flood_clean_vv, cmap="Blues")
axes[2].set_title("Detected Flood")
axes[2].axis("off")
plt.tight_layout()

fig_vh, axes_vh = plt.subplots(1, 3, figsize=(14, 6))
axes_vh[0].imshow(vh_before_db, cmap="gray")
axes_vh[0].set_title("VH Before Flooding (2022.06.19)")
axes_vh[0].axis("off") 
axes_vh[1].imshow(vh_after_db, cmap="gray")
axes_vh[1].set_title("VH After Flooding (2022.08.22)")
axes_vh[1].axis("off")
axes_vh[2].imshow(flood_clean_vh, cmap="Blues")
axes_vh[2].set_title("Detected Flood")
axes_vh[2].axis("off")
plt.tight_layout()


fig_vvvh, axes_vvvh = plt.subplots(1, 3, figsize=(14, 6))
axes_vvvh[0].imshow(flood_clean_vv, cmap="Blues")  # using VV-based flood mask for visualization since it is more sensitive to water surface and can better capture open water areas, while VH-based mask may be more fragmented due to vegetation effects.
axes_vvvh[0].set_title("VV filter Flood Mask")
axes_vvvh[0].axis("off") 
axes_vvvh[1].imshow(flood_clean_vh, cmap="Blues") # using VH-based flood mask for visualization since it is more sensitive to vegetation and can better capture flooded vegetation areas, while VV-based mask may miss some of these areas due to its sensitivity to surface roughness.
axes_vvvh[1].set_title("VH filter Flood Mask")
axes_vvvh[1].axis("off")
axes_vvvh[2].imshow(flood_clean_vvvh, cmap="Blues")  # using combined VV*VH-based flood mask for visualization since it can better capture both open water and flooded vegetation areas, while VV-based mask may miss some flooded vegetation areas and VH-based mask may be more fragmented due to vegetation effects.
axes_vvvh[2].set_title("VV*VH filter Flood Mask")
axes_vvvh[2].axis("off")
plt.tight_layout()

fig_vv.savefig("flood_detection.png", dpi=300)
fig_vh.savefig("flood_detection_vh.png", dpi=300)
fig_vvvh.savefig("flood_detection_vvvh.png", dpi=300)


# 8. Save result
profile.update(dtype=rasterio.uint8, count=1)

with rasterio.open("flood_map.tiff", "w", **profile) as dst:
    dst.write(flood_clean_vvvh.astype(rasterio.uint8), 1)

# print("Flood map saved as flood_map.tiff")


fig_over, ax_over = plt.subplots(figsize=(8, 6))

ax_over.imshow(vv_after_db, cmap="gray")
ax_over.imshow(flood_clean_vvvh, cmap="Blues", alpha=0.4)

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
flood_pixels = np.sum(flood_clean_vv)  # count the number of pixels classified as flooded (True values in the flood_clean_vv mask)

# Total flooded area
flood_area_m2 = flood_pixels * pixel_area_m2
flood_area_km2 = flood_area_m2 / 1e6

print(f"Flooded area: {flood_area_km2:.2f} km²")
#--------------------------------------------------------------------------------------------------


plt.show()
