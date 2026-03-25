import rasterio
import rasterio.features
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_opening
from skimage.filters import threshold_otsu
import os
import matplotlib as mpl
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# ── Cosmetic parameters ────────────────────────────────────────────────────────
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
font_size = 16
mpl.rcParams.update({'font.size': font_size, 'lines.linewidth': 1.5,
                     'axes.linewidth': 1., 'axes.labelsize': font_size+1,
                     'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
                     'legend.fontsize': 16})

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FOLDER_before = "Data/Larkana_large_2022_06_19"
DATA_FOLDER_after  = "Data/Larkana_large_2022_08_30"

# Optional: path to a reference flood shapefile (e.g. from UNOSAT or Copernicus EMS)
# Set to None to skip validation. 
# REFERENCE_SHAPEFILE = None  # e.g. "reference/UNOSAT_Pakistan_flood_2022.shp"
REFERENCE_SHAPEFILE = "EMSR629_AOI01_DEL_PRODUCT_r1_RTP01_v2_vector/EMSR629_AOI01_DEL_PRODUCT_observedEventA_r1_v2.shp"  # example shapefile path (not used in this code, but can be used for validation)

# ── Compute pixel-level metrics ───────────────────────────────────────
def compute_metrics(pred_mask, ref_mask):
    """
    Compute precision, recall, F1 score, and IoU between
    a predicted binary mask and a reference binary mask.
    Both inputs must be boolean numpy arrays of the same shape.
    """
    TP = np.sum(pred_mask & ref_mask)
    FP = np.sum(pred_mask & ~ref_mask)
    FN = np.sum(~pred_mask & ref_mask)
    TN = np.sum(~pred_mask & ~ref_mask)

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    iou       = TP / (TP + FP + FN + 1e-9)
    accuracy  = (TP + TN) / (TP + FP + FN + TN + 1e-9)

    return {"Precision": precision, "Recall": recall,
            "F1": f1, "IoU": iou, "Accuracy": accuracy,
            "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN)}


# ── Helper: compute flooded area from a mask ──────────────────────────────────
def compute_flooded_area_km2(mask, transform):
    """
    Given a binary flood mask and the raster affine transform,
    returns the flooded area in km².
    """
    pixel_size_x =  transform[0]   # metres per pixel in x
    pixel_size_y = -transform[4]   # metres per pixel in y (negative in geotiff convention)
    pixel_area_m2 = pixel_size_x * pixel_size_y
    return np.sum(mask) * pixel_area_m2 / 1e6


# ── 1. Load images ─────────────────────────────────────────────────────────────
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

with rasterio.open(vv_before_path) as src:
    vv_before_db = src.read(1).astype(float)
    profile = src.profile
    bounds  = src.bounds
    crs     = src.crs
    transform = src.transform

with rasterio.open(vh_before_path) as src:
    vh_before_db = src.read(1).astype(float)
with rasterio.open(vv_after_path) as src:
    vv_after_db  = src.read(1).astype(float)
with rasterio.open(vh_after_path) as src:
    vh_after_db  = src.read(1).astype(float)

print(f"CRS   : {crs}")
print(f"Bounds: {bounds}")
print(f"Shape : {vv_before_db.shape}")

# ── 2. dB → linear ────────────────────────────────────────────────────────────
vv_before = 10 ** (vv_before_db / 10)
vv_after  = 10 ** (vv_after_db  / 10)
vh_before = 10 ** (vh_before_db / 10)
vh_after  = 10 ** (vh_after_db  / 10)

# ── 3. Speckle noise reduction ────────────────────────────────────────────────
vv_before_f = median_filter(vv_before, size=3)
vv_after_f  = median_filter(vv_after,  size=3)
vh_before_f = median_filter(vh_before, size=3)
vh_after_f  = median_filter(vh_after,  size=3)

# ── 4. Change detection ratios ────────────────────────────────────────────────
vv_ratio = vv_after_f / (vv_before_f + 1e-6)
vh_ratio = vh_after_f / (vh_before_f + 1e-6)
feature  = vv_ratio * vh_ratio
feature  = (feature - np.nanmean(feature)) / np.nanstd(feature)   # normalise

# ── 5. Otsu thresholding ──────────────────────────────────────────────────────
threshold_vvvh = threshold_otsu(feature[np.isfinite(feature)])
threshold_vv   = threshold_otsu(vv_ratio[np.isfinite(vv_ratio)])
threshold_vh   = threshold_otsu(vh_ratio[np.isfinite(vh_ratio)])

flood_mask_vv   = vv_ratio < threshold_vv
flood_mask_vh   = vh_ratio < threshold_vh
flood_mask_vvvh = feature  < threshold_vvvh

# ── 6. Morphological cleaning ─────────────────────────────────────────────────
struct = np.ones((3, 3))
flood_vv   = binary_opening(flood_mask_vv,   structure=struct)
flood_vh   = binary_opening(flood_mask_vh,   structure=struct)
flood_vvvh = binary_opening(flood_mask_vvvh, structure=struct)

# ── 7. Quantitative area reporting ───────────────────────────────────────────
area_vv   = compute_flooded_area_km2(flood_vv,   transform)
area_vh   = compute_flooded_area_km2(flood_vh,   transform)
area_vvvh = compute_flooded_area_km2(flood_vvvh, transform)

print("\n── Flooded area estimates ──────────────────────────")
print(f"  VV-only   : {area_vv:.1f} km²")
print(f"  VH-only   : {area_vh:.1f} km²")
print(f"  VV×VH     : {area_vvvh:.1f} km²")

# ── 8. Optional validation against reference mask ────────────────────────────
if REFERENCE_SHAPEFILE is not None:
    print("\n── Validation against reference mask ───────────────")

    # Load reference shapefile and rasterise it to match our raster grid
    gdf = gpd.read_file(REFERENCE_SHAPEFILE).to_crs(crs.to_epsg())

    ref_mask = rasterio.features.rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=vv_before_db.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    ).astype(bool)

    for label, pred in [("VV-only", flood_vv),
                         ("VH-only", flood_vh),
                         ("VV×VH",   flood_vvvh)]:
        m = compute_metrics(pred, ref_mask)
        print(f"\n  {label}:")
        print(f"    Precision : {m['Precision']:.3f}")
        print(f"    Recall    : {m['Recall']:.3f}")
        print(f"    F1 score  : {m['F1']:.3f}")
        print(f"    IoU       : {m['IoU']:.3f}")
        print(f"    Accuracy  : {m['Accuracy']:.3f}")
        print(f"    TP={m['TP']:,}  FP={m['FP']:,}  FN={m['FN']:,}  TN={m['TN']:,}")

else:
    print("\n[INFO] No reference shapefile provided — skipping quantitative validation.")

# ── 9. Visualisations ─────────────────────────────────────────────────────────
# VV comparison
fig_vv, axes = plt.subplots(1, 3, figsize=(14, 6))
axes[0].imshow(vv_before_db, cmap="gray");  axes[0].set_title("VV Before (2022-06-19)"); axes[0].axis("off")
axes[1].imshow(vv_after_db,  cmap="gray");  axes[1].set_title("VV After (2022-08-22)");  axes[1].axis("off")
axes[2].imshow(flood_vv,     cmap="Blues"); axes[2].set_title(f"VV Flood Mask\n{area_vv:.1f} km²"); axes[2].axis("off")
plt.tight_layout()
fig_vv.savefig("flood_detection_vv.png", dpi=300)

# VH comparison
fig_vh, axes_vh = plt.subplots(1, 3, figsize=(14, 6))
axes_vh[0].imshow(vh_before_db, cmap="gray");  axes_vh[0].set_title("VH Before (2022-06-19)"); axes_vh[0].axis("off")
axes_vh[1].imshow(vh_after_db,  cmap="gray");  axes_vh[1].set_title("VH After (2022-08-22)");  axes_vh[1].axis("off")
axes_vh[2].imshow(flood_vh,     cmap="Blues"); axes_vh[2].set_title(f"VH Flood Mask\n{area_vh:.1f} km²"); axes_vh[2].axis("off")
plt.tight_layout()
fig_vh.savefig("flood_detection_vh.png", dpi=300)

# Mask comparison
fig_vvvh, axes_vvvh = plt.subplots(1, 3, figsize=(14, 6))
axes_vvvh[0].imshow(flood_vv,   cmap="Blues"); axes_vvvh[0].set_title(f"VV mask\n{area_vv:.1f} km²");   axes_vvvh[0].axis("off")
axes_vvvh[1].imshow(flood_vh,   cmap="Blues"); axes_vvvh[1].set_title(f"VH mask\n{area_vh:.1f} km²");   axes_vvvh[1].axis("off")
axes_vvvh[2].imshow(flood_vvvh, cmap="Blues"); axes_vvvh[2].set_title(f"VV×VH mask\n{area_vvvh:.1f} km²"); axes_vvvh[2].axis("off")
plt.tight_layout()
fig_vvvh.savefig("flood_detection_comparison.png", dpi=300)

# Overlay
fig_over, ax_over = plt.subplots(figsize=(8, 6))
ax_over.imshow(vv_after_db, cmap="gray")
ax_over.imshow(flood_vvvh, cmap="Blues", alpha=0.4)
ax_over.set_title(f"Flood Overlay on SAR Image\nDetected area: {area_vvvh:.1f} km²")
ax_over.axis("off")
fig_over.savefig("flood_overlay.png", dpi=300)

# ── 10. Save GeoTIFF ──────────────────────────────────────────────────────────
profile.update(dtype=rasterio.uint8, count=1)
with rasterio.open("flood_map.tiff", "w", **profile) as dst:
    dst.write(flood_vvvh.astype(rasterio.uint8), 1)
print("\nFlood map saved → flood_map.tiff")

plt.show()
