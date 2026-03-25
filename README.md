# SAR Flood Detection using Sentinel-1 (VV & VH)

A Python-based project for detecting flood events using Synthetic Aperture Radar (SAR) satellite data from the Sentinel-1 mission by the European Space Agency.
This project implements a multi-polarization flood detection pipeline using VV and VH backscatter, combined with adaptive thresholding (Otsu method).

SAR satellites emit microwave signals and measure their reflection.
- Water surfaces is smooth, giving low backscatter (dark)
- Land surface is rough, giving high backscatter (bright)
Flooding causes a drop in backscatter intensity.

Floods significantly change the radar backscatter of the Earth’s surface. This project detects flooded areas by analyzing changes in SAR signals between two dates:
- Before the flood
- After the flood

Data is obtained from the [Copernicus Programme](https://dataspace.copernicus.eu).

The European Space Agency (ESA) mission Sentinel-1 provides multiple polarizations:
- VV (Vertical transmit / vertical receive), which is sensitive to surface roughness 
- VH (Vertical transmit / horizontal receive), which is sensitive to vegetation structure 

Floodings typically cause:
- Moderate drop in VV polarization
- Strong drop in VH polarization
This module analyses both polarization as well as their combining, which typically improves flood detection, especially in vegetated areas.

### Requirements
All you need is Python and the following libraries: numpy, scipy, matplotlib, and rasterio.
If you do not have installed them in your machine already, you can install them with:
```
pip install numpy rasterio matplotlib scipy scikit-image
```

### Key features:
- Multi-polarization SAR analysis (VV + VH)
- Radiometric processing (gamma0, dB → linear)
- Speckle noise reduction
- Ratio-based change detection
- Automatic thresholding using Otsu’s method
- Flood map generation (GeoTIFF)
- Flood extent estimation (km²)


## Example case study: the flooding of Indus river in Pakistan (August 2022)
Choose two dates of interest: 
- Before the flood (19 June 2022)
- After the flood (22 August 2022)

Select an Area of Interest (AOI) around the Indus river in Pakistan. For this example we choose an area near the city of Larkana. 
For each date, download the following products:
- VV – decibel gamma0
- VH – decibel gamma0

Compute:
- VV_ratio = VV_after / VV_before
- VH_ratio = VH_after / VH_before
- VVVH_ratio = VV_ratio × VH_ratio
Flooded areas have very low values of VV_ratio, VH_ratio, and VVVH_ratio.


Use Otsu’s method to automatically separate flood vs non-flood.
Otsu finds a threshold that maximizes separation between two classes (no manual tuning is required):
- low values ~ flood
- high values ~ non-flood

Here are some examples of images for our case study
![VV flood detection](https://github.com/lorenzogiombi-hub/Flooding_detection_SAR/blob/main/flood_detection_vv.png)
![VH flood detection](https://github.com/lorenzogiombi-hub/Flooding_detection_SAR/blob/main/flood_detection_vh.png)
![VVVH flood detection](https://github.com/lorenzogiombi-hub/Flooding_detection_SAR/blob/main/flood_detection_vvvh.png)


### Validation against a reference flood mask
Finally we validate the code against the official report of [Copernicus EMS](https://emergency.copernicus.eu) (Emergency Management Service). For the 2022 Pakistan floods the activation code is [EMSR629](https://mapping.emergency.copernicus.eu/activations/EMSR629/). The flood extent is contained in the file observed_event.shp

This Figure shows a comparison between the areo of interest analized by Copernicus (in color) and the area of interest used in this project (black and white)

Then we quantify the detected flooded areas in our code compared to the reference case. 
Comparing each flooded mask with the reference case, four possibilities arise 

|   | Reference predicts FLOOD | Reference predicts NO FLOOD |
|  ------------- | ------------- | ------------- |
| Code predicts FLOOD | TP          | FP         |
| Code pridicts NO FLOOD  | FN       | TN     |

where TP = True Positive, TN = True Negative, FP = False Positive, and FN = False Negative.
Then we measure 
- Precision = TP / (TP + FP) --> percentage (normalized to 1) of detected flood pixels that correspond to actual flood. 
- Recall = TP / (TP + FN) --> percentage of flooded pixel detected by this pipeline.
- F1 = 2 * precision * recall / (precision + recall) --> harmonic mean of Precision and Recall
- Intersection over Union = TP / (TP + FP + FN) -->  Area of overlap/area of union between the code mask and the reference mask. 
- Accuracy = (TP + TN) / (TP + FP + FN + TN) --> percentage of pixel classified correcly

The output of this example run is
```
── Flooded area estimates ──────────────────────────
  VV-only   : 382.5 km²
  VH-only   : 183.8 km²
  VV×VH     : 428.8 km²

── Validation against reference mask ───────────────

  VV-only:
    Precision : 0.827
    Recall    : 0.693
    F1 score  : 0.754
    IoU       : 0.605
    Accuracy  : 0.771
    TP=1,648,644  FP=344,484  FN=729,859  TN=1,969,513

  VH-only:
    Precision : 0.660
    Recall    : 0.266
    F1 score  : 0.379
    IoU       : 0.234
    Accuracy  : 0.558
    TP=631,669  FP=325,965  FN=1,746,834  TN=1,988,032

  VV×VH:
    Precision : 0.797
    Recall    : 0.749
    F1 score  : 0.772
    IoU       : 0.628
    Accuracy  : 0.776
    TP=1,780,397  FP=454,410  FN=598,106  TN=1,859,587
```
