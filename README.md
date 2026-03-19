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

