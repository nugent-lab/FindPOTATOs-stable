# FindPOTATOs (Beta Version)


## Summary
This software links together minor planet detections to form a tracklet. Tracklets are required so that minor planet observations can be submitted to the [Minor Planet Center](https://minorplanetcenter.net). 

This software is designed to be robust, and can accurately find near-Earth Objects (NEOs) as well as Trans-Neptunian Objects (TNOs). Written in Python, this software gets its incredible speed from the use of Ball Tree algorithms. These structures efficiently partition space and reduce searching time during various steps through the algorithm. 

The code assembles length-three tracklets from three candidate detection sets. In the code, they are labeled A, B, and C. However, the code is intended to be flexible and customizable, and could be adapted to a range of cadences. If you adapt this code, please cite this work (see Section "How To Cite", below.)

This builds upon substantial work by Nicole Tan (University of Canterbury), with further work by Prof. Carrie Nugent (Olin College). If you'd like more information on the original version of this code, see [N. Tan's Wellesley Honors Thesis](https://repository.wellesley.edu/object/ir1199).

This software is currently under development. Use this *beta version* at your own risk.

## Setup

1. Clone the repository.
2. (Recommended) Setup and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```
4. (Recommended) To be most effective, this code relies on Find_Orb to screen the found tracklets. To install Find_Orb on your machine, follow the instructions here: https://projectpluto.com/install.htm 

5. Modify settings, which are found at the beginning of findPotatos.py. This code has several options, choose what is right to you. (See Settings section, below.)

6. Prepare input. The code seeks three source detection files, in ``.csv`` format. Detection are each expected to have the following values:

| Name     | Value |
| -------- | ------- |
| `id `  | String. Unique detection identifier    |
| `RA` | Float. Right ascension of detection, degrees.     |
| `Dec`    | Float. Declination of detection, degrees.    |
|`mjd`| Float. Date and time of midpoint of observation, in Modified Julian Date format. This value should be the same for all detections in a single file|
| `observatory_code`| MPC-assigned observatory code where observations were taken. This value should also be constant for all detections in the file.|

# TBD more columns that are needed for ADES here.

It also seeks an `image_triplets.csv` file. Each row of this file should be the names of the `.csv` detection files that will be searched for a length three tracklet, seperated by commas. These detection files need to be of the same region of the sky. They do not need to be listed in this file in the order they will taken, FINDPOTATOs will sort that out for you based on the `mjd` values.

6. Run using
``` 
python3 findPOTATOS.py
```
By default, the code will produce observations in MPC 80-char format.

## Settings
This code was designed to link tracklets as part of Carrie Nugent's [NEAT Reprocessing](https://ui.adsabs.harvard.edu/abs/2022DPS....5450402N/abstract) work. The following settings may be useful.

`export_ades` If enabled, it will also export your results in ADES 2017 format. Also see our [Unofficial ADES repository](https://github.com/nugent-lab/unofficial_ADES) for stand-alone code to help with this task.

`print_thumbs` If enabled, this will print thumbnails of the sources in your resulting tracklets. It's always a good idea to check your sources by eye, to ensure you are submitting high-quality astrometry to the MPC. This does, however, slow down the code a bit. If you have independently validated your sources via another method, you can turn this off.

## Examples

Example 1: NEAT
can run with
show_sky_image = 'y' #query skyveiw for cutout of sky
export_ades = "y"
include_image_thumbs = 'y' 

python findPOTATOs.py NEAT


Example 2: CSS
Needs to run with
show_sky_image = 'n' #query skyveiw for cutout of sky
export_ades = "n"
include_image_thumbs = 'n' 

python findPOTATOs.py CSS

Example 3: ATLAS
- very close together need smaller
- can find asteroids that change in velocity across the plane of hte sky, so increase "timing uncertainty" to ~100 s to compensate. 

double linkages- ex: expanding timing_uncertainty = 5000, max_speed =  2 


## How to Cite

If you use this software, or adapt it for your own purposes, please cite the following papers:

1. Tan and Nugent et al., in prep, FINDPOTATOs: Open source asteroid linking software accelerated by binary trees

2. [N. Tan's Wellesley Honors Thesis](https://repository.wellesley.edu/object/ir1199).

3. If you use the Find_Orb-enabled screening, please cite [Bill Gray's work](https://projectpluto.com/find_orb.htm).

## License 

[![Build Status](https://img.shields.io/static/v1.svg?label=CSL&message=software%20against%20climate%20change&color=green?style=flat&logo=github)](https://img.shields.io/static/v1.svg?label=CSL&message=software%20against%20climate%20change&color=green?style=flat&logo=github)