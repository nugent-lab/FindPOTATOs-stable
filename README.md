# FindPOTATOs

If you use this software, please [cite](#how-to-cite) our paper and [let us know](www.crnugent.com/contact)! Knowing how people use this software will help us secure funding for improvements.

## Summary
This software links together minor planet detections to form a tracklet. Tracklets are required so that minor planet observations can be submitted to the [Minor Planet Center](https://minorplanetcenter.net). 

This software is designed to be robust and can accurately find near-Earth Objects (NEOs) as well as Trans-Neptunian Objects (TNOs). Written in Python, this software gets its speed from the use of Ball Tree algorithms. These structures efficiently partition data and reduce searching time during various steps through the algorithm. 

The code assembles tracklets from candidate detection sets. 

We encourage users to make modifications to fit their particular needs. If you adapt this code, please cite this work. [How to Cite.](#how-to-cite) We welcome pull requests.

## Authors
This builds upon Nicole Tan's (University of Canterbury) [Wellesley Honors Thesis](https://repository.wellesley.edu/object/ir1199). Further development and validation was done by [Carrie Nugent](https://crnugent.com) (Olin College). Valuable feedback was provided by [James "Gerbs" Bauer](https://www.astro.umd.edu/people/gerbsb.html) (University of Maryland). 

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
4. (Optional) To be most effective, this code relies on Find_Orb to screen the found tracklets. To install Find_Orb on your machine, follow the instructions here: https://projectpluto.com/install.htm 

5. Modify settings, which are found at the beginning of `findPOTATOs.py`. This code has several options, choose what is right to you. (See [Examples](#examples).)

6. Prepare input (or, skip to [Examples](#examples)). The code seeks three source detection files, in ``.csv`` format. Detection are each expected to have the following values:

| Name 	   | Value   | Required? |
| -------- | ------- | ------- |
| `RA`   | Float. Right ascension of detection, degrees. 	        | yes |
| `Dec`	| Float. Declination of detection, degrees.	                | yes |
| `magnitude`| Float. Brightness of detection, in magnitudes.       | yes |
|`mjd`| Float. Date and time of midpoint of observation, in Modified Julian Date format. This value should be the same for all detections in a single file                     | yes |
| `observatory_code`| String. MPC-assigned observatory code where observations were taken. This value should also be constant for all detections in the file.                            | yes |
|`band`| String. MPC-defined observing band (wavelength) of detections      | yes |
|`mag_err`| Float. Uncertainty on magnitude, as defined by ADES export format. | When using ADES export format.|
|`RA_err`| Float. Uncertainty on Right Ascension, as defined by ADES export format. | When using ADES export format.|
|`Dec_err`| Float. Uncertainty on Declination, as defined by ADES export format. |When using ADES export format.|
|`ml_probs`| Float. The probability that the detection is true; assigned by a machine learning algorithm.| No |


The code seeks an `image_groups_[label].csv` file. Each row of this file should be the names of the `.csv` detection files that will be searched for tracklets, seperated by commas. These detection files need to be of the same region of the sky. They do not need to be listed in this file in the order they will taken, FINDPOTATOs will sort that out for you based on the `mjd` values.

6. Run using
``` 
python3 findPOTATOS.py [fileID]
```
`fileID` is a string that identifies the input files, see below for examples. By default, the code will produce observations in MPC 80-char format. 

7. (Optional) Remove tracklets with duplicate detections. The MPC will not accept a submission if multiple tracklets share a single detection. You can identify these and weed them out using `remove_duplicates.py` and the following call:

``` 
python3 remove_duplicates.py [fileID]
```

## Examples
We have provided examples in `sample_source_files/`. We encourage users to experiment with these examples to increase their understanding of how the code works and how to change the parameters for their particular dataset. Installation of Find_Orb is not needed for these examples.

The uncertanties associated with brigtness and position in this example data are placeholder values, and are not actual values associated with measurements by sky surveys.

### Example 1: CSS
This example uses data from the [Catalina Sky Survey](https://catalina.lpl.arizona.edu/). 
First copy the example parameters file to the working directory.
```
cp sample_source_files/parameters_CSS.py parameters.py
``` 
Then run 
``` 
python findPOTATOs.py CSS
``` 
All tracklets should be returned, as can be verified by the `sources_and_tracklets_0.png` figure. Inspect the output in the `output/` directory. Individual tracklet information is displayed in tracklet figure files in the `output/` directory as well. For real data, these figures should be inspected to ensure that the tracklet appears as you expect, and that your tracklet does not overlap with stationary sources as displayed in the SDSS thumbnail. (Note that SDSS thumbnails are not available for all regions of the sky.)

### Example 2: ATLAS
This example uses data adapted from real observations submitted by the [Asteroid Terrestrial-impact Last Alert System (ATLAS)](https://atlas.fallingstar.com/). ATLAS has it's own linking software, `PUMALINK`, as described in the excellent paper "Linking Sky-plane Observations of Moving Objects." ([Tonry, 2023](https://arxiv.org/abs/2309.15344)). `PUMALINK` is the best linking software to use on this dataset, however, we include ATLAS data in our examples as an instructive illustration of how to use FindPOTATOs. 

ATLAS targets close, fast-moving near-Earth objects. Because of this, FindPOTATOs requires different parameters to find objects than in the Catalina Sky Survey case. Whereas Catalina objects move in generally straight lines with minimal variation in velocity and brightness between detections, the ATLAS objects have apparent velocity changes over a tracklet as observed from Earth, as well as brightness variations. 

These changes mean that the search parameters in FindPOTATOs must be broader than needed for Catalina, and are an example of the types of parameter changes needed to fit different observing cadences and minor planet subgroups (such as very close approaching NEOs vs. NEOs generally vs main-belt objects vs trans-Neptunian objects.)

First copy the example parameters file to the working directory.

```
cp sample_source_files/parameters_ATLAS.py parameters.py
``` 
Then run 
``` 
python findPOTATOs.py ATLAS
``` 
All tracklets should be returned, as can be verified by the `sources_and_tracklets_0.png` figure.

### Example 3: NEAT
The previous examples only included tracklets and no extraneous noise. To see how parameters can be used to weed out spurious linkages between noise points, we include examples derived from Near-Earth Asteroid Tracking ([NEAT](https://en.wikipedia.org/wiki/Near-Earth_Asteroid_Tracking)) data. This data was reprocessed as part of research work led by C. Nugent.

This example works well with the same settings as Catalina Sky Survey.
Copy the example parameters file to the working directory.
```
cp sample_source_files/parameters_CSS.py parameters.py
``` 
Then run 
``` 
python findPOTATOs.py NEAT
``` 

### Example 4: TNOs
This example uses ephemeridies generated by [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) for a set of ten trans-Neptunian objects (TNOs); as viewed from Pan-STARRS 1 (Haleakala). The observing cadence mimics the real observed cadence of TNOs (for example, see [these published observations](https://minorplanetcenter.net/db_search/show_object?object_id=533676)). The CSS parameters retreive all these sources; however, tighter parameters could be used to rule out noise sources for a TNO-specific dataset. 

Copy the example parameters file to the working directory.
```
cp sample_source_files/parameters_CSS.py parameters.py
``` 
Then run 
``` 
python findPOTATOs.py TNO
``` 

### Example 5: Southern objects
This example uses ephemeridies generated by [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) for a set of three objects at <68 degrees declination as viewed from  Mount John Observatory, Lake Tekapo at one hour intervals. This demonstrates FindPOTATO's efficiency at high declinations. The CSS parameters retreive all these sources; however, tighter parameters could be used to rule out noise sources for a TNO-specific dataset. 

Copy the example parameters file to the working directory.
```
cp sample_source_files/parameters_CSS.py parameters.py
``` 
Then run 
``` 
python findPOTATOs.py south
``` 


## Caveats 
Please keep in mind the following caveats.
1. Ensure that your data is clean, and does not include stationary sources. The SDSS sky image is provided to help ensure stationary sources are not present in the tracklet; however, the SDSS images are not available for all regions of the sky. It is the user's responsibility to ensure data fidelity. [Aladin Lite](http://aladin.cds.unistra.fr/AladinLite/) is a good resource for ruling out stationary sources.
2. FindPOTATOs may return multiple tracklets that contain the same detection. Before submission of tracklets to the MPC, please check to ensure that each tracklet contains unique detections.
3. While gaining familiarity with the code, it is suggested that you check your tracklets against the MPC's [Minor Planet Checker](https://minorplanetcenter.net/cgi-bin/checkmp.cgi). Confirm that you are finding known objects in your images before submitting any unknown objects.
4. If the `max_speed` parameter is too high, given the source density of your data, FindPOTATOs will start linking every detection to every other detection, and will slow down considerably. If too many tracklets are found, it is best to reduce `max_speed` until a manageable number of tracklets are produced.
5. Ensure that the timing of your observations is correct. Timing errors, particularly offsets between recorded time and actual time, are fairly common and can cause serious issues. One way to ensure that timing is correct is to compare a known, numbered, minor planet's position in your images with predicted positions from [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/). Timing errors will not necessarily be caught by use of the Minor Planet Checker described in #3.

## How to Cite
If you use or adapt this software please cite the following:

1. C. R. Nugent, N. Tan, and J. Bauer, FINDPOTATOs: Minor Planet Observation Linking Software. Accepted to The Planetary Science Journal, 2024. DOI: 10.3847/PSJ/ad94eb

2. If you use the Find_Orb-enabled screening, please cite [Bill Gray's work](https://projectpluto.com/find_orb.htm).

## Other linking software 
Before 2022, there was no publicly available minor planet detection linking software. But today, there's multiple options besides FindPOTATOs. You may be interested in:

1. [Heliolinc3D](https://github.com/lsst-dm/heliolinc2)
2. [PUMA](https://github.com/atlas-ifa/puma/tree/main)
3. [THOR](https://github.com/moeyensj/thor)
4. [CANFind](https://github.com/katiefasbender/CANFind)

## License 

[![Build Status](https://img.shields.io/static/v1.svg?label=CSL&message=software%20against%20climate%20change&color=green?style=flat&logo=github)](https://img.shields.io/static/v1.svg?label=CSL&message=software%20against%20climate%20change&color=green?style=flat&logo=github)