from astropy import units as u

########## PARAMETERS ##########
# Input filename provided at call
input_directory = "sample_source_files/"

# Location of fits files
image_path = "images/"

# Length 5 string to start increments with.
starting_tracklet_id = "00000"  # string, length = 5

# your unique tracklet code, will precede the tracklet id
# your tracklets will be named tracklet_tag+starting_tracklet_id
tracklet_tag = "aa"  # string, reccomended length = 2

# turn this on if you want to save diagnostic images and plots of each tracklet
save_tracklet_images = True

# query skyveiw for cutout of sky
show_sky_image = True  

# turn this on (='y') if you want observations exported in XML ADES format
# more on ADES here: https://minorplanetcenter.net/iau/info/ADES.html
export_ades = True

# the max distance between two sources in order for them
# to be considered the same, and therefore stationary, and removed.
stationary_dist_deg = 0.1 * u.arcsec

# the maximum amount brightness can vary across a tracklet
max_mag_variance = 2  # mag

# maximum speed an asteroid can travel to be detected
# you don't want this more than ~1/5th of the size of the frame, anything
# faster is both unlikely and undetectable as it will leave the frame before
# three detections can be made
max_speed = 0.05  # arcseconds/second

# Maximum standard deviation in velocity between detections
# Close-approaching objects may have apparent velocity changes over
# a tracklet, so this should be higher to find close NEAs
velocity_metric_threshold = 0.01   # arcseconds/second

# minimum angle between a-b, b-c, will be used to search for det in frame c
min_tracklet_angle = 160  # degrees

# Timing uncertanty of observations, in seconds.
# Code will pick the biggest radius derived from min_tracklet_angle and
# timing_uncertanty to determine search radius.
# You can increase timing_uncertanty to increase search radius; this is
# especially helpful if you need to find objects whose apparent speed
# across the plane of the sky changes betweeen A-B and B-C.
timing_uncertainty = 10  # seconds

# smallest distance you will accept an asteroid to move between frames
min_dist_deg = 0.1  # arcseconds

# Check tracklets using Bill Gray's Find Orb for accuracy.
findorb_check = False  
# values below this threshold will be rejected
maximum_residual = 0.9  # arcseconds

# Optional; screen on ml_probs associated with sources
# this is intended to allow the use of a machine learning
# algorithm to assign a probability of source "realness" 
# to each detection
# If the input file doesn't have a column named ml_probs
# then the code will ignore and move on.
ml_thresh=0.55

########## ADES PARAMETERS ##########
# header information. None of these will be changed by the
# code.
# more on ADES here: https://minorplanetcenter.net/iau/info/ADES.html
ades_dict = {
    "mpcCode": "535", 
    "observatoryName": "Palermo Astronomical Observatory",
    "submitter": "D. Bowie",
    "observers": "B. Yonce",
    "measurers": "D. Bowie",
    "coinvestigators": "F. Apple",
    "telescope_design": "reflector",
    "telescope_aperture": "1.1",
    "telescope_detector": "CCD",
    "fundingSource": "NASA",
    "comment": "None",
}
# Observation information. Some of these are dummy values that will
# be updated later.
ades_obs_dict = {
    # various codes can be found here:
    # https://www.minorplanetcenter.net/iau/info/ADESFieldValues.html
    # You should add fields as appropriate for your data.
    "trkSub": "None",  # Observer-assigned tracklet identifier
    "mode": "CCD",  # Mode of instrumentation (probably CCD)
    "stn": "535",  # Observatory code assigned by the MPC
    # UTC date and time of the observation, ISO 8601 exended format,
    # i.e. yyyy-mm-ddThh:mm:ss.sssZ.
    # The reported time precision should be appropriate for the
    # astrometric accuracy, but no more than 6 digits are permitted
    # after the decimal. The trailing Z indicates UTC and is required.
    "obsTime": "1801-01-01T12:23:34.12Z",
    #'rmsTime': '3' #Random uncertainty in obsTime in seconds as estimated by the observer
    "ra": "3.639",  # decimal degrees in the J2000.0 reference frame
    "dec": "16.290",  # decimal degrees in the J2000.0 reference frame
    # For ra-dec and deltaRA- deltaDec observations, the random component
    # of the RA*COS(DEC) and DEC uncertainty (1-sigma) in arcsec as estimated
    # by the observer as part of the image processing and astrometric reduction.
    "rmsRA": "1.5",
    "rmsDec": "1.5",
    # Correlation between RA and DEC or between distance and PA that may
    # result from the astrometric reduction.
    #'rmsCorr': '-0.214',
    "astCat": "UBSC",  # Star catalog used for the astrometric reduction
    # ‘UNK’, will be used for some archival observations to indicate that
    # the astrometric catalog is unknown.
    "mag": "21.91",  # Apparent magnitude in specified band.
    "rmsMag": "0.15",  # Apparent magnitude uncertainty, 1-sigma
    "band": "g",  # Passband designation for photometry.
    "photCat": "Gaia3",  # Star catalog used for the photometric reduction.
    # full list here: https://www.minorplanetcenter.net/iau/info/ADESFieldValues.html
    "remarks": "None",  # A comment provided by the observer. This field can be
    # used to report additional information that is not reportable in the notes
    # field, but that may be of relevance for interpretation of the observations.
    # Should be used sparingly by major producers.
}
#################################
