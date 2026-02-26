import xml.etree.ElementTree as XMLElement
import xml.dom.minidom as minidom

# T. Linder, June 2023
# refactored by C. Nugent, September 2023


def generate_xml(data, obs_data):
    """
    Takes all the ADES 2017 data and puts it into XML tree.
    Call this program the first time you're creating the
    XML submission, this call includes header information.
    Args:
        data: dictionary, the first half of the ADES fields
        obs_data: dictionary, with optical-observatory related data
        filename: string, name of output file
    Returns:
        XMLElement, ades, obsData: the xml data
    """
    #
    ades = XMLElement.Element("ades", version="2017")
    obsBlock = XMLElement.SubElement(ades, "obsBlock")

    # ObsBlock Information Section to the next #--
    # ---------------------------------------------------------
    obsContext = XMLElement.SubElement(obsBlock, "obsContext")

    # Observatory Information section to the next ####
    # ##############################################################
    observatory = XMLElement.SubElement(obsContext, "observatory")

    mpcCode = XMLElement.SubElement(observatory, "mpcCode")
    mpcCode.text = data["mpcCode"]

    name = XMLElement.SubElement(observatory, "name")
    name.text = data["observatoryName"]
    ##############################################################

    # data People
    submitter = XMLElement.SubElement(obsContext, "submitter")
    submitter_name = XMLElement.SubElement(submitter, "name")
    submitter_name.text = data["submitter"]

    observers = XMLElement.SubElement(obsContext, "observers")
    observers_name = XMLElement.SubElement(observers, "name")
    observers_name.text = data["observers"]

    coinvestigators = XMLElement.SubElement(obsContext, "coinvestigators")
    coinvestigators_name = XMLElement.SubElement(coinvestigators, "name")
    coinvestigators_name.text = data["coinvestigators"]

    measurers = XMLElement.SubElement(obsContext, "measurers")
    measurers_name = XMLElement.SubElement(measurers, "name")
    measurers_name.text = data["measurers"]

    # data obsContext/telescope
    telescope = XMLElement.SubElement(obsContext, "telescope")
    design = XMLElement.SubElement(telescope, "design")
    design.text = data["telescope_design"]
    aperture = XMLElement.SubElement(telescope, "aperture")
    aperture.text = data["telescope_aperture"]
    detector = XMLElement.SubElement(telescope, "detector")
    detector.text = data["telescope_detector"]

    # data obsContext/fundingSource
    fundingSource = XMLElement.SubElement(obsContext, "fundingSource")
    fundingSource.text = data["fundingSource"]

    # data obsContext/comment
    comment = XMLElement.SubElement(obsContext, "comment")
    comment_line = XMLElement.SubElement(comment, "line")
    comment_line.text = data["comment"]
    # ---------------------------------------------------------

    # obsData Information to the next #$$$$$
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    obsData = XMLElement.SubElement(obsBlock, "obsData")
    optical = XMLElement.SubElement(obsData, "optical")

    # Set optical values in one neat loop
    for key, value in obs_data.items():
        element = XMLElement.SubElement(optical, key)
        element.text = str(value)

    # Pass this back, and then use update_xml to add observations as needed.
    return XMLElement, ades, obsData


def update_xml(XMLElement, ades, obsData, obs_data):
    """
    Update xml with the updated obs data.
    Args:
        XMLElement, ades, obsData: the  xml data
        obs_data: dictionary, with optical-observatory related data
    Returns:
        XMLElement, ades, obsData: the updated xml data
    """
    optical = XMLElement.SubElement(obsData, "optical")

    for key, value in obs_data.items():
        element = XMLElement.SubElement(optical, key)
        element.text = str(value)

    return XMLElement, ades, obsData


if __name__ == "__main__":
    ades_dict = {
        "mpcCode": "535",  # MPC-assigned observatory code
        "observatoryName": "Palermo Astronomical Observatory",
        "submitter": "Janelle Moneae",
        "observers": "David Bowie",
        "measurers": "Mobb Deep",
        "coinvestigators": "Etta James",
        "telescope_design": "reflector",
        "telescope_aperture": "1.1",
        "telescope_detector": "CCD",
        "fundingSource": "NASA",
        "comment": "None",
    }
    ades_obs_dict = {
        # various codes can be found here:
        # https://www.minorplanetcenter.net/iau/info/ADESFieldValues.html
        # 'permID': '04933',#IAU permanent designation
        # 'provID': '2022 LB1',#MPC provisional designation (in unpacked form)
        # for unnumbered objects.
        "trkSub": "None",  # Observer-assigned tracklet identifier
        "mode": "CCD",  # Mode of instrumentation (probably CCD)
        "stn": "644",  # Observatory code assigned by the MPC
        # UTC date and time of the observation, ISO 8601 exended format,
        # i.e. yyyy-mm-ddThh:mm:ss.sssZ.
        # The reported time precision should be appropriate for the
        # astrometric accuracy, but no more than 6 digits are permitted
        # after the decimal. The trailing Z indicates UTC and is required.
        # Reccomend use "from astropy.time import Time" and then something like
        # str(time.isot)+'Z' where time is a Time astropy object with the
        # time of your observation.
        "obsTime": "2016-08-29T12:23:34.12Z",
        #'rmsTime': '3' #Random uncertainty in obsTime in seconds as estimated by the observer
        "ra": "215.10254",  # decimal degrees in the J2000.0 reference frame
        "dec": "-12.547",  # decimal degrees in the J2000.0 reference frame
        # For ra-dec and deltaRA- deltaDec observations, the random component
        # of the RA*COS(DEC) and DEC uncertainty (1σ) in arcsec as estimated
        # by the observer as part of the image processing and astrometric reduction.
        "rmsRA": "0.015",
        "rmsDec": "0.015",
        # Correlation between RA and DEC or between distance and PA that may
        # result from the astrometric reduction.
        #'rmsCorr': '-0.214',
        "astCat": "Gaia2",  # Star catalog used for the astrometric reduction
        # ‘UNK’, will be used for some archival observations to indicate that
        # the astrometric catalog is unknown.
        "mag": "21.91",  # Apparent magnitude in specified band.
        "rmsMag": "0.25",  # Apparent magnitude uncertainty, 1-sigma
        "band": "g",  # Passband designation for photometry.
        "photCat": "Gaia3",  # Star catalog used for the photometric reduction.
        # full list here: https://www.minorplanetcenter.net/iau/info/ADESFieldValues.html
        #'photAp': '13.3', #Photometric aperture radius in arcsec.
        #'logSNR': '0.78', #The log10 of the signal-to-noise ratio of the source
        # in the image integrated on the entire aperture used for the astrometric centroid.
        #'seeing': '0.8', #Size of seeing disc in arcsec, measured at Full-Width,
        # Half-Max (FWHM) of target point spread function (PSF).
        "exp": "20.0",  # Exposure time in seconds.
        #'remarks': 'None' #A comment provided by the observer. This field can be
        # used to report additional information that is not reportable in the notes
        # field, but that may be of relevance for interpretation of the observations.
        # Should be used sparingly by major producers.
    }
