import os
import subprocess
import re  # regular expressions, used to search for mean residuals in Find_orb output files
from time import sleep
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astroquery.skyview import SkyView
from astroquery.simbad import Simbad
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sklearn.neighbors import BallTree
from astroquery.ipac.irsa import Irsa

def remove_file_if_exists(filename):
    """
    Checks if file exists, removes it if it does

    Args:
        filename: name of file to check

    Returns:
        None
    """
    if os.path.exists(filename):
        os.remove(filename)
        #print(f"File '{filename}' removed.")
    #else:
    #    print(f"File '{filename}' does not exist.")
    return 

def find_orb(maxResidual, nullResid=True, MOIDLim=False):
    """
    Feeds observations in MPC format that are located ~/.find_orb/fo.txt to
    the non-interactive version of find orb, fo. find_orb stores orbital
    elements in  ~/.find_orb/elements.txt, which this function will read to
    find the mean residual to the orbital fit. If the mean residual is less
    than maxResidual (specified in ") and all observations in
    ~/.find_orb/fo.txt was used to generate the orbital fit, then the
    function will return True. In other cases (e.g. find_orb doesn't run;
    mean residual greater than maxResidual; not all observations in
    ~/.find_orb/fo.txt used), the function will return False.

    Args:
        maxResidual: float, maximum residual allowed for tracklet to be approved

    Returns:
        trackletFound: string, equals 'yes' if passed
        res: associated residual with found tracklet
    """
    trackletFound = "n"
    elements_path = "~/.find_orb/elements.txt"  # for mac
    if os.path.exists(os.path.expanduser(elements_path)):
        os.remove(os.path.expanduser(elements_path))
    # this line works on mac & some unix installs but not the MGHPCC
    # sp = Popen(['cd ~/.find_orb\n~/find_orb/find_orb/fo fo.txt -c'], shell=True)
    # this line is for the MGHPCC. Either way, you need the directory where your fo files are
    # the subprocess module reacts poorly to the supercomputer.
    # os.system('fo fo.txt -c')
    findorb_call = "fo fo.txt -c"
    sp = subprocess.call(findorb_call, shell=True)
    totSleep = 0
    # wait for find_orb to create elements.txt. If it takes longer than 20 seconds
    # then find_orb probably can't find an orbit.
    while not os.path.exists(os.path.expanduser(elements_path)):
        sleep(0.2)
        totSleep = totSleep + 0.2
        if totSleep > 20:
            break
    if os.path.exists(os.path.expanduser(elements_path)):
        if os.path.getsize(os.path.expanduser(elements_path)) == 0:
            sleep(0.2)
        # numObs = sum(1 for line in open(os.path.expanduser("~/.find_orb/fo.txt")))
        numObs = sum(
            1
            for line in open(os.path.expanduser("/projectnb/ct-ast/findPOTATOs/fo.txt"))
        )

        # save all inputs to find_orb
        open("outputs/AllPotentialTracklets.txt", "a+").writelines(
            [l for l in open(
                    os.path.expanduser("/projectnb/ct-ast/findPOTATOs/fo.txt")
                ).readlines()
            ]
        )
        for line in open(os.path.expanduser(elements_path)):
            li = line.strip()
            if not li.startswith("#"):
                open("outputs/AllPotentialTracklets.txt", "a").writelines(line.rstrip())
                open("outputs/AllPotentialTracklets.txt", "a").writelines("\n")
        open("outputs/AllPotentialTracklets.txt", "a").writelines("\n\n")

        resCheck = False
        for line in open(os.path.expanduser(elements_path)):
            match = re.search('mean residual (\d+)".(\d+)', line)
            match2 = re.search("MOID: (\d+).(\d+)", line)
            if match:
                res = int(match.group(1)) + float(("0." + match.group(2)))
                if nullResid:
                    if (res < maxResidual) & (res > 0):  # maxResidual in "
                        resCheck = True
                    else:
                        resCheck = False
                else:
                    if res < maxResidual:  # maxResidual in "
                        resCheck = True
                    else:
                        resCheck = False
            if match2:
                if MOIDLim:
                    MOID = int(match2.group(1)) + float(("0." + match2.group(2)))
                    if MOID > MOIDLim:
                        print("MOID:", MOID, " exceeds MOIDLim:", MOIDLIM)
                        break
        if resCheck:
            trackletFound = "y"

    else:
        print("Could not open file", os.path.expanduser(elements_path))
    return trackletFound, res


def remove_stationary_sources(df1, df2, df3, thresh, showplots='n'):
    """
    Compares three dataframes, removes sources that are at same coordinate
    location to within threshold. Returns cleaned dataframes that
    consist of transitory sources.

    The for loop through the matches is a bit unpythonic, could be
    improved but works for now.

    Args:
        df1: first dataset to be considered, dataframe. Needs columns ra_rad, dec_rad (ra and dec in radians).
        df2: second dataset that will be compared to first.  Needs columns ra_rad, dec_rad (ra and dec in radians).
        df3: third datset that will be compared to first.  Needs columns ra_rad, dec_rad (ra and dec in radians).
        thresh: threshold in arcsec that we should consider stationary sources, in arcseconds.

    Returns:
        df1_moving: just the transitory sources in the first dataframe
        df2_moving: just the transitory sources in the second dataframe
        df3_moving: just the transitory sources in the third dataframe
    """
    if showplots=='y':
        plt.scatter(df1.RA, df1.Dec, color='red', alpha=0.4, label='A')
        plt.scatter(df2.RA, df2.Dec, color='blue',alpha=0.4, label='B')
        plt.scatter(df3.RA, df3.Dec, color='yellow', alpha=0.4, label='C')
        plt.show()
        plt.close()
    # convert threshold (arsec) to degrees, then to radians
    thresh_rad = np.radians(thresh.to(u.deg).value)
    # print("thresh_rad",thresh_rad)

    # intialize output dataframes
    # gonna delete the duplicates from the df?_moving dataframes
    df1_moving = df1.copy(deep=True)
    df2_moving = df2.copy(deep=True)
    df3_moving = df3.copy(deep=True)

    tree1 = BallTree(df1[["ra_rad", "dec_rad"]], leaf_size=5, metric="haversine")
    tree2 = BallTree(df2[["ra_rad", "dec_rad"]], leaf_size=5, metric="haversine")
    tree3 = BallTree(df3[["ra_rad", "dec_rad"]], leaf_size=5, metric="haversine")

    # compare 1 to 2
    dupes_array1_2=[]
    indicies = tree2.query_radius(df1[["ra_rad", "dec_rad"]], r=thresh_rad)
    for i in range(len(indicies)): #for each source in df2
        if len(indicies[i]) > 0: #these are the matches in df1
            dupes_array1_2.append(1)
        else:
            dupes_array1_2.append(0)
    # compare 1 to 3
    dupes_array1_3=[]
    indicies = tree3.query_radius(df1[["ra_rad", "dec_rad"]], r=thresh_rad)
    for i in range(len(indicies)): #for each source in df2
        if len(indicies[i]) > 0: #these are the matches in df1
            dupes_array1_3.append(1)
        else:
            dupes_array1_3.append(0)

    df1_moving["dupes1_2"] = dupes_array1_2
    df1_moving["dupes1_3"] = dupes_array1_3
    df1_moving = df1_moving[df1_moving['dupes1_2'] < 1]
    df1_moving = df1_moving[df1_moving['dupes1_3'] < 1]

    # compare 2 to 1
    dupes_array2_1=[]
    indicies = tree1.query_radius(df2[["ra_rad", "dec_rad"]], r=thresh_rad)
    for i in range(len(indicies)): #for each source in df2
        if len(indicies[i]) > 0: #these are the matches in df1
            dupes_array2_1.append(1)
        else:
            dupes_array2_1.append(0)
    # compare 2 to 3
    dupes_array2_3=[]
    indicies = tree3.query_radius(df2[["ra_rad", "dec_rad"]], r=thresh_rad)
    for i in range(len(indicies)): #for each source in df2
        if len(indicies[i]) > 0: #these are the matches in df1
            dupes_array2_3.append(1)
        else:
            dupes_array2_3.append(0)
    
    df2_moving["dupes2_1"] = dupes_array2_1
    df2_moving["dupes2_3"] = dupes_array2_3
    df2_moving = df2_moving[df2_moving['dupes2_1'] < 1]
    df2_moving = df2_moving[df2_moving['dupes2_3'] < 1]

    # compare 3 to 1
    dupes_array3_1=[]
    indicies = tree1.query_radius(df3[["ra_rad", "dec_rad"]], r=thresh_rad)
    for i in range(len(indicies)): #for each source in df2
        if len(indicies[i]) > 0: #these are the matches in df1
            dupes_array3_1.append(1)
        else:
            dupes_array3_1.append(0)
    
    # compare 3 to 2
    dupes_array3_2=[]
    indicies = tree2.query_radius(df3[["ra_rad", "dec_rad"]], r=thresh_rad)
    for i in range(len(indicies)): #for each source in df2
        if len(indicies[i]) > 0: #these are the matches in df1
            dupes_array3_2.append(1)
        else:
            dupes_array3_2.append(0)
    
    df3_moving["dupes3_1"] = dupes_array3_1
    df3_moving["dupes3_2"] = dupes_array3_2
    df3_moving = df3_moving[df3_moving['dupes3_1'] < 1]
    df3_moving = df3_moving[df3_moving['dupes3_2'] < 1]

    # Reset index after dropping rows
    df1_moving.reset_index(inplace=True, drop=True)
    df2_moving.reset_index(inplace=True, drop=True)
    df3_moving.reset_index(inplace=True, drop=True)

    # print(
    #     "Percentage of sources remaining from original dataframes df1, df2, df3:",
    #     round(len(df1_moving) / len(df1), 2),
    #     round(len(df2_moving) / len(df2), 2),
    #     round(len(df3_moving) / len(df3), 2),
    # )
    if showplots=='y':
        plt.scatter(df1.RA, df1.Dec, color='red', alpha=0.2, label='A')
        plt.scatter(df2.RA, df2.Dec, color='blue',alpha=0.2, label='B')
        plt.scatter(df3.RA, df3.Dec, color='yellow', alpha=0.2, label='C')
        plt.scatter(df1_moving.RA, df1_moving.Dec, color='black', alpha=1, label='A')
        plt.scatter(df2_moving.RA, df2_moving.Dec, color='black',alpha=1, label='B')
        plt.scatter(df3_moving.RA, df3_moving.Dec, color='black', alpha=1, label='C')
        plt.show()
        plt.close()
    # convert threshold (
    return df1_moving, df2_moving, df3_moving


def return_thumbnail(x_pos, y_pos, telescope_image):
    """
    Return thumbnail of source 
    Args:
        x_pos: float, RA of source you want thumbnail of
        y_pos: float, Dec of source you want thumbnail of
        telescope_image: array, actual image
    Returns:
        cropped_image: image of source, scaled
    """
    buffer = 9
    left = int(x_pos) - (buffer - 1)
    right = int(x_pos) + (buffer)
    upper = int(y_pos) + (buffer - 1)
    lower = int(y_pos) - (buffer)

    if left < 0:
        left = 0
    elif upper < 0:
        upper = 0
    elif right > telescope_image.shape[1]:
        right = telescope_image.shape[1]
    elif lower >= telescope_image.shape[0]:
        lower = telescope_image.shape[0]

    # Crop the image and scale it
    cropped_array = telescope_image[lower:upper, left:right]
    try:
        scaled_array = (
            (cropped_array - np.min(cropped_array))
            / (np.max(cropped_array) - np.min(cropped_array))
            * 255
        ).astype(np.uint8)
        cropped_image = Image.fromarray(scaled_array, mode="L")
    except ValueError:
        print("Error rescaling, here's the cropped_array", cropped_array)
        cropped_image =np.zeros((17, 17))
    return cropped_image


def increment_identifier(identifier='00000'):
    """
    Produces unique alphanumeric identifier for tracklets. 
    Args:
        identifier: the last identifier used. 
    Returns:
        next identifier in sequence
    """
    allowable_id_chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    index = allowable_id_chars.index(identifier[-1])
    if index < len(allowable_id_chars) - 1:
        return identifier[:-1] + allowable_id_chars[index + 1]
    return increment_identifier(identifier[:-1]) + allowable_id_chars[0]


def query_skyview(ra, dec, width_pix, height_pix, survey='SDSSr'):
    """
    Query SkyView to retrieve an image of the sky at the given RA and Dec.

    Args:
        ra (float): Right ascension (RA) in degrees.
        dec (float): Declination (Dec) in degrees.
        width (int): width of image in pix
        height (int): height of image in pix
        survey (str): Survey to retrieve image from (default='SDSSr').
        
    Returns:
        sky_image = image of sky 
        wcs = header info about sky
    """
    if width_pix > height_pix:
        bigger=int(width_pix)+10
    else:
        bigger=int(height_pix)+10
    # Query SkyView
    try: 
        images = SkyView.get_images(
            position=f'{ra} {dec}', 
            coordinates='J2000', 
            survey=survey, 
            pixels=str(bigger)
        )
        sky_image = images[0][0].data
        #sky_image = np.flip(sky_image, axis=1)
        #sky_image = images[0][0]
        header = images[0][0].header
        wcs = WCS(header)
    except:
        print("query failed.")
        sky_image= np.zeros([bigger,bigger])
        wcs = False 

    return sky_image, wcs
    
    