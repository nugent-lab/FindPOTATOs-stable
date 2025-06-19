import os
import subprocess
import re  # regular expressions, used to search for mean residuals in Find_orb output files
from time import sleep
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astroquery.skyview import SkyView
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.ticker as ticker
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from sklearn.neighbors import BallTree

def angular_separation_metric(x, y):
    """
    Custom distance function for sklearn to calculate angular separation
    between two points on the sky.

    Parameters:
    coord1, coord2 : array-like
        Arrays of coordinates in the form [RA, Dec], where RA and Dec
        are in degrees.

    Returns:
    separation : float
        The angular separation between the two coordinates in arcseconds.
    """
    ra1, dec1 = x
    ra2, dec2 = y
    
    # Create SkyCoord objects for each point
    sky_coord1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
    sky_coord2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')
    
    # Calculate the angular separation
    separation = sky_coord1.separation(sky_coord2).arcsecond
    
    return separation

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
        tracklet_found: string, equals True if passed
        res: associated residual with found tracklet
    """
    tracklet_found = False
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
            for line in open(os.path.expanduser("fo.txt"))
        )

        # save all inputs to find_orb
        open("output/AllPotentialTracklets.txt", "a+").writelines(
            [l for l in open(
                    os.path.expanduser("fo.txt")
                ).readlines()
            ]
        )
        for line in open(os.path.expanduser(elements_path)):
            li = line.strip()
            if not li.startswith("#"):
                open("output/AllPotentialTracklets.txt", "a").writelines(line.rstrip())
                open("output/AllPotentialTracklets.txt", "a").writelines("\n")
        open("output/AllPotentialTracklets.txt", "a").writelines("\n\n")

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
            tracklet_found = True

    else:
        print("Could not open file", os.path.expanduser(elements_path))
    return tracklet_found, res


def remove_stationary_sources(source_dataframes, thresh, showplots=False):
    """
    Compares dataframes, removes sources that are at same coordinate
    location to within threshold. Returns cleaned dataframes that
    consist of transitory sources.

    Args:
        source_dataframes: dictionary of dataframes of sources
        thresh: threshold in arcsec that we should consider stationary sources, in arcseconds.
        showplots: flag to show plots or not
    Returns:
        source_dataframes: dictionary of dataframes, with just the transitory sources
    """
    if showplots:
        for name, df in source_dataframes.items():
            plt.scatter(df.RA, df.Dec, alpha=0.4, label=name)
        plt.title("Sources before stationary cleaning.")
        plt.xlabel("Right ascension (degrees)")
        plt.ylabel("Declination (degrees)")
        plt.legend()
        plt.show()
        plt.close()
    for key, df in source_dataframes.items():
        df["ra_rad"] = np.radians(df["RA"])
        df["dec_rad"] = np.radians(df["Dec"])

    # this is a hybrid method for speed- find subset of 
    # matches within 2x threshold using inprecise Haversine approx
    # then check with astropy separation for realsies.
    double_thresh_rad= 2 * np.radians(thresh.to(u.deg).value)

    for key, df in source_dataframes.items():
        dupes_array_sum=np.zeros(len(df))
        for key2, df2 in source_dataframes.items(): #remove duplicates
            if key != key2: #but only compare different frames
                #tree=BallTree(df2[["RA", "Dec"]],  metric='pyfunc', func=angular_separation_metric)
                tree=BallTree(df2[["ra_rad", "dec_rad"]],  metric='haversine')

                #tree=BallTree(df2[["RA", "Dec"]],  metric='pyfunc', func=angular_separation_metric)
                indicies = tree.query_radius(df[["ra_rad", "dec_rad"]], r=double_thresh_rad)


                for i in range(len(indicies)): #for each source in df
                    findany=False
                    for j in range(len(indicies[i])):
                        if not findany:
                            ra1, dec1 = df2.iloc[indicies[i][j]].RA.item(), df2.iloc[indicies[i][j]].Dec.item()
                            ra2, dec2 = df["RA"].iloc[i].item(), df["Dec"].iloc[i].item()
                            sky_coord1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
                            sky_coord2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')
                            # Calculate the angular separation because its more accurate than haversine
                            separation_val = sky_coord1.separation(sky_coord2).arcsecond
                            #print("separation",separation_val, "thresh", removal_dist.value)
                            if separation_val < thresh.value:
                                #print("labeling for removal")
                                dupes_array_sum[i] += 1 #source in df has match in df2
                                findany = True
                            else:
                                dupes_array_sum[i] += 0 #source in df does not have match

                    #identify the duplicates but don't delete them, you'll need to reference
                    # them as this loop goes on
                    df["dupes"] = dupes_array_sum
                    #df.loc[:, "dupes"] = dupes_array_sum


    # Clean duplicates after all comparison is done.
    source_dataframes_moving = {f'{key}_moving': df.copy() for key, df in source_dataframes.items()}
    for key, df in source_dataframes_moving.items():
        source_dataframes_moving[key] = df[df['dupes'] < 1]
        source_dataframes_moving[key].reset_index(inplace=True, drop=True)

    if showplots:
        for name, df in source_dataframes.items():
            plt.scatter(df.RA, df.Dec, alpha=0.4, label=name)
        for name2, df2 in source_dataframes_moving.items():
            plt.scatter(df2.RA, df2.Dec, color='black', alpha=1, label=name2)
        plt.title("Sources after stationary cleaning. Transitory sources are black.")
        plt.xlabel("Right ascension (degrees)")
        plt.ylabel("Declination (degrees)")
        plt.legend()
        plt.show()
        plt.close()
    return source_dataframes_moving

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
    SkyView.URL = 'https://skyview.gsfc.nasa.gov/current/cgi/basicform.pl'
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
        print("SkyView query failed.")
        sky_image= np.zeros([bigger,bigger])
        wcs = False 

    return sky_image, wcs
    

def zero_pad(string):
    if len(string) < 5:
        num_zeros = 5 - len(string)
        padded_string = string + '0' * num_zeros
        return padded_string
    else:
        return string
    
def print_dict(dictionary, indent=0):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print('  ' * indent + str(key) + ':')
            print_dict(value, indent + 1)
        elif isinstance(value, list):
            print('  ' * indent + str(key) + ':')
            for item in value:
                print('  ' * (indent + 1) + str(item))
        else:
            print('  ' * indent + str(key) + ': ' + str(value))

def format_data (tracklet):
    """
    Format data in the Minor Planet Center (MPC) 80-char format
    Args:
        tracklet (df):a dataframe of tracklet information
            
    Returns:
        formatted_data (str): string with proper formatting
    """
    first=True
    for index, row in tracklet.iterrows():
        if first:
            formatted_data = "     "
            first = False
        else:
            formatted_data += "     "
        obstime=row["tracklet_time"].strftime("%Y %m %d")
        coord=row["coords"]
        formatted_data += "{}".format(row["tracklet_id"]) + "  C"
        formatted_data += "{}".format(obstime) + "."
        formatted_data += "{:1}".format(row["decimal_time"]) + " "
        formatted_data += (
            coord.to_string(style="hmsdms", pad=True, sep=" ", precision=2)
            + "         "
        )
        formatted_data += "{:.1f}".format(row["mag"]) + "   "
        formatted_data += (
            row["band"]
            + "    "
            + str(row["observatory_code"])
            + "\n"
        )
    return formatted_data

def create_diagnostic_figure(tracklet, figname, show_sky_image):
    """
    Args:
        tracklet (df): dataframe with relevant tracklet information
        figname (str) : string with name of figure 
        show_sky_image (bool): if you want image of the sky from SDSS
    Returns:
        none, though it will save an image
    """
            
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    title_str = (
        "Tracklet:"
        + tracklet["tracklet_id"].iloc[0]
        + " Velocity metric:"
        + str(np.round(tracklet["vdiff"].std(), 4))
    )
    fig.suptitle(title_str, fontsize=16)

    # Plot magnitude
    axs[0].scatter(np.arange(len(tracklet)),tracklet["mag"], c="purple")
    axs[0].set_title("Lightcurve")
    axs[0].set_xlabel("Gridlines are 0.5mag", fontsize=8)
    axs[0].set_ylabel("Magnitudes, mag")
    axs[0].grid(
        True, linestyle="--", which="both", color="gray", linewidth=0.5
    )
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[0].set_xticks(np.arange(len(tracklet)))
    axs[0].xaxis.grid(False)

    # Plot tracklet in RA/DEC
    #tracklet_coord = SkyCoord(ra=tracklet["ra"], dec=tracklet["dec"], unit="deg")
    ra_list=tracklet["ra"]
    dec_list=tracklet["dec"]
    axs[1].scatter(
        ra_list, dec_list, c="purple", label="Tracklet"
    )
    axs[1].set_title("Tracklet position on sky.")
    axs[1].set_xlabel("RA, deg.", fontsize=8)
    axs[1].set_ylabel("Dec, deg")
    axs[1].legend(fontsize=6)
    axs[1].grid(
        True, linestyle="--", which="both", color="gray", linewidth=0.5
    )
    # axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.0027))
    # axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.0027))
    axs[1].set(aspect="equal")

    # Calculate largest extent in x or y
    x_span = max(ra_list) - min(ra_list) + 0.01
    y_span = max(dec_list) - min(dec_list) + 0.01
    max_span = max(x_span, y_span)

    # Set aspect ratio manually by adjusting xlim and ylim
    mid_x = (max(ra_list) + min(ra_list)) / 2
    mid_y = (max(dec_list) + min(dec_list)) / 2
    axs[1].set_xlim(mid_x - max_span / 2, mid_x + max_span / 2)
    axs[1].set_ylim(mid_y - max_span / 2, mid_y + max_span / 2)

    # Image of sky from another source
    six_title = "SDSS sky image overlayed with tracklet (red)"
    six_xlabel = "Reject tracklet if circles overlap sources."
    six_ylabel = ""
    xspan_deg = u.Quantity(x_span, unit=u.deg)
    yspan_deg = u.Quantity(y_span, unit=u.deg)
    width_pix = xspan_deg.to(u.arcsec).value / 0.396
    height_pix = yspan_deg.to(u.arcsec).value / 0.396

    if show_sky_image == True:
        sky_image, wcs = query_skyview(
            tracklet["ra"].mean(),
            tracklet["dec"].mean(),
            np.round(width_pix),
            np.round(height_pix),
        )
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(sky_image)
        if wcs == False:
            axs[2].imshow(
                sky_image.data,
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            axs[2].remove()
            axs[2] = plt.subplot(133, projection=wcs)
            pix_x_a, pix_y_a = tracklet["coords"][0].to_pixel(wcs)
            pix_x_b, pix_y_b = tracklet["coords"][1].to_pixel(wcs)
            pix_x_c, pix_y_c = tracklet["coords"][2].to_pixel(wcs)
            axs[2].imshow(
                sky_image.data,
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            axs[2].scatter(
                [pix_x_a, pix_x_b, pix_x_c],
                [pix_y_a, pix_y_b, pix_y_c],
                marker="o",
                facecolors="none",
                edgecolors="red",
                s=40,
            )
            axs[2].set_xlim(0, sky_image.data.shape[1])
            axs[2].set_ylim(0, sky_image.data.shape[0])

    axs[2].set_title(six_title, fontsize=8)
    axs[2].set_xlabel(six_xlabel, fontsize=8)
    axs[2].set_ylabel(six_ylabel)

    plt.tight_layout()
    plt.savefig(figname)
    plt.close("all")
    return