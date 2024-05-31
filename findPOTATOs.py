#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from astropy.time import Time, TimeDelta
from sklearn.neighbors import BallTree
from os.path import exists
from astropy.coordinates import SkyCoord, Angle, Distance
from astropy.visualization import ZScaleInterval
import matplotlib.ticker as ticker
import sys
import fitsio  # if you want thumbnails of sources
from linking_library import *
from ades import *
from parameters import *

# call:
# python findPOTATOs.py [data_id]

#  C.R. Nugent, N. Tan, S. Matsumoto
#  Last updated May 15, 2024

data_id = sys.argv[1]

print("Running with the following parameters from parameters.py")
print("Input directory:", input_directory)
print("Image path:", image_path)
print("Starting tracklet ID:", starting_tracklet_id)
print("Save tracklet images:", save_tracklet_images)
print("Show sky image:", show_sky_image)
print("Export ADES:", export_ades)
print("Stationary distance:", stationary_dist_deg)
print("Max magnitude variance:", max_mag_variance)
print("Max speed:", max_speed)
print("Velocity metric threshold:", velocity_metric_threshold)
print("Min tracklet angle:", min_tracklet_angle)
print("Timing uncertainty:", timing_uncertainty)
print("Maximum residual:", Maximum_residual)
print("Min distance:", min_dist_deg)
print("Findorb check:", findorb_check)
print("Tracklet tag:", tracklet_tag)

if export_ades == "y":
    print("******************************")
    print(
        "You've chosen to export to ADES format. Please review these ADES parameters for accuracy. If anything is incorrect, quit with ctrl-D and update the values in parameters.py."
    )
    print_dict(ades_dict)
    name = input("Press any key to continue.")
    print(
        "Please review these observation-related ADES parameters for accuracy. Please refrence the ADES documentation if you have questions. If anything is incorrect, quit with ctrl-D and update the values in parameters.py."
    )
    print("mode:", ades_obs_dict["mode"])
    print("rmsRA:", ades_obs_dict["rmsRA"])
    print("rmsDec:", ades_obs_dict["rmsDec"])
    print("rmsMag:", ades_obs_dict["rmsMag"])
    print("band:", ades_obs_dict["band"])
    print("photCat:", ades_obs_dict["photCat"])
    name = input("Press any key to continue.")


input_filename = "image_triplets_" + str(data_id) + ".csv"
image_triplets_list = pd.read_csv(input_filename)
tracklet_num = 0

os.makedirs("output/" + data_id, exist_ok=True)
# Clean up any output files from a previous run
tracklet_features = "output/" + data_id + "/tracklet_features" + data_id + ".txt"
trackletfilename = "output/" + data_id + "/tracklets_" + data_id + ".txt"
xml_filename = "output/" + data_id + "/tracklets_ADES_" + data_id + ".xml"
remove_file_if_exists(tracklet_features)
remove_file_if_exists(trackletfilename)
remove_file_if_exists(xml_filename)

if export_ades == "y":
    xml_tracklet_found = "n"

for m in np.arange(len(image_triplets_list)):
    file_a = image_triplets_list.filea[m]
    file_b = image_triplets_list.fileb[m]
    file_c = image_triplets_list.filec[m]

    # Put frames in exposure order, so that frame a is first, b is second, and c is third.
    init_a = pd.read_csv(input_directory + file_a)
    init_b = pd.read_csv(input_directory + file_b)
    init_c = pd.read_csv(input_directory + file_c)

    init_a_time = Time(init_a.mjd[0].astype(float), format="mjd", scale="utc")
    init_b_time = Time(init_b.mjd[0].astype(float), format="mjd", scale="utc")
    init_c_time = Time(init_c.mjd[0].astype(float), format="mjd", scale="utc")

    # put frames in order
    order_frames = pd.DataFrame(
        {
            "names": [file_a, file_b, file_c],
            "times": [init_a_time, init_b_time, init_c_time],
        }
    )

    order_frames.sort_values(by=["times"], inplace=True)
    order_frames.reset_index(inplace=True)

    a = pd.read_csv(input_directory + order_frames.names[0])
    b = pd.read_csv(input_directory + order_frames.names[1])
    c = pd.read_csv(input_directory + order_frames.names[2])
    print(
        "Checking file triplet number",
        m,
        "consisting of:",
        order_frames.names[0],
        order_frames.names[1],
        order_frames.names[2],
    )
    a_time = Time(a.mjd[0].astype(float), format="mjd", scale="utc")
    b_time = Time(b.mjd[0].astype(float), format="mjd", scale="utc")
    c_time = Time(c.mjd[0].astype(float), format="mjd", scale="utc")
    decimal_time_a = str(a_time).split(".")
    decimal_time_b = str(b_time).split(".")
    decimal_time_c = str(c_time).split(".")
    decimal_time_a = decimal_time_a[1][:5]
    decimal_time_a = zero_pad(decimal_time_a)
    decimal_time_b = decimal_time_b[1][:5]
    decimal_time_b = zero_pad(decimal_time_b)
    decimal_time_c = decimal_time_c[1][:5]
    decimal_time_c = zero_pad(decimal_time_c)

    # The nearest neighbors code (balltree, haversine metric)
    # needs RA and Dec in radians. This code is used a lot here
    # so we'll just create the needed columns once right now.
    a["ra_rad"] = np.radians(a["RA"])
    a["dec_rad"] = np.radians(a["Dec"])
    b["ra_rad"] = np.radians(b["RA"])
    b["dec_rad"] = np.radians(b["Dec"])
    c["ra_rad"] = np.radians(c["RA"])
    c["dec_rad"] = np.radians(c["Dec"])
    a_moving, b_moving, c_moving = remove_stationary_sources(
        a, b, c, stationary_dist_deg, showplots="n"
    )

    # if any frames are empty after ML and/or stationary source screening, then no tracklets will
    # be found here. Skip to next frame triplet
    if a_moving.empty or b_moving.empty or c_moving.empty:
        print("One of the frames has no sutiable sources. Skipping to next iteration.")
        print(len(a_moving), len(b_moving), len(c_moving))
        continue

    # In the following, distances stored in dataframes are in radians, because that's
    # what python likes. RA and DEC are stored in degrees.
    start_time = datetime.now()
    # After removing stationary sources, tree of b comparing to a.

    # minimum speed an asteroid can travel to be detected, in arcseconds/second
    # this is calculated based on our astromemtric accuracy and time between frames
    time_interval_s = (b_time - a_time).sec

    max_dist_rad = np.radians(max_speed * time_interval_s / 3600)
    min_dist_rad = np.radians(min_dist_deg / 3600)

    tree_a = BallTree(a_moving[["ra_rad", "dec_rad"]], leaf_size=5, metric="haversine")
    indicies_b, distances_b = tree_a.query_radius(
        b_moving[["ra_rad", "dec_rad"]], r=max_dist_rad, return_distance=True
    )
    pair_id = []
    a_source_index = []
    b_source_index = []
    ab_dist = []
    dec_a = []
    dec_b = []
    ra_a = []
    ra_b = []
    mag_a = []
    mag_b = []
    observatory_code = []
    band = []
    tracklet_count = 0
    prob_a = []
    prob_b = []
    # We want every source in a that is within a certain radius of each source in b, but not too slow.
    for i in range(len(indicies_b)):
        for j in range(len(indicies_b[i])):
            if distances_b[i][j] > min_dist_rad:
                pair_id.append(tracklet_count)
                a_source_index.append(indicies_b[i][j])
                b_source_index.append(i)
                print("found pair", a_moving["RA"][indicies_b[i][j]], b_moving["RA"][i])
                ab_dist.append(distances_b[i][j])
                dec_a.append(a_moving["Dec"][indicies_b[i][j]])
                dec_b.append(b_moving["Dec"][i])
                ra_a.append(a_moving["RA"][indicies_b[i][j]])
                ra_b.append(b_moving["RA"][i])
                mag_a.append(a_moving["magnitude"][indicies_b[i][j]])
                mag_b.append(b_moving["magnitude"][i])
                observatory_code.append(b_moving["observatory_code"][i])
                band.append(b_moving["band"][i])
                tracklet_count += 1

    # to pandas df
    candidate_tracklet = pd.DataFrame(pair_id)
    candidate_tracklet["point_a"] = a_source_index
    candidate_tracklet["point_b"] = b_source_index
    candidate_tracklet["ab_dist"] = ab_dist  # don't forget this is in radians
    candidate_tracklet["dec_a"] = dec_a
    candidate_tracklet["dec_b"] = dec_b
    candidate_tracklet["ra_a"] = ra_a
    candidate_tracklet["ra_b"] = ra_b
    candidate_tracklet["mag_a"] = mag_a
    candidate_tracklet["mag_b"] = mag_b
    candidate_tracklet["observatory_code"] = observatory_code
    candidate_tracklet["band"] = band

    if len(candidate_tracklet) == 0:
        print("No pairs found.")
        continue  # skip to next frame triplet

    # I keep thinking there's a cleaner way to do this, but
    # this works so who cares.
    # We're moving the existing tracklet information into
    # a numpy matrix so we can easily add rows and ignore others
    np_tracklets = candidate_tracklet.to_numpy()
    new_tracklets = []
    point_c = []
    dec_c = []
    ra_c = []
    mag_c = []
    bc_dist = []
    prob_c = []

    time_interval2_s = (c_time - a_time).sec
    tree_c = BallTree(c_moving[["ra_rad", "dec_rad"]], leaf_size=5, metric="haversine")

    for k in range(len(candidate_tracklet)):
        # Predict location in frame c that tracklet will be.
        # This is where your search will be centered.
        temp_coord_a = SkyCoord(
            candidate_tracklet["ra_a"][k],
            candidate_tracklet["dec_a"][k],
            unit=(u.deg, u.deg),
            frame="icrs",
        )
        temp_coord_b = SkyCoord(
            candidate_tracklet["ra_b"][k],
            candidate_tracklet["dec_b"][k],
            unit=(u.deg, u.deg),
            frame="icrs",
        )
        sep = temp_coord_a.separation(temp_coord_b)
        # print("separation vs ab_dist", sep.deg, candidate_tracklet["ab_dist"][k]*(180/np.pi))
        pos_ang = temp_coord_a.position_angle(temp_coord_b)
        predict_c = temp_coord_b.directional_offset_by(
            position_angle=pos_ang, separation=sep
        )
        predict_c_np = [[predict_c.ra.radian, predict_c.dec.radian]]
        # print("predicted", predict_c.ra.deg, predict_c.dec.deg)

        # Determine search radius based on velocity and angle
        # pick the bigger of the two
        # these are both in radians
        r_due_to_angle = np.tan(np.radians(180 - min_tracklet_angle) * sep.rad)
        r_due_to_timing = (time_interval2_s + timing_uncertainty) * (
            sep.rad / time_interval_s
        )  # radians
        r_to_search_c = np.max([r_due_to_angle, r_due_to_timing])
        # print("r_to_search",r_to_search_c*(180/np.pi))

        # see if anything is around the predicted position in c
        indicies_c, distances_c = tree_c.query_radius(
            predict_c_np, r=r_to_search_c, return_distance=True
        )

        # if detection(s) are around precdicted position in c, then add to tracklet list.

        for j in range(len(indicies_c)):
            try:
                print("Dec", c_moving["Dec"][indicies_c[j][0]])
            except:
                nope = 1
            else:
                print(i, j, np_tracklets[:][j])
                new_tracklets.append(np_tracklets[:][k])
                point_c.append(indicies_c[j][0])
                dec_c.append(c_moving["Dec"][indicies_c[j][0]])
                ra_c.append(c_moving["RA"][indicies_c[j][0]])
                mag_c.append(c_moving["magnitude"][indicies_c[j][0]])
                bc_dist.append(distances_c[j][0])
        else:
            "No length 3 tracklets found in these frames."

    # Reassemble that dataframe
    complete_tracklets = pd.DataFrame(
        new_tracklets,
        columns=[
            "pair_id",
            "point_a",
            "point_b",
            "ab_dist",
            "dec_a",
            "dec_b",
            "ra_a",
            "ra_b",
            "mag_a",
            "mag_b",
            "observatory_code",
            "band",
        ],
    )
    complete_tracklets["point_c"] = point_c
    complete_tracklets["dec_c"] = dec_c
    complete_tracklets["ra_c"] = ra_c
    complete_tracklets["mag_c"] = mag_c
    complete_tracklets["bc_dist"] = bc_dist

    #print("after 3rd linkage", len(complete_tracklets))

    if len(complete_tracklets) == 0:
        print("No tracklets found.")
        continue  # skip to next frame triplet

    # # Tracklet screening
    # A slow moving tracklet has a relatively large search radious for
    # point c, meaning that in some cases the resulting tracklet might
    # have an extreme angle between a-b-c (a c is found, but behind a)
    # so do another screening.
    # this assumes an arbitrary distance to calculate angle.
    # also screens for magnitude
    ab_bc_vratio = []
    mag_array = []
    mag_min_array = []
    for i in range(len(complete_tracklets)):
        print(
            complete_tracklets.mag_a[i],
            complete_tracklets.mag_b[i],
            complete_tracklets.mag_c[i],
        )
        mag_min = np.min(
            [
                complete_tracklets.mag_a[i],
                complete_tracklets.mag_b[i],
                complete_tracklets.mag_c[i],
            ]
        )
        mag_max = np.max(
            [
                complete_tracklets.mag_a[i],
                complete_tracklets.mag_b[i],
                complete_tracklets.mag_c[i],
            ]
        )
        #print("Magnitude thresholds", mag_max, mag_min, max_mag_variance)
        if (mag_max - mag_min) < max_mag_variance:
            #print("Passed mag screening.")
            coordA = SkyCoord(
                ra=complete_tracklets.ra_a[i],
                dec=complete_tracklets.dec_a[i],
                unit=(u.deg, u.deg),
                frame="icrs",
                # distance=70 * u.kpc,
            )
            coordB = SkyCoord(
                ra=complete_tracklets.ra_b[i],
                dec=complete_tracklets.dec_b[i],
                unit=(u.deg, u.deg),
                frame="icrs",
                # distance=70 * u.kpc,
            )
            coordC = SkyCoord(
                ra=complete_tracklets.ra_c[i],
                dec=complete_tracklets.dec_c[i],
                unit=(u.deg, u.deg),
                frame="icrs",
                # distance=70 * u.kpc,
            )
            lenAB = coordA.separation(coordB).arcsecond
            lenBC = coordB.separation(coordC).arcsecond
            lenCA = coordC.separation(coordA).arcsecond

            # Metric to screen for velocity. For some objects/observing cadences
            # you expect that the velocity between A-B is about the same as B-C
            # If that is the case for your objects/observing cadence, you can use this
            # quick screen to find objects that have velocity changes between A-B and B-C
            # The result is the number of arseconds traveled; this metric was chosen
            # becuase unlike a velocity ratio, it doesn't unjustly screen out slow-moving
            # objects.
            vdiff = (lenAB / (b_time - a_time).sec) - (lenBC / (c_time - b_time).sec)
            velocity_metric = np.absolute(vdiff) * (c_time.value - a_time.value)
            # print ("velocity metric", velocity_metric, (c_time - a_time).sec, np.absolute(vdiff))
            # print ("Arcseconds moved in 30 seconds, A-B:", (lenAB /(b_time - a_time).sec)*30,"B-C:",(lenBC /(c_time - b_time).sec)*30,)
            if velocity_metric > velocity_metric_threshold:
                complete_tracklets.drop(index=[i], inplace=True)
            else:
                ab_bc_vratio.append(velocity_metric)
                mag_array.append(mag_max - mag_min)
                mag_min_array.append(
                    mag_max
                )  # because the "minimum" mag you want is the faintest one

        else:
            #print("Failed mag screening.")
            complete_tracklets.drop(index=[i], inplace=True)

    complete_tracklets.reset_index(inplace=True)
    complete_tracklets["mag_diff"] = mag_array
    complete_tracklets["mag_min"] = mag_min_array
    complete_tracklets["ab_bc_vratio"] = ab_bc_vratio
    # print("Feature-based tracklet screening of", len(complete_tracklets), "complete.")
    sys.stdout.flush()  # print out everything before running FindOrb

    # now filter based on findorb
    for i in range(len(complete_tracklets)):
        # print(complete_tracklets["point_a"][i], complete_tracklets["point_b"][i],complete_tracklets["point_c"][i])

        if tracklet_num == 0:
            tracklet_id_string = increment_identifier(starting_tracklet_id)
        else:
            tracklet_id_string = increment_identifier(tracklet_id_string)
        tracklet_id = tracklet_tag + tracklet_id_string

        coordA = SkyCoord(
            ra=complete_tracklets.ra_a[i],
            dec=complete_tracklets.dec_a[i],
            unit=(u.deg, u.deg),
            distance=70 * u.kpc,
        )
        coordB = SkyCoord(
            ra=complete_tracklets.ra_b[i],
            dec=complete_tracklets.dec_b[i],
            unit=(u.deg, u.deg),
            distance=70 * u.kpc,
        )
        coordC = SkyCoord(
            ra=complete_tracklets.ra_c[i],
            dec=complete_tracklets.dec_c[i],
            unit=(u.deg, u.deg),
            distance=70 * u.kpc,
        )
        sky_sep = coordA.separation(coordC)
        sky_sep_arcs = sky_sep.arcsecond

        formatted_data = "     "
        formatted_data += "{}".format(tracklet_id) + "  C"
        formatted_data += "{}".format(a_time.strftime("%Y %m %d")) + "."
        formatted_data += "{:1}".format(decimal_time_a) + " "
        formatted_data += (
            coordA.to_string(style="hmsdms", pad=True, sep=" ", precision=2)
            + "         "
        )
        formatted_data += "{:.1f}".format(complete_tracklets.mag_a[i]) + "   "
        formatted_data += (
            complete_tracklets.band[i]
            + "    "
            + str(complete_tracklets.observatory_code[i])
            + "\n"
        )

        formatted_data += "     "
        formatted_data += "{}".format(tracklet_id) + "  C"
        formatted_data += "{}".format(b_time.strftime("%Y %m %d")) + "."
        formatted_data += "{:1}".format(decimal_time_b) + " "
        formatted_data += (
            coordB.to_string(style="hmsdms", pad=True, sep=" ", precision=2)
            + "         "
        )
        formatted_data += "{:.1f}".format(complete_tracklets.mag_b[i]) + "   "
        formatted_data += (
            complete_tracklets.band[i]
            + "    "
            + str(complete_tracklets.observatory_code[i])
            + "\n"
        )

        formatted_data += "     "
        formatted_data += "{}".format(tracklet_id) + "  C"
        formatted_data += "{}".format(c_time.strftime("%Y %m %d")) + "."
        formatted_data += "{:1}".format(decimal_time_c) + " "
        formatted_data += (
            coordC.to_string(style="hmsdms", pad=True, sep=" ", precision=2)
            + "         "
        )
        formatted_data += "{:.1f}".format(complete_tracklets.mag_c[i]) + "   "
        formatted_data += (
            complete_tracklets.band[i]
            + "    "
            + str(complete_tracklets.observatory_code[i])
            + "\n"
        )

        trackletFound = "n"  # this will change later if you use findorb and the tracklet is less than
        # the residual
        res = "NaN"  # this is the findorb residual; will stay 'NaN' if you don't run findorb.
        if findorb_check == "y":
            findOrbTxt = open("/projectnb/ct-ast/findPOTATOs/fo.txt", "w")
            findOrbTxt.writelines(formatted_data)
            findOrbTxt.close()

            trackletFound, res = find_orb(
                Maximum_residual, nullResid=True, MOIDLim=True
            )
            # print("results of find_orb:",trackletFound,res)

        if (findorb_check == "y" and trackletFound == "y") or findorb_check == "n":
            tracklet_num += 1
            # print("Candidate tracklet found!\n", formatted_data)
            print("Candidate tracklet ", tracklet_id, " found!")

            # print("Source IDs:",complete_tracklets["point_a"][i], complete_tracklets["point_b"][i],complete_tracklets["point_c"][i])

            if exists(trackletfilename):
                with open(trackletfilename, "a", encoding="utf-8") as f:
                    f.write(formatted_data)
                    f.close
            else:
                with open(trackletfilename, "x", encoding="utf-8") as f:
                    f.write(formatted_data)
                    f.close

            if exists(tracklet_features):
                with open(tracklet_features, "a", encoding="utf-8") as f:
                    f.write(
                        tracklet_id
                        + ","
                        + str(complete_tracklets.mag_diff[i])
                        + ","
                        + str(complete_tracklets.mag_min[i])
                        + ","
                        + str(res)
                        + ","
                        + str(sky_sep_arcs)
                        + ","
                        + str(complete_tracklets.ab_bc_vratio[i])
                        + "\n"
                    )
                    f.close
            else:
                with open(tracklet_features, "x", encoding="utf-8") as f:
                    f.write(
                        "tracklet_id,mag_diff,mag_min,residual,sky_sep,ab_bc_vratio\n"
                    )
                    f.write(
                        tracklet_id
                        + ","
                        + str(complete_tracklets.mag_diff[i])
                        + ","
                        + str(complete_tracklets.mag_min[i])
                        + ","
                        + str(res)
                        + ","
                        + str(sky_sep_arcs)
                        + ","
                        + str(complete_tracklets.ab_bc_vratio[i])
                        + "\n"
                    )
                    f.close

            if save_tracklet_images == "y":
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                title_str = (
                    "Tracklet:"
                    + tracklet_id
                    + " Velocity metric:"
                    + str(np.round(complete_tracklets.ab_bc_vratio[i], 4))
                )

                fig.suptitle(title_str, fontsize=16)

                # Plot magnitude
                mag_list_y = [
                    complete_tracklets.mag_a[i],
                    complete_tracklets.mag_b[i],
                    complete_tracklets.mag_c[i],
                ]
                mag_list_x = [1, 2, 3]
                axs[0].scatter(mag_list_x, mag_list_y, c="purple")
                axs[0].set_title("Lightcurve")
                axs[0].set_xlabel("Gridlines are 0.5mag", fontsize=8)
                axs[0].set_ylabel("Magnitudes, mag")
                axs[0].grid(
                    True, linestyle="--", which="both", color="gray", linewidth=0.5
                )
                axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                axs[0].set_xticks(mag_list_x)
                axs[0].xaxis.grid(False)
                axs[0].set_xticklabels(["A", "B", "C"])

                # Plot tracklet in RA/DEC
                ra_list = [
                    complete_tracklets.ra_a[i],
                    complete_tracklets.ra_b[i],
                    complete_tracklets.ra_c[i],
                ]
                dec_list = [
                    complete_tracklets.dec_a[i],
                    complete_tracklets.dec_b[i],
                    complete_tracklets.dec_c[i],
                ]

                tracklet_coord = SkyCoord(ra=ra_list, dec=dec_list, unit="deg")
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

                if show_sky_image == "y":
                    sky_image, wcs = query_skyview(
                        ra_list[1],
                        dec_list[1],
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
                        pix_x, pix_y = tracklet_coord.to_pixel(wcs)
                        axs[2].imshow(
                            sky_image.data,
                            cmap="gray",
                            origin="lower",
                            vmin=vmin,
                            vmax=vmax,
                        )
                        axs[2].scatter(
                            pix_x,
                            pix_y,
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

                # plt.show()
                # Save figure
                plt.tight_layout()
                plt.savefig(
                    "output/"
                    + data_id
                    + "/tracklet_"
                    + data_id
                    + "_"
                    + tracklet_id
                    + ".png"
                )
                plt.close("all")
                # plt.show()

            if export_ades == "y":
                # update keys in obsData dictionary for all three points
                # Coordinate A
                ades_obs_dict["trkSub"] = tracklet_id
                ades_obs_dict["obsTime"] = str(a_time.isot) + "Z"
                ades_a_ra = coordA.ra.deg
                ades_a_dec = coordA.dec.deg
                ades_obs_dict["ra"] = np.round(ades_a_ra, 5)
                ades_obs_dict["dec"] = np.round(ades_a_dec, 5)
                ades_obs_dict["mag"] = "{:.1f}".format(complete_tracklets.mag_a[i])

                if xml_tracklet_found == "n":  # first tracklet found, write header
                    xml_tracklet_found = "y"
                    XMLElement, ades_result, obsData = generate_xml(
                        xml_filename, ades_dict, ades_obs_dict
                    )

                else:  # update the existing xml
                    XMLElement, ades_result, obsData = update_xml(
                        XMLElement, ades_result, obsData, ades_obs_dict
                    )

                # Coordinate B
                ades_obs_dict["obsTime"] = str(b_time.isot) + "Z"
                ades_b_ra = coordB.ra.deg
                ades_b_dec = coordB.dec.deg
                ades_obs_dict["ra"] = np.round(ades_b_ra, 5)
                ades_obs_dict["dec"] = np.round(ades_b_dec, 5)
                ades_obs_dict["mag"] = "{:.1f}".format(complete_tracklets.mag_b[i])
                XMLElement, ades_result, obsData = update_xml(
                    XMLElement, ades_result, obsData, ades_obs_dict
                )

                # Coordinate C
                ades_obs_dict["obsTime"] = str(c_time.isot) + "Z"
                ades_c_ra = coordC.ra.deg
                ades_c_dec = coordC.dec.deg
                ades_obs_dict["ra"] = np.round(ades_c_ra, 5)
                ades_obs_dict["dec"] = np.round(ades_c_dec, 5)
                ades_obs_dict["mag"] = "{:.1f}".format(complete_tracklets.mag_c[i])
                XMLElement, ades_result, obsData = update_xml(
                    XMLElement, ades_result, obsData, ades_obs_dict
                )

        if findorb_check == "y" and trackletFound == "n":  # drop it
            print("tracklet rejected")
            complete_tracklets.drop(index=[i], inplace=True)

    # plot all
    plt.title("Sources (yellow,green,blue) and identified tracklets (red)")
    plt.scatter(a["RA"], a["Dec"], color="gold", alpha=0.6, label="Sources in A")
    plt.scatter(b["RA"], b["Dec"], color="limegreen", alpha=0.6, label="Sources in B")
    plt.scatter(c["RA"], c["Dec"], color="deepskyblue", alpha=0.6, label="Sources in C")
    plt.scatter(
        complete_tracklets.ra_a,
        complete_tracklets.dec_a,
        marker="o",
        facecolors="none",
        edgecolors="red",
    )
    plt.scatter(
        complete_tracklets.ra_b,
        complete_tracklets.dec_b,
        marker="o",
        facecolors="none",
        edgecolors="red",
    )
    plt.scatter(
        complete_tracklets.ra_c,
        complete_tracklets.dec_c,
        marker="o",
        facecolors="none",
        edgecolors="red",
    )
    plt.xlabel("RA, deg")
    plt.ylabel("Dec, deg")
    plt.savefig("output/" + data_id + "/sources_and_tracklets_" + str(m) + ".png")
    plt.show()
    plt.close()

    # save stats
    now = datetime.now()
    yearmonthday = now.strftime("%Y%m%d")
    outputname = "output/" + data_id + "/o_linking_log" + data_id + ".csv"
    run_time = str(now - start_time)
    num_sources = str(len(a) + len(b) + len(c))
    num_tracklets_prescreen = str(len(complete_tracklets))

    if not exists(outputname):
        f = open(outputname, "x", encoding="utf-8")
        f.write(
            "filea,fileb,filec,date_corrected,time_corrected,run_time_s,num_sources,num_tracklets_prescreen,export_ades,max_speed,velocity_metric_threshold,min_tracklet_angle,timing_uncertainty,max_mag_variance,Maximum_residual,min_dist_deg,findorb_check,stationary_dist_deg\n"
        )
    else:
        f = open(outputname, "a", encoding="utf-8")
    f.write(
        order_frames.names[0]
        + ","
        + order_frames.names[1]
        + ","
        + order_frames.names[2]
        + ","
        + datetime.today().strftime("%Y-%m-%d")
        + ","
        + datetime.today().strftime("%H:%M:%S")
        + ","
        + run_time
        + ","
        + num_sources
        + ","
        + num_tracklets_prescreen
        + ","
        + export_ades
        + ","
        + str(max_speed)
        + ","
        + str(velocity_metric_threshold)
        + ","
        + str(min_tracklet_angle)
        + ","
        + str(timing_uncertainty)
        + ","
        + str(max_mag_variance)
        + ","
        + str(Maximum_residual)
        + ","
        + str(min_dist_deg)
        + ","
        + str(findorb_check)
        + ","
        + str(stationary_dist_deg)
        + "\n"
    )
    f.close

if export_ades == "y":
    # write the ADES xml to file
    tree = XMLElement.ElementTree(ades_result)
    xml_string = minidom.parseString(XMLElement.tostring(ades_result)).toprettyxml()
    with open(xml_filename, "w", encoding="UTF-8") as files:
        files.write(xml_string)
    print("Successfully ouput ADES file.")
