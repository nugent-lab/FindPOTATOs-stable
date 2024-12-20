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
import sys
from linking_library import *
from ades import *
from parameters import *

# call:
# python findPOTATOs.py [data_id]
#  C.R. Nugent, N. Tan, S. Matsumoto
# Version 2.0, can create linkages of length > 3

data_id=str(sys.argv[1])

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
print("Min search radius:", min_search_radius)
print("Maximum residual:", maximum_residual)
print("Min distance:", min_dist_deg)
print("Findorb check:", findorb_check)
print("Tracklet tag:", tracklet_tag)

if export_ades:
    print("******************************")
    print(
        "You've chosen to export to ADES format. Please review these ADES parameters for accuracy. If anything is incorrect, quit with ctrl-D and update the values in parameters.py."
    )
    print_dict(ades_dict)
    blank = input("Press any key to continue.")
    print(
        "Please review these observation-related ADES parameters for accuracy. Please refrence the ADES documentation if you have questions. If anything is incorrect, quit with ctrl-D and update the values in parameters.py."
    )
    print("mode:", ades_obs_dict["mode"])
    print("rmsRA:", ades_obs_dict["rmsRA"])
    print("rmsDec:", ades_obs_dict["rmsDec"])
    print("rmsMag:", ades_obs_dict["rmsMag"])
    print("band:", ades_obs_dict["band"])
    print("photCat:", ades_obs_dict["photCat"])
    blank = input("Press any key to continue.")


input_filename = "image_groups_" + str(data_id) + ".csv"
image_series_list = pd.read_csv(input_filename)
tracklet_num = 0

os.makedirs("output/" + data_id, exist_ok=True)
# Clean up any output files from a previous run
trackletfilename = "output/" + data_id + "/tracklets_" + data_id + ".txt"
xml_filename = "output/" + data_id + "/tracklets_ADES_" + data_id + ".xml"
remove_file_if_exists(trackletfilename)
remove_file_if_exists(xml_filename)

if export_ades:
    xml_tracklet_found = False


tracklet_count=0
tracklet_num = 0 
for m in np.arange(len(image_series_list)):
    stop = False
    start_time = datetime.now() # so you can record running time
    # make dataframe of each set
    # sort by time
    file_list=image_series_list.loc[m].tolist() 
    names=[]
    times=[]
    decimal_times=[]
    for n in np.arange(len(file_list)): 
        init_filename=image_series_list.iloc[m,n]
        names.append(init_filename)
        # read in dataframe to get time
        try:
            inital_df=pd.read_csv(input_directory + init_filename)
            frame_time=Time(inital_df.mjd[0].astype(float), format="mjd", scale="utc")
            times.append(frame_time)
            decimal_time=str(frame_time).split(".")
            decimal_time = decimal_time[1][:5]
            decimal_time = zero_pad(decimal_time)
            decimal_times.append(decimal_time)
        except:
            print("Files not found.")
            stop = True
    if stop:
        continue
    order_frames = pd.DataFrame(
        {
            "name": names,
            "time": times,
            "decimal_time": decimal_times,
        }
    )
    order_frames.sort_values(by=["time"], inplace=True, ignore_index= True)
    order_frames.reset_index(inplace=True, drop=True)

    # creates dictionary of dataframes, df0, df1, etc
    source_dataframes = {f'df{n}': pd.read_csv(input_directory+order_frames['name'].iloc[n]) for n in range(len(order_frames))}
    print(
        f"Checking file group number {m}, consisting of: "
        + ", ".join(order_frames.name)
    )

    # Screen on ml_probs
    for name, df in source_dataframes.items():
        if 'ml_probs' not in df.columns:
            # Add 'ml_prob' column with NaN values
            df['ml_probs'] = np.nan
        else: 
            old_size=len(df)
            source_dataframes[name] = df[df['ml_probs'] >= ml_thresh]
            source_dataframes[name].reset_index(inplace=True, drop=True)
            print("Removed detections with a ml-assigned probability of being point sources less than:", ml_thresh, "in", name)
            print("Fraction",np.round(len(source_dataframes[name])/old_size,2),"remaining.")


    # Remove stationary sources
    source_dataframes_moving = remove_stationary_sources(
        source_dataframes, stationary_dist_deg, showplots=False
    )
    # if any frames are empty after ML and/or stationary source screening, then no tracklets will
    # be found here. Skip to next frame group
    stop=False
    for key, df in source_dataframes_moving.items():
        if df.empty:
            print("One of the frames has no sutiable sources. Skipping to next iteration.")
            print("Empty frame is ", key)
            stop = True
    if stop:
        continue

    # After removing stationary sources, compare first and second
    # source dataframes together. This first linkage is treated differently
    # than subsequent linkages.

    # Minimum speed an asteroid can travel to be detected, in arcseconds/second
    # this is calculated based on our astromemtric accuracy and time between frames
    time_interval_s = (order_frames.time[1] - order_frames.time[0]).sec
    max_dist_arcsec=max_speed * time_interval_s
    max_dist_rad = np.radians(max_dist_arcsec/3600)
    min_dist_rad = np.radians(min_dist_deg / 3600)
    df0= source_dataframes_moving['df0_moving'].copy(deep=True)
    df1= source_dataframes_moving['df1_moving'].copy(deep=True)
    tree_a = BallTree(df0[["RA", "Dec"]], leaf_size=5, metric='pyfunc', func=angular_separation_metric)
    indicies_b, distances_b = tree_a.query_radius(
        df1[["RA", "Dec"]], r=max_dist_arcsec, return_distance=True
    )
    tracklets = {} # this will be dictionary of dataframes, each df is a tracklet
    # We want every source in a that is within a certain radius of each source in b, but not too slow.
    for i in range(len(indicies_b)):
        coordB= SkyCoord(
            ra=df1["RA"][i],
            dec=df1["Dec"][i],
            unit=(u.deg, u.deg),
            frame="icrs",
        ) 
        for j in range(len(indicies_b[i])):
            coordA= SkyCoord(
                ra=df0["RA"][indicies_b[i][j]],
                dec=df0["Dec"][indicies_b[i][j]],
                unit=(u.deg, u.deg),
                frame="icrs",
            )   
            if distances_b[i][j] > min_dist_deg / 3600: 
                time_delta=(order_frames.time[1] - order_frames.time[0]).sec
                vdiff = (distances_b[i][j] / time_delta)
                row_data = {
                    "ra":[df0["RA"][indicies_b[i][j]], df1["RA"][i]],
                    "dec":[df0["Dec"][indicies_b[i][j]],df1["Dec"][i]],
                    "mag":[df0["magnitude"][indicies_b[i][j]], df1["magnitude"][i]],
                    "observatory_code":[df0["observatory_code"][indicies_b[i][j]], df1["observatory_code"][i]],
                    "band":[df0["band"][indicies_b[i][j]], df1["band"][i]],
                    "ml_probs":[df0["ml_probs"][indicies_b[i][j]], df1["ml_probs"][i]],
                    "ang_sep_arcsec":[0,distances_b[i][j]],
                    "coords" :[coordA,coordB],
                    "vdiff" : [np.nan, vdiff],
                    "tracklet_time" : [order_frames.time[0], order_frames.time[1]],
                    "decimal_time" : [order_frames.decimal_time[0], order_frames.decimal_time[1]],
                }
                if export_ades:
                    ades_data ={
                        "mag_err" : [df0["mag_err"][indicies_b[i][j]], df1["mag_err"][i]],
                        "RA_err"  : [df0["RA_err"][indicies_b[i][j]], df1["RA_err"][i]],
                        "Dec_err"  : [df0["Dec_err"][indicies_b[i][j]], df1["Dec_err"][i]],
                    }
                    row_data.update(ades_data)
                temp_tracklet = pd.DataFrame(row_data)
                tracklets[f"df_{tracklet_count}"] = temp_tracklet
                tracklet_count += 1

    if tracklet_count == 0:
        print("No pairs found.")
        stop = True
        continue  # skip to next frame

    # Seek future links
    number_links=2 # you are starting with two detections linked together 
    while number_links < len(order_frames): 
        new_tracklets = []
        print("Searching for links with", number_links+1,"detections.")
        link_index=number_links-1
        # time between this frame and the previous one 
        old_time_interval_s = (order_frames.time[number_links-1] - order_frames.time[number_links-2]).sec
        # time between this frame and the next one 
        next_time_interval_s= (order_frames.time[number_links] - order_frames.time[number_links-1]).sec
        # make tree of place you're looking
        next_sources='df'+str(link_index+1)+'_moving'
        next_sources_df=source_dataframes_moving[next_sources]
        tree_next = BallTree(next_sources_df[["RA", "Dec"]], leaf_size=5,  metric='pyfunc', func=angular_separation_metric)
        for key, tracklet_df in tracklets.items(): #iterate over tracklet set
            if len(tracklet_df) == number_links:
                # Predict location in frame that next source, if it exists, will be.
                # This is where your search will be centered.
                temp_coord_a = SkyCoord(
                    tracklet_df["ra"][number_links-2],
                    tracklet_df["dec"][number_links-2],
                    unit=(u.deg, u.deg),
                    frame="icrs",
                )
                temp_coord_b = SkyCoord(
                    tracklet_df["ra"][number_links-1],
                    tracklet_df["dec"][number_links-1],
                    unit=(u.deg, u.deg),
                    frame="icrs",
                )
                sep_prev = temp_coord_a.separation(temp_coord_b)
                pos_ang = temp_coord_a.position_angle(temp_coord_b)
                time_ratio=next_time_interval_s/old_time_interval_s
                c_ra, c_dec = temp_coord_a.spherical_offsets_to(temp_coord_b)
                predict_c = temp_coord_b.spherical_offsets_by(c_ra*time_ratio, c_dec*time_ratio)
                predict_c_np = [[predict_c.ra.deg, predict_c.dec.deg]]
                predict_sep=temp_coord_b.separation(predict_c)

                # Determine search radius based on time uncertantiy & velocity, or angle
                # pick the bigger of the two
                # these are both in radians
                added_angle=(pos_ang+(180 * u.deg -min_tracklet_angle))
                predict_c_high_angle = temp_coord_b.directional_offset_by(
                    position_angle=added_angle, separation=sep_prev*time_ratio
                )
                #calculate separation between high_angle and predicted c
                predict_angle_sep=predict_c.separation(predict_c_high_angle)

                r_due_to_angle = predict_angle_sep.arcsecond
                r_to_search_c = np.max([r_due_to_angle, min_search_radius.value])

                # see if anything is around the predicted position in c
                indicies_c, distances_c = tree_next.query_radius(
                    predict_c_np, r=r_to_search_c, return_distance=True
                )

                # if detection(s) are around predicted position in c, then add to tracklet list.
                check_empty=indicies_c[0].shape
                #print("Linking indicies", indicies_c, check_empty[0])
                for j in range(check_empty[0]):
                    if check_empty[0] == 0:
                        nope = 1
                    else:
                        found_index=indicies_c[0][j]
                        new_coord = SkyCoord(
                            ra=next_sources_df["RA"][found_index],
                            dec=next_sources_df["Dec"][found_index],
                            unit=(u.deg, u.deg),
                            frame="icrs",
                        )  
                        #print("angular", (temp_coord_b.separation(new_coord).arcsecond))
                        new_sep_dist = temp_coord_b.separation(new_coord).arcsecond
                        vdiff = new_sep_dist / next_time_interval_s
                        current_mag=next_sources_df["magnitude"][found_index]

                        new_link = {
                            "ra" : next_sources_df["RA"][found_index],
                            "dec" : next_sources_df["Dec"][found_index],
                            "mag" : current_mag,
                            "observatory_code" : next_sources_df["observatory_code"][found_index],
                            "band" : next_sources_df["band"][found_index],
                            "ml_probs" : next_sources_df["ml_probs"][found_index],
                            "ang_sep_arcsec" : new_sep_dist,
                            "coords" :new_coord,
                            "vdiff" : vdiff,
                            "tracklet_time" : order_frames.time[number_links],
                            "decimal_time" : order_frames.decimal_time[number_links],
                            }
                        if export_ades:
                            ades_data ={
                                "mag_err" : next_sources_df["mag_err"][found_index],
                                "RA_err"  : next_sources_df["RA_err"][found_index],
                                "Dec_err"  : next_sources_df["Dec_err"][found_index],
                            }
                            new_link.update(ades_data)
                        if j == 0:
                            #print("one possible linkage.", j)
                            tracklet_df.loc[number_links]  = new_link
                            tracklet_df_orig=tracklet_df.copy(deep = True)
                        else:
                            #print("multiple possible linkages.")
                            new_df = tracklet_df_orig.copy(deep = True)
                            new_df.loc[number_links]  = new_link
                            new_key_str= key +str(j)
                            new_tracklets.append((new_key_str, new_df))  # Append new tracklet data
                            
        for new_key, new_df in new_tracklets:
            tracklets[new_key] = new_df
        number_links += 1

    # trim out any dataframes with less than 3 rows
    keys_to_delete = [key for key, df in tracklets.items() if len(df) < 3]
    for key in keys_to_delete:
        del tracklets[key]

    if not tracklets:
        print("No tracklets found.")
        continue  # skip to next frame group

    # Tracklet screening
    sys.stdout.flush()  # print out everything before running FindOrb
    keys_to_delete = []
    for key,df in tracklets.items():
        tracklet_found = False 
        mag_min=df["mag"].min()
        mag_max=df["mag"].max()
        if (mag_max - mag_min) < max_mag_variance:
            #print("Passed mag screening with variance", np.round(mag_max - mag_min,3), "mag")       
            velocity_metric=abs(df["vdiff"].std())
            if velocity_metric < velocity_metric_threshold:
                #print("Passed velocity metric screening:", np.round(velocity_metric,6))
                if tracklet_num == 0:
                    tracklet_id_string = increment_identifier(starting_tracklet_id)
                else:
                    tracklet_id_string = increment_identifier(tracklet_id_string)
                tracklet_id = tracklet_tag + tracklet_id_string
                df['tracklet_id'] = tracklet_id
                tracklet_num += 1
                formatted_data=format_data(df)
                if findorb_check:
                    findOrbTxt = open("fo.txt", "w")
                    findOrbTxt.writelines(formatted_data)
                    findOrbTxt.close()
                    tracklet_found, res = find_orb(
                        maximum_residual, nullResid=True, MOIDLim=True
                    )
                    print("results of find_orb:",tracklet_found,res)
                    df['fo_res'] = res
                    if tracklet_found:
                        print("Candidate tracklet ", tracklet_id, " found!")
                else:
                    df['fo_res'] = np.nan
                    Maximum_residual=np.nan
                    tracklet_found = True
                    print("Candidate tracklet ", tracklet_id, " found!")   
        #     else:
        #         print("Failed velocity metric screening:", np.round(velocity_metric,6)) 
        # else:
        #     print("Failed mag screening with variance", np.round(mag_max - mag_min,3), "mag")       
            
        if tracklet_found: # Export tracklet information.
            print(formatted_data)
            # 80-char MPC data
            if exists(trackletfilename):
                with open(trackletfilename, "a", encoding="utf-8") as f:
                    f.write(formatted_data)
                    f.close
            else:
                with open(trackletfilename, "x", encoding="utf-8") as f:
                    f.write(formatted_data)
                    f.close
            
            if save_tracklet_images:
                figname=  "output/"+ data_id+ "/tracklet_"+ data_id+ "_"+ tracklet_id+ ".png"
                create_diagnostic_figure(df, figname, show_sky_image)
            
            if export_ades:
                # update keys in obsData dictionary for all points
                for index, row in df.iterrows(): 
                    coord=row["coords"]
                    ades_ra = coord.ra.deg
                    ades_dec = coord.dec.deg
                    ades_obs_dict["trkSub"] = row["tracklet_id"]
                    ades_obs_dict["obsTime"] = str(row["tracklet_time"].isot) + "Z"
                    ades_obs_dict["ra"] = np.round(ades_ra, 5)
                    ades_obs_dict["dec"] = np.round(ades_dec, 5)
                    ades_obs_dict["mag"] = "{:.1f}".format(row["mag"])
                    ades_obs_dict["rmsMag"] = row["mag_err"]
                    ades_obs_dict["rmsRA"] = row["RA_err"]
                    ades_obs_dict["rmsDec"] = row["Dec_err"]
                    if not xml_tracklet_found:  # first tracklet found, write header
                        xml_tracklet_found = True
                        XMLElement, ades_result, obsData = generate_xml(
                            ades_dict, ades_obs_dict
                        )
                    else:  # update the existing xml
                        XMLElement, ades_result, obsData = update_xml(
                            XMLElement, ades_result, obsData, ades_obs_dict
                        )
        else:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del tracklets[key]

    #diagnostic plot
    show_plot = True #you can stop plot form desplaying by making this False
    if show_plot:
        plt.title("Sources (yellow,green,blue) and identified tracklets (red)")
        plt.xlabel("Right ascension (degrees)")
        plt.ylabel("Declination (degrees)")
        for name, df in source_dataframes.items():
            plt.scatter(df.RA, df.Dec, alpha=0.4, label=name)
        for key,df2 in tracklets.items():
            plt.scatter( df2.ra, df2.dec, marker="o",
            facecolors="none",
            edgecolors="red",
        )
        plt.legend()
        plt.show()
        plt.close()
    
    # save stats
    now = datetime.now()
    yearmonthday = now.strftime("%Y%m%d")
    outputname = "output/" + data_id + "/o_linking_log" + data_id + ".csv"
    run_time = str(np.round((now - start_time).total_seconds(),5))
    print("run time",run_time,"seconds.")
    num_sources = str(len(tracklets))

    header = "filea,date_corrected,run_time_s,num_sources,export_ades,max_speed,velocity_metric_threshold,min_tracklet_angle,min_search_radius,max_mag_variance,maximum_residual,min_dist_deg,findorb_check,stationary_dist_deg\n"
    if not exists(outputname):
        mode = "x"
    else:
        mode = "a"

    with open(outputname, mode, encoding="utf-8") as f:
        if mode == "x":
            f.write(header)
            
        stat_string = (
            f"{order_frames.name[0]},{datetime.today().strftime('%Y-%m-%d')},{run_time},"
            f"{num_sources},{export_ades},{max_speed},{velocity_metric_threshold},{min_tracklet_angle},"
            f"{min_search_radius},{max_mag_variance},{maximum_residual},{min_dist_deg},"
            f"{findorb_check},{stationary_dist_deg}\n"
        )
        f.write(stat_string)
        f.close()
if not stop:
    if export_ades:
        # write the ADES xml to file
        tree = XMLElement.ElementTree(ades_result)
        xml_string = minidom.parseString(XMLElement.tostring(ades_result)).toprettyxml()
        with open(xml_filename, "w", encoding="UTF-8") as files:
            files.write(xml_string)
        print("Successfully ouput ADES file.")