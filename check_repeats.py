import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


def find_repeated_values(xml_file, field):
    """
    Check if multiple tracklets use the same points.

    Args:
        xml_file: xml file 
        feild: feild to look for repeats. ra works well.
    Returns:
        df: dataframe with values
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    values_found = []
    data = []
    for top_level_element in root:
        for obsData in top_level_element.findall('obsData'):
            for optical in obsData.findall('optical'):
                item_value = optical.find(field).text
                trkSub_value = optical.find('trkSub').text
                data.append({field: item_value, 'trkSub_value': trkSub_value})
                if item_value in values_found:
                    message = f"Point repeated in tracklet. IDs: {item_value}, trkSub: {trkSub_value}"
                    print(message)
                else:
                    values_found.append(item_value)
    df = pd.DataFrame(data)
    return df



import xml.etree.ElementTree as ET
import pandas as pd

def get_xml_data(xml_file):
    """
    Args:
        xml_file: Path to the XML file.
    
    Returns:
        df: DataFrame with ra, dec, and tracklet name
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # ra_dec_pairs_found = set()
    data = []
    
    for top_level_element in root:
        for obsData in top_level_element.findall('obsData'):
            for optical in obsData.findall('optical'):
                ra_value = optical.find('ra').text
                dec_value = optical.find('dec').text
                trkSub_value = optical.find('trkSub').text
                data.append({'ra': ra_value, 'dec': dec_value, 'trkSub_value': trkSub_value})
                #print("reading data",ra_value, dec_value, trkSub_value)
                # # Check if the pair has been seen before
                # if ra_dec_pair in ra_dec_pairs_found:
                #     message = f"Point repeated in tracklet. IDs: {ra_value}, {dec_value}, trkSub: {trkSub_value}"
                #     print(message)
                # else:
                #     ra_dec_pairs_found.add(ra_dec_pair)
    
    df = pd.DataFrame(data)
    return df



def display_images_and_ask(df, night):
    num_images = len(df)
    num_cols = 1  
    num_rows = (num_images + num_cols - 1) // num_cols  

    # Calculate figure size based on number of images, with a maximum size of 15 inches
    max_fig_width = 15
    max_fig_height = 15
    fig_width = min(max_fig_width, 5 * num_cols)  
    fig_height = min(max_fig_height, 4 * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    selected_indices = []
    rejected_indices = list(range(num_images))
    for idx, (_, row) in enumerate(df.iterrows()):
        # Calculate subplot index
        row_idx = idx // num_cols
        col_idx = idx % num_cols

        # Display 
        trkSub_value = row['trkSub_value']
        image_filename = f"output/{night}/tracklet_{night}_{trkSub_value}.png"
        print(f"Candidate {idx}: {trkSub_value}")

        if os.path.exists(image_filename):
            img = plt.imread(image_filename)
            axes[row_idx].imshow(img)
            axes[row_idx].set_title(f"Image {idx}: {image_filename}")
            axes[row_idx].axis('off')

    # Remove empty subplots
    for i in range(num_images, num_rows * num_cols):
        axes.flatten()[i].remove()

    # Ask user for input
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    print("Select images to keep (Enter the numbers separated by spaces):")
    plt.show(block=False)

    user_input = input().strip().split()
    plt.close()
    try:
        keep_indices = list(map(int, user_input))
    except ValueError:
        keep_indices = []

    all_indices = list(range(num_images))
    rejected_indices = list(set(all_indices) - set(keep_indices))
    df_selected = df.iloc[keep_indices]
    df_rejected = df.iloc[rejected_indices]

    # Print the trkSub values of rejected images
    rejected_trkSub_values = list(df_rejected['trkSub_value'])
    print(f"Rejected trkSub values: {rejected_trkSub_values}")

    return df_selected, df_rejected


def isolate_duplicates(df, night):
    df_duplicates =  df.groupby(['ra', 'dec']).filter(lambda x: len(x) > 1)

    # Lists to store selected and rejected DataFrames
    selected_dfs = []
    rejected_dfs = []
    rejected_names=[]

    # # Loop through unique values and produce separate DataFrames
    #for ra_value in unique_ra_values:
    for (ra_value, dec_value), group in df_duplicates.groupby(['ra', 'dec']):
        df_subset = df[df['ra'] == ra_value]
        df_subset = df_subset[df_subset['dec'] == dec_value]

        # check if you've rejected n-1 of the candidates
        # where n is the number of candidates
        num_tracklets=len(df_subset)    
        reject_count=0
        for trackid in df_subset['trkSub_value']:
            if trackid in rejected_names:
                reject_count += 1
            else:
                df_selected_temp = df_subset[df_subset['trkSub_value'] == trackid]
        if (reject_count+1) == num_tracklets:

            selected_dfs.append(df_selected_temp)
        elif reject_count == num_tracklets:
            print("Previously rejected all, moving on", df_subset['trkSub_value'])
        else: # ask about it 
            selected, rejected = display_images_and_ask(df_subset, night)
            # Append selected and rejected DataFrames to lists
            selected_dfs.append(selected)
            rejected_dfs.append(rejected)
        
            #keep track of the rejected names 
            rejected_names.extend(rejected['trkSub_value'].tolist())

    # Concatenate all selected and rejected DataFrames
    df_selected_final = pd.concat(selected_dfs, ignore_index=True)
    df_rejected_final = pd.concat(rejected_dfs, ignore_index=True)

    return df_selected_final, df_rejected_final


# call python check_repeats.py nightnumber
night=sys.argv[1]
inputname= 'output/'+night+'/tracklets_ADES_'+ night +'.xml'
output_unique= 'output/'+night+'/tracklets_ADES_'+ night +'_unique.xml'

#df=find_repeated_values(inputname,'ra')
df=get_xml_data(inputname)
df_selected_final, df_rejected_final = isolate_duplicates(df, night)

print("\nFinal Selected DataFrames:")
print(df_selected_final)

print("\nFinal Rejected DataFrames:")
print(df_rejected_final)

tree = ET.parse(inputname)
root = tree.getroot()
obsBlock = root.find('.//obsBlock')
obs_data = obsBlock.find("obsData")  # Replace 'parent_element' with the actual parent element name

if len(df_rejected_final) > 0:
    optical_elements = obs_data.findall('.//optical')
    trkSub_to_delete = df_rejected_final['trkSub_value'].tolist()
    optical_to_remove = []

    for optical in optical_elements:
        trkSub_element = optical.find('trkSub')
        if trkSub_element is not None and trkSub_element.text in trkSub_to_delete:
            optical_to_remove.append(optical)

    # Remove marked optical elements
    for optical in optical_to_remove:
        #print("trying", optical)
        obs_data.remove(optical)
    else:
        print("No repeats.")

print("Writing to file.")
xml_string = minidom.parseString(ET.tostring(root)).toprettyxml()
xml_string = os.linesep.join([s for s in xml_string.splitlines() if s.strip()])
with open(output_unique, "w", encoding="UTF-8") as files:
    files.write(xml_string)
print("Duplicate check complete.\n:)")