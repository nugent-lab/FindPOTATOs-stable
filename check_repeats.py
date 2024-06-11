import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


def find_repeated_values_ra_dec(xml_file):
    """
    Check if multiple tracklets use the same points.

    Args:
        xml_file: xml file 
        feild: feild to look for repeats. ra works well.
    Returns:
        df: dataframe with repeats
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    ra_values_found = []
    dec_values_found = []
    data = []
    for top_level_element in root:
        for obsData in top_level_element.findall('obsData'):
            for optical in obsData.findall('optical'):
                ra_value = optical.find('ra').text
                trkSub_value = optical.find('trkSub').text
                dec_value= optical.find('dec').text
                data.append({'ra': ra_value, 'trkSub_value': trkSub_value})
                if (ra_value in ra_values_found) and (dec_value in dec_values_found):
                    message = f"Point repeated in tracklet. IDs: {ra_value}, {dec_value} trkSub: {trkSub_value}"
                    print(message)
                else:
                    ra_values_found.append(ra_value)
                    dec_values_found.append(dec_value)
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


def process_by_ra_value(df, night):
    # Count duplicates for each ra_value
    df_duplicates = df.groupby('ra').filter(lambda x: len(x) > 1)

    # Sort ra_values by number of duplicates in descending order
    ra_value_counts = df_duplicates['ra'].value_counts()
    unique_ra_values = df_duplicates['ra'].unique()


    # Lists to store selected and rejected DataFrames
    selected_dfs = []
    rejected_dfs = []

    # # Loop through unique values and produce separate DataFrames
    for ra_value in unique_ra_values:
        df_subset = df[df['ra'] == ra_value]
        #print(f"\nDataFrame for ra_value = {ra_value}:")
        #print(df_subset)

        selected, rejected = display_images_and_ask(df_subset, night)
        # Append selected and rejected DataFrames to lists
        selected_dfs.append(selected)
        rejected_dfs.append(rejected)

    # Concatenate all selected and rejected DataFrames
    df_selected_final = pd.concat(selected_dfs, ignore_index=True)
    df_rejected_final = pd.concat(rejected_dfs, ignore_index=True)

    return df_selected_final, df_rejected_final


# call python check_repeats.py nightnumber
night=sys.argv[1]
inputname= 'output/'+night+'/tracklets_ADES_'+ night +'.xml'
output_unique= 'output/'+night+'/tracklets_ADES_'+ night +'_unique.xml'

#df=find_repeated_values(inputname,'ra')
df=find_repeated_values_ra_dec(inputname)
df_selected_final, df_rejected_final = process_by_ra_value(df, night)

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
    #print("opitcal_to_remove", optical_to_remove)
    # Remove marked optical elements
    for optical in optical_to_remove:
        #print("trying", optical)
        obs_data.remove(optical)

print("Writing to file.")
xml_string = minidom.parseString(ET.tostring(root)).toprettyxml()
xml_string = os.linesep.join([s for s in xml_string.splitlines() if s.strip()])
with open(output_unique, "w", encoding="UTF-8") as files:
    files.write(xml_string)
print("Duplicate check complete.\n:)")