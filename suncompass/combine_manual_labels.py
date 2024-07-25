
import pandas as pd
import click
import json

import numpy as np

from add_sun_vector import calculate_sun_vector_relative

from extract_motion import extract_motion


def load_manual_labels(csv_file: str) -> pd.DataFrame:
    # Load the manual labels from CSV
    """
    "Image","Latitude","Longitude"
    "images/5580.jpg","53.31874897116222","-6.31506621382057"
    "images/5640.jpg","53.318889389379535","-6.315044640663809"
    "images/5700.jpg","53.319048812606425","-6.315076156619734"
    "images/5760.jpg","53.31921127410351","-6.315141200188346"
    "images/5820.jpg","53.31937344666278","-6.315257743097269"
    "images/5880.jpg","53.31954879919812","-6.315375435101591"
    """
    manual_labels = pd.read_csv(csv_file)
    return manual_labels

def load_frames_meta_data(json_file: str) -> pd.DataFrame:
    # Load the meta data from the JSON file
    # Open the json file
    with open(json_file, 'r') as file:
        data = json.load(file)
    # Convert the data to a DataFrame
    meta_data = pd.DataFrame(data['frames'])
    return meta_data

# CLI
@click.command()
@click.argument("csv_file")
@click.argument("json_file")
def combine_manual_labels(csv_file: str, json_file: str):
  # Load the manual labels from CSV
    manual_labels = load_manual_labels(csv_file)

    # Load the metadata from the JSON file
    meta_data = load_frames_meta_data(json_file)

    # Extract image numbers from the 'Image' column in manual labels
    manual_labels['frame_number'] = manual_labels['Image'].apply(lambda x: x.split('/')[-1].split('.')[0])
    manual_labels['frame_number'] = manual_labels['frame_number'].astype(int)
    meta_data['frame_number'] = meta_data['frame_number'].astype(int)

    # Make sure we sort the data by frame number
    manual_labels = manual_labels.sort_values(by='frame_number')
    meta_data = meta_data.sort_values(by='frame_number')

    # Merge metadata with manual labels based on frame numbers, keeping all metadata frames
    # Missing latitude and longitude values will be NaNs
    merged_data = meta_data.merge(manual_labels, left_on='frame_number', right_on='frame_number', how='left')
    
    # Create image column in merged data
    merged_data['Image'] = merged_data['frame_number'].apply(lambda x: f"images/{x}.jpg")

    # Interpolate missing latitude and longitude values
    merged_data['Latitude'] = merged_data['Latitude'].astype(float)
    merged_data['Longitude'] = merged_data['Longitude'].astype(float)
    merged_data['Latitude'] = merged_data['Latitude'].interpolate()
    merged_data['Longitude'] = merged_data['Longitude'].interpolate()

    # Drop any rows which have a frame_number after the last frame_number in the manual labels
    merged_data = merged_data[merged_data['frame_number'] <= manual_labels['frame_number'].max()]

    # Update manual_labels with the merged and interpolated data
    manual_labels = merged_data

    # Extract the motion information
    manual_labels = extract_motion(manual_labels)

    # Plot the lat and long
    import matplotlib.pyplot as plt
    plt.plot(manual_labels['Longitude'], manual_labels['Latitude'])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Manual Labels')

    # plot the heading
    plt.figure()
    plt.plot(manual_labels['timestamp_unixus'], np.rad2deg(manual_labels['heading_rads_from_east']))
    plt.xlabel('Timestamp')
    plt.ylabel('Heading (degs from east)')
    # Fix axes to be betweeen -pi and pi
    plt.ylim(-180.0, 180.0)

    # plot the curvature
    plt.figure()
    plt.plot(manual_labels['timestamp_unixus'], manual_labels['curvature_invm'])
    plt.xlabel('Timestamp')
    plt.ylabel('Curvature (invm)')

    # Plot the speed
    plt.figure()
    plt.plot(manual_labels['timestamp_unixus'], manual_labels['speed_mps'])
    plt.xlabel('Timestamp')
    plt.ylabel('Speed (mps)')

    # Calculate the sun vector
    manual_labels['sun_vector_relative'] = manual_labels.apply(
        lambda x: calculate_sun_vector_relative(x['timestamp_unixus'], x['Latitude'], x['Longitude'], x['heading_rads_from_east']), axis=1)

    # Plot the sun vector
    sun_vectors = manual_labels['sun_vector_relative'].apply(pd.Series)
    plt.figure()
    plt.plot(manual_labels['timestamp_unixus'], sun_vectors[0], label='f')
    plt.plot(manual_labels['timestamp_unixus'], sun_vectors[1], label='l')
    plt.plot(manual_labels['timestamp_unixus'], sun_vectors[2], label='u')
    plt.xlabel('Timestamp')
    plt.ylabel('Sun Vector')
    plt.legend()

    plt.figure()
    # Plot the sun vector azimuth angle
    sun_vecs_x = np.array(sun_vectors[0])
    sun_vecs_y = np.array(sun_vectors[1])
    azimuth_angle_degs = np.rad2deg(np.arctan2(sun_vecs_y, sun_vecs_x))
    plt.plot(manual_labels['timestamp_unixus'], azimuth_angle_degs)
    plt.xlabel('Timestamp')
    plt.ylabel('Azimuth Angle (degs)')
    plt.ylim(-180.0, 180.0)

    plt.figure()
    plt.hist(azimuth_angle_degs, bins=10)
    plt.xlabel('Azimuth Angle (degs)')
    plt.ylabel('Frequency')

    # Split sun vector into 3 columns
    manual_labels[['sun_f', 'sun_l', 'sun_u']] = pd.DataFrame(manual_labels['sun_vector_relative'].tolist(), index=manual_labels.index)
    # Remove everything except the columns we want
    manual_labels['image_name'] = manual_labels['Image']
    manual_labels = manual_labels[['image_name', 'sun_f', 'sun_l', 'sun_u']]

    # Save the manual labels
    manual_labels.to_csv('manual_labels.csv', index=False)

    plt.show()

if __name__ == "__main__":
    combine_manual_labels()
    # Example usage:
    # python3 combine_manual_labels.py ../manual_labels/dublin.csv ../meta_info/dublin.json
