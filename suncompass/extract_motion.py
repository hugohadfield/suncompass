
import pandas as pd
import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in kilometers is 6371
    m =  c * 6371000
    return m


def calculate_bearing_clockwise_from_north(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points on the Earth.
    
    Parameters:
    lat1, lon1 -- latitude and longitude of the first point in radians
    lat2, lon2 -- latitude and longitude of the second point in radians
    
    Returns:
    Bearing in radians from the first point to the second point.
    """
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    theta = np.arctan2(y, x)
    return theta


def calculate_heading_anticlockwise_from_east(lat1, lon1, lat2, lon2):
    # Convert bearing from clockwise from North to anticlockwise from East
    heading = np.pi / 2 - calculate_bearing_clockwise_from_north(lat1, lon1, lat2, lon2)
    # Normalize the heading to be between -pi and pi
    heading = (heading + np.pi) % (2 * np.pi) - np.pi
    return heading


def calculate_headings(latitudes_deg, longitudes_deg, starting_orientation=None):
    # Calculate the heading anti-clockwise from east
    est_starting_orientation = starting_orientation
    headings_rads_anticlockwise_from_east = []
    for i in range(1, len(latitudes_deg)):
        if latitudes_deg[i] == latitudes_deg[i-1] and longitudes_deg[i] == longitudes_deg[i-1]:
            if len(headings_rads_anticlockwise_from_east) == 0:
                headings_rads_anticlockwise_from_east.append(est_starting_orientation)
            else:
                headings_rads_anticlockwise_from_east.append(headings_rads_anticlockwise_from_east[-1])
        else:
            heading = calculate_heading_anticlockwise_from_east(
                np.deg2rad(latitudes_deg[i-1]), 
                np.deg2rad(longitudes_deg[i-1]), 
                np.deg2rad(latitudes_deg[i]), 
                np.deg2rad(longitudes_deg[i])
            )
            headings_rads_anticlockwise_from_east.append(heading)
            if est_starting_orientation is None:
                est_starting_orientation = heading
    return headings_rads_anticlockwise_from_east, est_starting_orientation


def extract_motion(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the speed, global heading and curvature from the given data.
    """
    # Get timestamps
    timestamps_s = data['timestamp_unixus'] / 1e6

    # Get the latitude and longitude
    latitudes_deg = data['Latitude']
    longitudes_deg = data['Longitude']

    # Calculate the distance between each point
    distances_m = []
    for i in range(1, len(latitudes_deg)):
        distances_m.append(haversine(latitudes_deg[i-1], longitudes_deg[i-1], latitudes_deg[i], longitudes_deg[i]))
    
    # Calculate the time between each point
    times_s = []
    for i in range(1, len(timestamps_s)):
        times_s.append(timestamps_s[i] - timestamps_s[i-1])

    # Calculate the speed between each point
    speeds_mps = []
    for i in range(len(distances_m)):
        speeds_mps.append(distances_m[i] / times_s[i])

    # Calculate the heading in radians
    _, starting_orient = calculate_headings(latitudes_deg, longitudes_deg)
    headings_rads_anticlockwise_from_east = calculate_headings(latitudes_deg, longitudes_deg, starting_orient)[0]
    print(headings_rads_anticlockwise_from_east)
    
    # Calculate the curvature in inverse meters
    # Positive curvature means turning left, negative curvature means turning right
    curvatures_inv_m = []
    for i in range(1, len(headings_rads_anticlockwise_from_east)):
        delta_heading = headings_rads_anticlockwise_from_east[i] - headings_rads_anticlockwise_from_east[i-1]
        if distances_m[i] == 0:
            curvature = 0
        else:
            curvature = delta_heading / distances_m[i]
        curvatures_inv_m.append(curvature)

    # Add the speed, heading and curvature to the data
    print(data['timestamp_unixus'].shape)
    print(len(speeds_mps))
    print(len(curvatures_inv_m))
    data['speed_mps'] = [speeds_mps[0]] + speeds_mps
    data['heading_rads_from_east'] = [headings_rads_anticlockwise_from_east[0]] + headings_rads_anticlockwise_from_east
    data['curvature_invm'] = [0] + [0] + curvatures_inv_m
    return data


def test_clockwise_anticlockwise():
    """
    90 degs clockwise from north is 0 anti east 
    0 degs clockwise from north is 90 anti east
    180 degs clockwise from north is -90 anti east
    """
    # Directly east
    np.testing.assert_allclose(calculate_heading_anticlockwise_from_east(0, 0, 0, -np.pi/2), 0, abs_tol=1e-6)
    # Directly north
    np.testing.assert_allclose(calculate_heading_anticlockwise_from_east(0, 0, -np.pi/2+0.00001, 0), np.pi/2, abs_tol=1e-6)
    # Directly south
    np.testing.assert_allclose(calculate_heading_anticlockwise_from_east(np.pi-0.00001, 0, 0, 0), -np.pi/2, abs_tol=1e-6)
