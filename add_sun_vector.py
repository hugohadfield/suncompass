from typing import Union

from datetime import datetime, timezone
import numpy as np

from suncalc import get_position


def calculate_sun_vector_ENU(time: datetime, lat_degs: float, long_degs: float):
    """
    Calculate the sun vector in the ENU frame at the given time and location.
    """
    res = get_position(time, long_degs, lat_degs)
    az_anticlockwise_from_south = res['azimuth']
    az_anti_clockwise_from_east = -np.pi / 2 - az_anticlockwise_from_south
    altitude_from_horizon = res['altitude']
    x = np.cos(az_anti_clockwise_from_east) * np.cos(altitude_from_horizon)
    y = np.sin(az_anti_clockwise_from_east) * np.cos(altitude_from_horizon)
    z = np.sin(altitude_from_horizon)
    sun_vector_enu = np.array([x, y, z])
    return sun_vector_enu


def calculate_sun_vector_relative(time_unixus: int, lat_degs: float, long_degs: float, heading_rads_from_east: Union[float, np.complex128]):
    """
    Calculate the sun vector in the relative frame at the given time, location and heading.
    """
    # Calculate the sun vector from the ENU frame to a local FLU frame
    time = datetime.fromtimestamp(time_unixus / 1e6, tz=timezone.utc)
    sun_vector_enu = calculate_sun_vector_ENU(time, lat_degs, long_degs)

    rotation_matrix = np.array([
        [np.cos(heading_rads_from_east), np.sin(heading_rads_from_east), 0],
        [-np.sin(heading_rads_from_east), np.cos(heading_rads_from_east), 0],
        [0, 0, 1]
    ], dtype=np.result_type(heading_rads_from_east, sun_vector_enu.dtype))

    # Calculate the sun vector in the relative frame
    sun_vector_relative = rotation_matrix@sun_vector_enu
    return sun_vector_relative


def calculate_relative_azimuth_angle(time_unixus: int, lat_degs: float, long_degs: float, heading_rads_from_east: float):
    """
    Calculate the azimuth angle of the sun vector in the relative frame at the given time, location and heading.
    """
    sun_vector_relative = calculate_sun_vector_relative(time_unixus, lat_degs, long_degs, heading_rads_from_east)
    azimuth_angle_rads = np.atan2(sun_vector_relative[1], sun_vector_relative[0])
    return azimuth_angle_rads


def test_known_sun_vector():
    # At 12:00 UTC on the 21st of June at the equator, the sun vector should be mostly up
    # ie. [0, 0, 1]
    time = datetime(2022, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    lat_degs = 0
    long_degs = 0
    sun_vector = calculate_sun_vector_ENU(time, lat_degs, long_degs)
    assert abs(sun_vector[2]) > abs(sun_vector[0])
    assert abs(sun_vector[2]) > abs(sun_vector[1])
    assert sun_vector[2] > 0
    assert abs(sun_vector[0]) < 0.1

    # At 5:00 UTC on the 21st of June at 0 degrees, 
    # the sun vector should mostly point east and slightly up ie. [1, 0, 0.5]
    time = datetime(2022, 6, 21, 5, 0, 0, tzinfo=timezone.utc)
    lat_degs = 0
    long_degs = 0.0
    sun_vector = calculate_sun_vector_ENU(time, lat_degs, long_degs)
    assert abs(sun_vector[0]) > abs(sun_vector[1])
    assert abs(sun_vector[0]) > abs(sun_vector[2])
    assert sun_vector[0] > 0

    # At 12:00 UTC on the 21st of June at at -90 degrees east, the sun vector should be mostly east
    # ie. [1, 0, 0]
    time = datetime(2022, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    lat_degs = 0
    long_degs = -90.0
    sun_vector = calculate_sun_vector_ENU(time, lat_degs, long_degs)
    assert abs(sun_vector[0]) > abs(sun_vector[1])
    assert abs(sun_vector[0]) > abs(sun_vector[2])
    assert sun_vector[0] > 0
    assert abs(sun_vector[2]) < 0.1

    # At 8pm UTC on the 21st of June at 0 degrees west,
    # the sun vector should mostly point west and slightly up ie. [-1, 0, 0.5]
    time = datetime(2022, 6, 21, 20, 0, 0, tzinfo=timezone.utc)
    lat_degs = 0
    long_degs = 0.0
    sun_vector = calculate_sun_vector_ENU(time, lat_degs, long_degs)
    assert abs(sun_vector[0]) > abs(sun_vector[1])
    assert abs(sun_vector[0]) > abs(sun_vector[2])
    assert sun_vector[0] < 0

    # At 12:00 UTC on the 21st of June at 90 degrees east, the sun vector should be mostly west
    # ie. [-1, 0, 0]
    time = datetime(2022, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    lat_degs = 0
    long_degs = 90.0
    sun_vector = calculate_sun_vector_ENU(time, lat_degs, long_degs)
    assert abs(sun_vector[0]) > abs(sun_vector[1])
    assert abs(sun_vector[0]) > abs(sun_vector[2])
    assert sun_vector[0] < 0
    assert abs(sun_vector[2]) < 0.1

    # At 12:00 UTC on the 21st of June in the north pole the sun vector should be mostly south
    # ie. [0, -1, 0]
    time = datetime(2022, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    lat_degs = 89.0
    long_degs = 0
    sun_vector = calculate_sun_vector_ENU(time, lat_degs, long_degs)
    assert abs(sun_vector[1]) > abs(sun_vector[0])
    assert abs(sun_vector[1]) > abs(sun_vector[2])
    assert sun_vector[1] < 0
    assert abs(sun_vector[0]) < 0.1

    # At 12:00 UTC on the 21st of June in the south pole the sun vector should be mostly north
    # ie. [0, 1, 0]
    time = datetime(2022, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    lat_degs = -89.0
    long_degs = 0
    sun_vector = calculate_sun_vector_ENU(time, lat_degs, long_degs)
    assert abs(sun_vector[1]) > abs(sun_vector[0])
    assert abs(sun_vector[1]) > abs(sun_vector[2])
    assert sun_vector[1] > 0
    assert abs(sun_vector[0]) < 0.1


def test_known_calculate_relative_azimuth_angle():
    # If you are near the north pole and facing south at 12:00 UTC on the 21st of June, the sun vector azimuth angle should be 0
    time = datetime(2022, 6, 21, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1e6
    lat_degs = 89.0
    long_degs = 0
    heading_rads_from_east = -np.pi/2
    azimuth_angle_rads = calculate_relative_azimuth_angle(time, lat_degs, long_degs, heading_rads_from_east)
    assert abs(np.rad2deg(azimuth_angle_rads)) < 1 # 1 degree tolerance

    # If you are near the north pole and facing east at 12:00 UTC on the 21st of June, the sun vector azimuth angle should be -90
    time = datetime(2022, 6, 21, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1e6
    lat_degs = 80.0
    long_degs = 0
    heading_rads_from_east = 0
    azimuth_angle_rads = calculate_relative_azimuth_angle(time, lat_degs, long_degs, heading_rads_from_east)
    assert abs(np.rad2deg(azimuth_angle_rads) + 90) < 1 # 1 degree tolerance

    # If you are at 90 longitude and facing east at 12:00 UTC on the 21st of June, the sun vector azimuth angle should be 180 / -180
    time = datetime(2022, 6, 21, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1e6
    lat_degs = 0
    long_degs = 90.0
    heading_rads_from_east = 0
    azimuth_angle_rads = calculate_relative_azimuth_angle(time, lat_degs, long_degs, heading_rads_from_east)
    assert abs(np.rad2deg(azimuth_angle_rads)) - 180 < 30 # 30 degree tolerance


if __name__ == '__main__':
    test_known_sun_vector()
    test_known_calculate_relative_azimuth_angle()