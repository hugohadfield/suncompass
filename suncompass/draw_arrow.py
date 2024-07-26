import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import click


def apply_projection_to_point(projection_matrix, point):
    """
    Apply a projection matrix to a 3D point.
    """
    # Convert to homogeneous coordinates
    point_hom = np.append(point, 1)

    # Project to 2D image coordinates
    point_image_hom = projection_matrix @ point_hom

    if point_image_hom[2] != 0:
        point_image = point_image_hom[:2] / point_image_hom[2]
        return point_image
    else:
        return None
    

def generate_projection_matrix(K, R, t):
    """
    Generate a projection matrix from the camera intrinsics matrix (K), rotation matrix (R) and translation vector (t).
    """
    extrinsics = np.hstack((R, t))
    # Image flip matrix, allows us to use focal lengths with positive values
    flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    return K @ flip_matrix @ extrinsics


def generate_guessed_projection_matrix(K, camera_height):
    """
    Generate a projection matrix for a camera placed at a height above the ground, looking directly forward.
    """
    R = np.eye(3)
    t = np.array([0, -camera_height, 0]).reshape(3, 1)
    return generate_projection_matrix(K, R, t)



def draw_point_in_image(img, K, point_3d, camera_height_m, color=(255, 255, 255)):
    """
    Draw a 3D point in the image.
    """
    img_shape = img.shape

    # Create the projection matrix
    projection_matrix = generate_guessed_projection_matrix(K, camera_height_m)

    # Apply projection to the point
    point_image = apply_projection_to_point(projection_matrix, point_3d)

    # Draw point on the image
    point = tuple(point_image.astype(int))
    if all(0 <= p < img_shape[1] for p in point):
        cv2.circle(img, point, 1, color, -1)
    else:
        print(f'Point is out of image bounds: {point}')

    return img


def draw_circle_in_image(img, K, circle_forward_m, circle_radius_m, camera_height_m, color=(255, 255, 255)):
    """
    Generate a circle in the image.
    """
    # Generate a set of points on the circle
    angles = np.linspace(0, 2 * np.pi, 1000)
    circle_points = np.array([[-circle_radius_m * np.sin(theta), 0.0, circle_forward_m + circle_radius_m * np.cos(theta)] for theta in angles])

    # Draw each point on the image
    for point in circle_points:
        img = draw_point_in_image(img, K, point, camera_height_m, color=color)
    return img


def draw_arrow_in_image(img, K, arrow_forward_m=5.0, theta_rad=0.0, camera_height_m=1.5, arrow_length_m=1.0, 
                        color=(255, 255, 255)):
    """
    Camera and Ground Frame Conventions:

    Camera Frame:
        - Origin: Optical center of the camera.
        - x-axis: Points to the right.
        - y-axis: Points downward.
        - z-axis: Points forward (out of the camera lens).

    Ground Frame:
        - Assumed to be a horizontal plane at y = 0.
        - x-axis: Points to the right.
        - y-axis: Points downward.
        - z-axis: Points forward (aligned with the camera's view direction).

    Transformations:
        - The camera is placed at a height (camera_height_m) above the ground, looking directly forward so that the camera bore is parallel to the plane of the ground
        - The extrinsics matrix combines a rotation matrix (R) and a translation vector (t) to transform points from the ground frame to the camera frame.
        - The rotation matrix (R) is typically the identity matrix in this scenario, indicating no rotation.
        - The translation vector (t) represents the camera's height in the negative y direction in the ground frame.

    Projection:
        - Points in the camera frame are projected onto the 2D image plane using the camera intrinsics matrix (K).
    """
    img_shape = img.shape

    # Define arrow coordinates in the ground frame
    arrow_start_ground = np.array([0, 0, arrow_forward_m])  # Start of the arrow in ground frame
    arrow_end_ground = np.array([-arrow_length_m * np.sin(theta_rad), 0, arrow_forward_m + arrow_length_m * np.cos(theta_rad)])  # End of the arrow in ground frame

    # Create the projection matrix
    projection_matrix = generate_guessed_projection_matrix(K, camera_height_m)

    # Apply projection to arrow points
    arrow_start_image = apply_projection_to_point(projection_matrix, arrow_start_ground)
    arrow_end_image = apply_projection_to_point(projection_matrix, arrow_end_ground)

    if arrow_start_image is not None and arrow_end_image is not None:
        # Draw arrow on the image
        start_point = tuple(arrow_start_image.astype(int))
        end_point = tuple(arrow_end_image.astype(int))

        if (start_point[1] < img_shape[0] and end_point[1] < img_shape[0] and start_point[0] < img_shape[1] and end_point[0] < img_shape[1]) \
            and (start_point[1] >= 0 and end_point[1] >= 0 and start_point[0] >= 0 and end_point[0] >= 0):
                cv2.arrowedLine(img, start_point, end_point, color, 2)
        else:
            print(f'Arrow points are out of image bounds. Start: {start_point}, End: {end_point}')
    else:
        print("Arrow points are behind the camera.")

    return img


def draw_all_on_image(img, K, theta_rad):
    """
    Draws everything on the image.
    """
    # Forward arrow offset meters (arrow_forward_m)
    arrow_forward_m = 10.0
    # Height of the camera above the ground meters (camera_height_m)
    camera_height_m = 1.5
    # Length of the arrow itself 
    arrow_length_m = 2.0
    # Draw the sun arrow and circle
    circle_forward_m = arrow_forward_m
    circle_radius_m = arrow_length_m
    img = draw_circle_in_image(img, K, circle_forward_m, circle_radius_m, camera_height_m, 
                                color=(100, 100, 50))
    img = draw_arrow_in_image(img, K, arrow_forward_m=arrow_forward_m, theta_rad=0, camera_height_m=camera_height_m, arrow_length_m=arrow_length_m,
                            color=(255, 255, 100))
    img = draw_arrow_in_image(img, K, arrow_forward_m=arrow_forward_m, theta_rad=theta_rad, camera_height_m=camera_height_m, arrow_length_m=arrow_length_m,
                            color=(0, 0, 255))
    return img


def draw_all_on_image_fl(img, K: np.ndarray, f_m: float, l_m: float):
    """
    Draws everything on the image.
    """
    theta_rad = np.arctan2(l_m, f_m)
    return draw_all_on_image(img, K, theta_rad)


def test_draw_arrow():
    # Define the camera intrinsic matrix K
    K = np.array([[518.3461 ,   0.     , 215.1491 ],
        [  0.     , 518.98865, 121.62189],
        [  0.     ,   0.     ,   1.     ]], dtype=np.float32)

    # Define the image shape
    img_shape = (240, 420, 3)

    # Yaw angle relative to the camera's bore axis (theta_rad)
    theta_rad = np.deg2rad(45.0)

    # Create a blank image
    img = np.zeros(img_shape, dtype=np.uint8)
    img = draw_all_on_image(K, img, theta_rad=theta_rad)

    # Display the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Arrow in Camera Image')
    plt.axis('off')
    plt.show()


@click.command()
@click.option('--output_dir', default='output', help='Directory to save output images')
@click.option('--k_npy', default='K.npy', help='Numpy file containing the intrinsic matrix K')
@click.option('--csv', default='data.csv', help='CSV file containing the results of running fl on the images')
def cli(k_npy, output_dir, csv):
    """
    Create a video from a set of images.
    """
    # Load the intrinsic matrix K
    K = np.load(k_npy)

    # Load the results from the CSV file
    results = pd.read_csv(csv, header=None, names=['image_name', 'f', 'l', 'f_var', 'l_var'])

    # Loop through the images
    for i, row in results.iterrows():
        # Load the image
        img_path = row['image']
        img = cv2.imread(img_path)

        # Get the values of f and l
        f_m = row['f']
        l_m = row['l']

        # Draw the arrow on the image
        img = draw_all_on_image_fl(img, K, f_m, l_m)

        # Save the image
        output_path = os.path.join(output_dir, row['image_name'])
        cv2.imwrite(output_path, img)

if __name__ == '__main__':
    cli()
