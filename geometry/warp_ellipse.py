import numpy as np
import cv2


def warp_ellipse_to_circle(image, ellipse_center, ellipse_axes, ellipse_angle):
    image_height, image_width = image.shape[:2]

    # Define the source points (the coordinates of the four corners of the ellipse)
    x, y = ellipse_center
    major_axis, minor_axis = ellipse_axes
    source_points = np.array(
        [[-major_axis / 2, -minor_axis / 2], [major_axis / 2, -minor_axis / 2],
         [major_axis / 2, minor_axis / 2], [-major_axis / 2, minor_axis / 2]],
        dtype=np.float32)
    square_size = max(major_axis, minor_axis)
    destination_points = np.array([[-square_size / 2, -square_size / 2],
                                   [square_size / 2, -square_size / 2],
                                   [square_size / 2, square_size / 2],
                                   [-square_size / 2, square_size / 2]],
                                  dtype=np.float32)

    rotation_matrix = np.array(
        [[np.cos(ellipse_angle), np.sin(ellipse_angle)],
         [-np.sin(ellipse_angle),
          np.cos(ellipse_angle)]])
    source_points = source_points @ rotation_matrix
    destination_points = destination_points @ rotation_matrix

    source_points[:, 0] += x
    source_points[:, 1] += y
    destination_points[:, 0] += x
    destination_points[:, 1] += y

    source_points = source_points.astype(np.float32)
    destination_points = destination_points.astype(np.float32)

    # Calculate the perspective transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(
        source_points, destination_points)

    # Warp the image
    offsetSize = 0

    warped_image = cv2.warpPerspective(
        image, transformation_matrix,
        (image_width + offsetSize, image_height + offsetSize))

    return warped_image, transformation_matrix


def map_point_original_image(point_warp, transformation_matrix):
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    point_in_original_image = cv2.perspectiveTransform(
        np.array([[point_warp]], dtype=np.float32),
        inverse_transformation_matrix)[0][0]
    return point_in_original_image


def map_point_transformed_image(point_original, transformation_matrix):
    point_warp = cv2.perspectiveTransform(
        np.array([[point_original]], dtype=np.float32),
        transformation_matrix)[0][0]
    return point_warp
