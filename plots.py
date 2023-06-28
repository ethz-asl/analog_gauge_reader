import os
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches

import cv2
from PIL import Image

# pylint: disable=no-member
from evaluation import constants
from geometry.ellipse import get_ellipse_pts, get_point_from_angle

matplotlib.use('Agg')

RUN_PATH = 'run'


class Plotter:
    def __init__(self, run_path, image):
        self.run_path = run_path
        os.mkdir(self.run_path)
        self.image = image

    def set_image(self, image):
        self.image = image

    def save_img(self):
        im = Image.fromarray(self.image)
        path = os.path.join(self.run_path, constants.ORIGINAL_IMG_FILE_NAME)
        im.save(path)

    def plot_image(self, title):
        plt.figure()
        plt.imshow(self.image)
        path = os.path.join(self.run_path, f"image_{title}.jpg")
        plt.savefig(path)
        # plt.show()

    def plot_any_image(self, img, title):
        plt.figure()
        plt.imshow(img)
        path = os.path.join(self.run_path, f"image_{title}.jpg")
        plt.savefig(path)

    def plot_point_img(self, img, points, title):
        plt.figure()
        plt.imshow(img)
        plt.scatter(points[:, 0], points[:, 1])
        path = os.path.join(self.run_path, f"image_{title}.jpg")
        plt.savefig(path)

    def plot_ocr_visualization(self, vis, degree=None):
        plt.figure()
        plt.imshow(vis)
        if degree is None:
            path = os.path.join(self.run_path,
                                "ocr_visualization_results_chosen.jpg")
        else:
            path = os.path.join(self.run_path,
                                f"ocr_visualization_results{degree}.jpg")
        plt.savefig(path)

    def plot_bounding_box_img(self, boxes):
        """
        plot detected bounding boxes. boxes is the result of the yolov8 detection
        :param img: image to draw bounding boxes on
        :param boxes: list of bounding boxes
        """
        img = np.copy(self.image)
        for bbox in boxes:
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))

            color = (0, 255, 0)

            img = cv2.rectangle(img,
                                start_point,
                                end_point,
                                color=color,
                                thickness=3)

        plt.figure()
        plt.imshow(img)

        path = os.path.join(self.run_path, "bbox_results.jpg")
        plt.savefig(path)

    def plot_test_point(self, point, title):
        plt.figure(figsize=(12, 8))
        plt.imshow(self.image)
        plt.scatter(point[0], point[1], s=100, c='red', marker='x')
        plt.title(f"{title} Point")

        plt.tight_layout()

        path = os.path.join(self.run_path, f"{title}_point_result.jpg")
        plt.savefig(path)

    def plot_key_points(self, key_point_list):
        plt.figure(figsize=(12, 8))

        titles = ['Start', 'Middle', 'End']

        if len(key_point_list) == 1:
            key_points = key_point_list[0]
            plt.imshow(self.image)
            plt.scatter(key_points[:, 0],
                        key_points[:, 1],
                        s=50,
                        c='red',
                        marker='x')
            plt.title('Predicted Key Point')

        else:
            for i in range(3):
                key_points = key_point_list[i]
                plt.subplot(1, 3, i + 1)
                plt.imshow(self.image)
                plt.scatter(key_points[:, 0],
                            key_points[:, 1],
                            s=50,
                            c='red',
                            marker='x')
                plt.title(f'Predicted Key Point {titles[i]}')

        plt.tight_layout()

        path = os.path.join(self.run_path, "key_point_results.jpg")
        plt.savefig(path)
        # plt.show()

    def plot_just_ellipse(self, image, ellipse_params, title):
        plt.figure()
        plt.imshow(image)
        x, y = get_ellipse_pts(ellipse_params)
        plt.plot(x, y)  # plot ellipse
        path = os.path.join(self.run_path, f"ellipse_{title}.jpg")
        plt.savefig(path)

    def plot_ellipse(self,
                     points,
                     ellipse_params,
                     title,
                     annotations=None,
                     annotation_colors=None):
        """
        plot ellipse and points with annotations.
        points is a 2d numpy array with one point per row
        """
        plt.figure()

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        ax.imshow(self.image)

        x = points[:, 0]
        y = points[:, 1]

        ax.scatter(x, y, c='#ff0000', s=50)  # plot points

        if annotations is not None and annotation_colors is not None:
            for x_coord, y_coord, annotation, color in zip(
                    x, y, annotations, annotation_colors):
                ax.annotate(annotation, (x_coord, y_coord),
                            fontsize=25,
                            c=color,
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(facecolor='#ffffff',
                                      alpha=0.5,
                                      edgecolor='none'))

        x, y = get_ellipse_pts(ellipse_params)
        plt.plot(x, y)  # plot ellipse

        path = os.path.join(self.run_path, f"ellipse_results_{title}.jpg")
        plt.savefig(path)
        # plt.show()

    def plot_zero_point_ellipse(self, zero_point, start_end_point,
                                ellipse_params):
        """
        plot ellipse and points with annotations.
        points is a 2d numpy array with one point per row
        """
        plt.figure()

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        ax.imshow(self.image)

        x = start_end_point[:, 0]
        y = start_end_point[:, 1]

        zero_point_color = '#41ff00'
        start_end_color = '#ff0000'

        x, y = get_ellipse_pts(ellipse_params)
        plt.plot(x, y)  # plot ellipse

        x = start_end_point[:, 0]
        y = start_end_point[:, 1]
        ax.scatter(x, y, c=start_end_color, s=100)  # plot start end point

        x = zero_point[0]
        y = zero_point[1]
        ax.scatter(x, y, c=zero_point_color, s=100)  # plot start end point

        zero_patch = patches.Patch(color=zero_point_color, label='zero-point')
        start_end_patch = patches.Patch(color=start_end_color,
                                        label='Start and End Point')
        plt.legend(handles=[zero_patch, start_end_patch])

        path = os.path.join(self.run_path, "ellipse_zero_point.jpg")
        plt.savefig(path)
        # plt.show()

    def plot_project_points_ellipse(self, number_labels, ellipse_params):
        projected_points = []
        annotations = []

        for number in number_labels:
            proj_point = get_point_from_angle(number.theta, ellipse_params)
            projected_points.append(proj_point)
            annotations.append(number.reading)

        projected_points_arr = np.array(projected_points)

        if len(projected_points) == 1:
            np.expand_dims(projected_points_arr, axis=0)

        ocr_color = '#38761d'
        annotation_colors = [ocr_color for _ in annotations]

        self.plot_ellipse(projected_points_arr,
                          ellipse_params,
                          title='projected',
                          annotations=annotations,
                          annotation_colors=annotation_colors)

    def plot_final_reading_ellipse(self, number_labels, needle_point, reading,
                                   ellipse_params):
        projected_points = []
        annotations = []

        for number in number_labels:
            proj_point = get_point_from_angle(number.theta, ellipse_params)
            projected_points.append(proj_point)
            annotations.append(number.reading)

        projected_points.append(needle_point)
        annotations.append(reading)

        projected_points_arr = np.array(projected_points)

        if len(projected_points) == 1:
            np.expand_dims(projected_points_arr, axis=0)

        ocr_color = '#38761d'
        final_reading_color = '#2b00ff'
        annotation_colors = [ocr_color for _ in annotations]
        annotation_colors[-1] = final_reading_color

        self.plot_ellipse(projected_points_arr,
                          ellipse_params,
                          title='final',
                          annotations=annotations,
                          annotation_colors=annotation_colors)

    def plot_ocr(self, readings, title):
        plt.figure()

        threshold = 0.9
        fig, ax = plt.subplots()

        fig.set_size_inches(8, 6)

        # Display the image using imshow
        ax.imshow(self.image)

        for reading in readings:
            if reading.confidence > threshold:
                polygon_patch = Polygon(reading.polygon,
                                        linewidth=2,
                                        edgecolor='r',
                                        facecolor='none')
                ax.add_patch(polygon_patch)
                ax.scatter(reading.center[0], reading.center[1])
                ax.annotate(reading.reading,
                            (reading.center[0], reading.center[1]),
                            fontsize=25,
                            c='#38761d',
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(facecolor='#ffffff',
                                      alpha=0.5,
                                      edgecolor='none'))
        plt.title(f"ocr results {title}")
        path = os.path.join(self.run_path, f"ocr_results_{title}.jpg")
        plt.savefig(path)
        # plt.show()

    def plot_segmented_line(self, x_coords, y_coords, x_start_end,
                            line_coeffs):
        line_fn = np.poly1d(line_coeffs)
        # Plot the line on top of the image
        plt.figure()
        plt.imshow(self.image)
        plt.scatter(x_coords, y_coords)
        plt.plot(x_start_end, line_fn(x_start_end), color='red')

        path = os.path.join(self.run_path, "segmentation_results.jpg")
        plt.savefig(path)

        # plt.show()

    def plot_heatmaps(self, heatmaps):
        plt.figure(figsize=(12, 8))

        titles = ['Start', 'Middle', 'End']

        if heatmaps.shape[0] == 1:
            heatmap_plot = plt.imshow(heatmaps[0],
                                      cmap=plt.cm.viridis,
                                      vmin=0,
                                      vmax=1)
            plt.colorbar(heatmap_plot, shrink=0.5)
            plt.title('Predicted Heatmap')

        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1,
                                                ncols=3,
                                                figsize=(15, 5))
            plt.subplots_adjust(wspace=0.2,
                                hspace=0.1,
                                left=0.1,
                                right=1.0,
                                top=0.9,
                                bottom=0.1)
            axis = [ax1, ax2, ax3]
            for i in range(3):
                im = axis[i].imshow(heatmaps[i],
                                    cmap=plt.cm.viridis,
                                    vmin=0,
                                    vmax=1)
                axis[i].set_title(f'Predicted Heatmap {titles[i]}')
            fig.colorbar(im, ax=axis, shrink=0.8)

        # plt.tight_layout()
        path = os.path.join(self.run_path, "heatmaps_results.jpg")
        plt.savefig(path)
        # plt.show()

    def plot_linear_fit(self, ocr_numbers, needle, line):
        plt.figure()

        x = [0, 2 * np.pi]

        plt.scatter(ocr_numbers[:, 0],
                    ocr_numbers[:, 1],
                    color='orange',
                    label='OCR_readings')
        plt.plot(x, line(x), color='blue', label='Fitted Line')
        plt.scatter(needle[0], needle[1], color='red', label='needle_point')

        # Add labels and title
        plt.xlabel('angle on ellipse')
        plt.ylabel('reading on gauge')

        # Add legend
        plt.legend()

        # Show the plot
        path = os.path.join(self.run_path, "reading_line_fit.jpg")
        plt.savefig(path)

    def plot_linear_fit_ransac(self, ocr_numbers, needle, line, inlier_mask,
                               outlier_mask):
        plt.figure()

        line_x = [0, 2 * np.pi]
        line_y = line(line_x)

        plt.scatter(ocr_numbers[:, 0][inlier_mask],
                    ocr_numbers[:, 1][inlier_mask],
                    color='orange',
                    label='Inlier ocr')

        plt.scatter(ocr_numbers[:, 0][outlier_mask],
                    ocr_numbers[:, 1][outlier_mask],
                    color='gold',
                    label='Outlier ocr')
        plt.plot(line_x, line_y, color='royalblue', label='Fitted Line')
        plt.scatter(needle[0], needle[1], color='red', label='needle_point')

        # Add labels and title
        plt.xlabel('angle on ellipse')
        plt.ylabel('reading on gauge')

        # Add legend
        plt.legend()

        # Show the plot
        path = os.path.join(self.run_path, "reading_line_fit.jpg")
        plt.savefig(path)
