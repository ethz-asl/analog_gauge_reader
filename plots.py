import os
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt
import cv2

from geometry.ellipse import get_ellipse_pts, get_point_from_angle

RUN_PATH = 'run'


class Plotter:
    def __init__(self, run_path, image):
        self.run_path = run_path
        os.mkdir(self.run_path)
        self.image = image

    def set_image(self, image):
        self.image = image

    def plot_image(self, title):
        plt.figure()
        plt.imshow(self.image)
        path = os.path.join(self.run_path, f"image_{title}.jpg")
        plt.savefig(path)
        # plt.show()

    def plot_ocr_visualization(self, vis):
        plt.figure()
        plt.imshow(vis)
        path = os.path.join(self.run_path, "ocr_visualization_results.jpg")
        plt.savefig(path)

    def plot_bounding_box_img(self, boxes):
        """
        plot detected bounding boxes. boxes is the result of the yolov8 detection
        :param img: image to draw bounding boxes on
        :param boxes: list of bounding boxes
        """
        img = np.copy(self.image)
        for box in boxes:
            bbox = box.xyxy[0].int()
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))

            color_face = (0, 255, 0)
            color_needle = (255, 0, 0)
            if box.cls == 0:
                color = color_face
            else:
                color = color_needle

            img = cv2.rectangle(img,
                                start_point,
                                end_point,
                                color=color,
                                thickness=5)

        plt.figure()
        plt.imshow(img)

        path = os.path.join(self.run_path, "bbox_results.jpg")
        plt.savefig(path)
        # plt.show()

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

        ax.scatter(x, y, marker='x', c='red', s=50)  # plot points

        if annotations is not None and annotation_colors is not None:
            for x_coord, y_coord, annotation, color in zip(
                    x, y, annotations, annotation_colors):
                ax.annotate(annotation, (x_coord, y_coord),
                            fontsize=20,
                            c=color)

        x, y = get_ellipse_pts(ellipse_params)
        plt.plot(x, y)  # plot ellipse

        path = os.path.join(self.run_path, f"ellipse_results_{title}.jpg")
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

        ocr_color = '#ff7600'
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

        ocr_color = '#ff7600'
        final_reading_color = '#00ccff'
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
        plt.title(f"ocr results {title}")
        path = os.path.join(self.run_path, f"ocr_results_{title}.jpg")
        plt.savefig(path)
        # plt.show()

    def plot_segmented_line(self, x_coords, y_coords, line_coeffs):
        line_fn = np.poly1d(line_coeffs)
        # Plot the line on top of the image
        plt.figure()
        plt.imshow(self.image)
        plt.scatter(x_coords, y_coords)
        plt.plot(x_coords, line_fn(x_coords), color='red')

        path = os.path.join(self.run_path, "segmentation_results.jpg")
        plt.savefig(path)

        # plt.show()

    def plot_heatmaps(self, heatmaps):
        plt.figure(figsize=(12, 8))

        titles = ['Start', 'Middle', 'End']

        if heatmaps.shape[0] == 1:
            plt.imshow(heatmaps[0], cmap=plt.cm.viridis)
            plt.title('Predicted Heatmap')

        else:
            for i in range(3):
                plt.subplot(1, 3, i + 1)
                plt.imshow(heatmaps[i], cmap=plt.cm.viridis)
                plt.title(f'Predicted Heatmap {titles[i]}')

        plt.tight_layout()
        path = os.path.join(self.run_path, "heatmaps_results.jpg")
        plt.savefig(path)
        # plt.show()
