import os
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


class EvalPlotter:
    def __init__(self, run_path, image):
        self.run_path = run_path
        self.image = image

    def set_image(self, image):
        self.image = image

    def plot_image(self, title):
        plt.figure()
        plt.imshow(self.image)
        path = os.path.join(self.run_path, f"image_{title}.jpg")
        plt.savefig(path)

    def plot_bounding_box_img(self, ann_boxes, pred_boxes, title):

        plt.figure()

        # pylint: disable-next=unused-variable
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(self.image)

        # Draw the bounding boxes on the image
        for bbox in pred_boxes:
            rect = patches.Rectangle((bbox['x'], bbox['y']),
                                     bbox['width'],
                                     bbox['height'],
                                     linewidth=2,
                                     edgecolor='g',
                                     facecolor='none')
            ax.add_patch(rect)

        for bbox in ann_boxes:
            rect = patches.Rectangle((bbox['x'], bbox['y']),
                                     bbox['width'],
                                     bbox['height'],
                                     linewidth=2,
                                     edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

        green_patch = patches.Patch(color='green', label='Predictions')
        red_patch = patches.Patch(color='red', label='Annotations')
        plt.legend(handles=[green_patch, red_patch])

        path = os.path.join(self.run_path, f"{title}_bbox_results.jpg")
        plt.savefig(path)

    def plot_key_points(self, ann_keypoints, pred_keypoints, title):
        plt.figure()

        # pylint: disable-next=unused-variable
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(self.image)

        ax.scatter(ann_keypoints[:, 0],
                   ann_keypoints[:, 1],
                   s=50,
                   c='red',
                   marker='x')
        ax.scatter(pred_keypoints[:, 0],
                   pred_keypoints[:, 1],
                   s=50,
                   c='green',
                   marker='x')

        green_patch = patches.Patch(color='green', label='Predictions')
        red_patch = patches.Patch(color='red', label='Annotations')
        plt.legend(handles=[green_patch, red_patch])

        path = os.path.join(self.run_path, f"{title}_keypoint_results.jpg")
        plt.savefig(path)
