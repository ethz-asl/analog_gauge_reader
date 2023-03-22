import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from scipy.spatial.distance import cdist


def extract_key_points(heatmap, threshold, bandwidth, visualize=False):
    # Get pixel coordinates of pixels with value greater than 0.5
    coords = np.argwhere(heatmap > threshold)

    # swap coordinates
    coords[:, [1, 0]] = coords[:, [0, 1]]

    # Perform mean shift clustering
    ms = MeanShift(bandwidth=bandwidth, n_jobs=-1)
    ms.fit(coords)

    # Plot results
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    if visualize:
        plt.scatter(coords[:, 0], coords[:, 1], c=labels)
        plt.scatter(cluster_centers[:, 0],
                    cluster_centers[:, 1],
                    marker='x',
                    color='red',
                    s=300,
                    linewidths=1,
                    zorder=10)
        plt.show()


def plot_key_points(image, key_points):
    plt.imshow(image)
    plt.scatter(key_points[:, 0], key_points[:, 1], s=50, c='red', marker='x')
    # show the plot
    plt.show()


def key_point_metrics(predicted, ground_truth, threshold=5):
    """
    Gives back three different metrics to evaluate the predicted keypoints.
    For mean_distance each prediction is assigned to the true keypoint
    with smallest distance to it and then these distances are averaged
    For p_non_assigned we have the percentage of predicted key points
    that are not close to any true keypoint and therefore are non_assigned.
    For pck we have the percentage of true key points,
    where at least one predicted key point is close to it.

    For both p_non_assigned and pck,
    two key_points being close means that their distance is smaller than the threshold.
    :param predicted:
    :param ground_truth:
    :param threshold:
    :return:
    """
    distances = cdist(predicted, ground_truth)

    cor_pred_indices = np.argmin(
        distances, axis=1)  # indices of truth that are closest to predictions
    cor_true_indices = np.argmin(
        distances, axis=0)  # indices of predictions that are closest to truth

    # extract the corresponding ground truth points
    corresponding_truth = ground_truth[cor_pred_indices]

    # calculate the Euclidean distances between predicted points and corresponding groundtruths
    pred_distances = np.linalg.norm(predicted[:len(corresponding_truth)] -
                                    corresponding_truth,
                                    axis=1)
    mean_distance = np.mean(pred_distances)

    non_assigned = np.sum(pred_distances > threshold)
    p_non_assigned = non_assigned / len(predicted)

    # extract the corresponding predicted points
    corresponding_pred = predicted[cor_true_indices]

    gt_distances = np.linalg.norm(ground_truth[:len(corresponding_pred)] -
                                  corresponding_pred,
                                  axis=1)
    correct = np.sum(gt_distances <= threshold)
    pck = correct / len(
        ground_truth
    )  # compute PCK as percentage of correctly predicted keypoints

    return mean_distance, pck, p_non_assigned
