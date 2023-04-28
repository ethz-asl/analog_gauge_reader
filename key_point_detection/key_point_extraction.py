import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from scipy.spatial.distance import cdist


def full_key_point_extraction(heatmaps,
                              threshold=0.4,
                              bandwidth=5,
                              visualize=False):
    key_point_list = []
    for i in range(heatmaps.shape[0]):
        middle = i == 1
        if middle:
            threshold = 0.6
        else:
            threshold = 0.8
        cluster_centers = extract_key_points(heatmaps[i], threshold, bandwidth,
                                             visualize)
        key_point_list.append(cluster_centers)
    return key_point_list


def extract_key_points(heatmap, threshold, bandwidth=5, visualize=False):
    """
    threshold is minimum confidence for points to be considered in clustering.
    increasing the threshold increases performance
    bandwidth is bandwidth parameter of Mean shift.
    return extracted cluster centers
    """

    # Get pixel coordinates of pixels with value greater than 0.5
    coords = np.argwhere(heatmap > threshold)
    # swap coordinates
    coords[:, [1, 0]] = coords[:, [0, 1]]

    # if none detected with given threshold
    if coords.shape[0] == 0:
        if threshold <= 0.1:
            print(f"No point with confidence at least {threshold} detected.")
            return coords

        new_threshold = threshold / 2
        print(f"No point with confidence at least {threshold} detected. "
              f"Trying threshold {new_threshold}")
        return extract_key_points(heatmap, threshold=new_threshold)

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

    return cluster_centers


def plot_key_points(image, key_points, file_path, plot=False):
    plt.imshow(image)
    plt.scatter(key_points[:, 0], key_points[:, 1], s=50, c='red', marker='x')
    # save plot
    plt.savefig(file_path, bbox_inches='tight')

    # Show the plot
    if plot:
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
