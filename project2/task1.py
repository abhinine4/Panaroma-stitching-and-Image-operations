"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
from scipy.spatial.distance import cdist


# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    min_match = 8

    left_g = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d_SIFT.create(nfeatures=500)  # create sift object

    right_p, right_d = sift.detectAndCompute(right_g, None)
    left_p, left_d = sift.detectAndCompute(left_g, None)
    rp = cv2.KeyPoint_convert(right_p)
    lp = cv2.KeyPoint_convert(left_p)

    good_matches = []
    for i in range(len(right_d)):
        x = right_d[i]
        distances = []
        for j in range(len(left_d)):
            y = left_d[j]
            dist = calEuclidian(x, y)
            distances.append((j, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbours = []
        for k in range(2):
            neighbours.append(distances[k])
        if neighbours[0][1] < 0.75 * neighbours[1][1]:
            good_matches.append((rp[i], lp[neighbours[0][0]]))
    hom, best_inliers = ransac(good_matches)

    (w1, h1), (w2, h2) = left_g.shape, right_g.shape
    left_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    right_dims = cv2.perspectiveTransform(np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2), hom)
    result_dims = np.concatenate((left_dims, right_dims), axis=0)

    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    transformation = np.array([[1, 0, -x_min],
                               [0, 1, -y_min],
                               [0, 0, 1]])

    result_img = cv2.warpPerspective(right_img, transformation.dot(hom),
                                     (x_max - x_min, y_max - y_min))
    result_img[-y_min:w1 + -y_min, -x_min:h1 + -x_min] = left_img

    return result_img


def ransac(matches):
    x1 = []
    x2 = []
    for i in range(len(matches)):
        x1.append(matches[i][0])
        x2.append(matches[i][1])
    k = 5000
    tr = 5
    n = 4
    max_no_inliers = -1
    best_inliers = None
    best_hom = None
    for iter in range(k):
        permutated_index = np.random.permutation(np.arange(len(x1) - 1))
        sample_indices = permutated_index[:n]
        test_indices = permutated_index[n:]
        inliers = []

        x1_sample = [x1[i] for i in sample_indices]
        x2_sample = [x2[i] for i in sample_indices]

        x1_test = [x1[i] for i in test_indices]
        x2_test = [x2[i] for i in test_indices]

        hom = calhomography(x1_sample, x2_sample)

        for i in range(len(x1_test)):
            d = geoDist(x1_test[i], x2_test[i], hom)
            if d < tr:
                inliers.append((x1_test, x2_test))

        if len(inliers) > max_no_inliers:
            max_no_inliers = len(inliers)
            best_hom = hom
            best_inliers = inliers
    return best_hom, best_inliers


def calhomography(x1, x2):
    n = 4
    H = np.zeros((2 * n, 9), dtype=np.float32)
    for j in range(4):
        x = x1[j][0]
        y = x1[j][1]
        x_prime = x2[j][0]
        y_prime = x2[j][1]
        row1 = np.array([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])
        row2 = np.array([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y, -y_prime])

        H[2 * j] = row1
        H[(2 * j) + 1] = row2
    u, s, vt = np.linalg.svd(H)
    h = np.reshape(vt[8], (3, 3))
    w = h[2][2]
    h = (1 / w) * h
    return h


def geoDist(x1, x2, hom):
    p1 = np.transpose(np.matrix([x1[0], x1[1], 1]))
    estimatep2 = np.dot(hom, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2
    p2 = np.transpose(np.matrix([x2[0], x2[1], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


def calEuclidian(l, r):
    x_np = np.asarray(l)
    y_np = np.asarray(r)
    distance = np.sqrt(np.sum((x_np - y_np) ** 2))
    return distance


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)
