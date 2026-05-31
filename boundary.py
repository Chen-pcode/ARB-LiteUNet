import os  
import numpy as np
import cv2
from scipy.optimize import differential_evolution
import time


input_folder = '/data1/users/caojiarui/cjr/LDEB-UNet/data/isic2018/train/masks/'
output_folder = '/data1/users/caojiarui/cjr/LDEB-UNet/data/isic2018/train/boundary/'


os.makedirs(output_folder, exist_ok=True)


def generate_edge_mask(mask, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    edge = dilated - eroded
    return edge


def calculate_iou_polygon(points, b):
    mask = np.zeros_like(b, dtype=np.uint8)
    points = np.int32(points)
    cv2.fillPoly(mask, [points], 255)
    intersection = cv2.bitwise_and(mask, b)
    union = cv2.bitwise_or(mask, b)
    intersection_area = np.count_nonzero(intersection)
    union_area = np.count_nonzero(union)
    iou = intersection_area / float(union_area) if union_area != 0 else 0
    return iou


def objective_function(points, a_points, b):
    points = points.reshape(-1, 2)
    for i in range(len(points)):
        dist = np.linalg.norm(a_points - points[i], axis=1)
        nearest_point_idx = np.argmin(dist)
        points[i] = a_points[nearest_point_idx]
    return -calculate_iou_polygon(points, b)  # 目标是最大化IoU，返回负值


for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        output_file_path = os.path.join(output_folder, filename)

        if os.path.exists(output_file_path):
            print(f"File {filename} already processed, skipping.")
            continue

        file_path = os.path.join(input_folder, filename)
        b = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 读取掩膜图像

        a = generate_edge_mask(b, kernel_size=3)

        _, a_binary = cv2.threshold(a, 200, 255, cv2.THRESH_BINARY)
        a_points = np.column_stack(np.where(a_binary == 255))

        n = 6
        np.random.seed(42)
        selected_points = a_points[np.random.choice(a_points.shape[0], n, replace=False)]

        start_time = time.time()

        bounds = [(a_points[:, 0].min(), a_points[:, 0].max()), (a_points[:, 1].min(), a_points[:, 1].max())] * n
        result = differential_evolution(objective_function, bounds, args=(a_points, b), maxiter=1000)

        optimized_points = result.x.reshape(-1, 2)
        for i in range(len(optimized_points)):
            dist = np.linalg.norm(a_points - optimized_points[i], axis=1)
            nearest_point_idx = np.argmin(dist)
            optimized_points[i] = a_points[nearest_point_idx]

        optimized_iou = -result.fun

        a_color = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

        print(f"Optimizing points for {filename}")
        for point in optimized_points:
            cv2.circle(a_color, (int(point[1]), int(point[0])), radius=2, color=(255, 255, 255), thickness=-1)

        cv2.imwrite(output_file_path, a_color)

        print(f"Optimized IoU for {filename}: {optimized_iou}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for generating keypoints for {filename}: {elapsed_time:.4f} seconds")
