import math
import os
import random

import numpy as np
import json
import tqdm

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def corrupt_rotate(pointcloud, level=0):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    angle_clip = math.pi / 3
    angle_clip = angle_clip / 3 * (level + 1)
    angles = np.random.uniform(-angle_clip, angle_clip, size=(3))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pointcloud = np.dot(pointcloud, R)
    return pointcloud

def corrupt_jitter(pointcloud, level):
    """
    Jitter the input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    sigma = 0.01 * (level * 0.1 + 1)
    N, C = pointcloud.shape
    pointcloud = pointcloud + sigma * np.random.randn(N, C)
    return pointcloud

def corrupt_scale(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    xyz = np.random.uniform(low=1. / s, high=s, size=[3])
    return pc_normalize(np.multiply(pointcloud, xyz).astype('float32'))

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

if __name__ == "__main__":
    color_map = np.load("color_map_final.npy") # (50,3)

    root_path = "ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal"
    parts = ["shuffled_train_file_list", "shuffled_test_file_list"]
    parts_dict = {"shuffled_train_file_list": "Train_dataset", "shuffled_test_file_list": "Test_dataset"}
    train_list = []
    test_list = []

    augmentations = ["clean", "rotation", "scale", "jitter"]

    for part in parts:
        file_list_path = os.path.join(root_path, "train_test_split", part + ".json")
        file_list = json.load(open(file_list_path))
        for file in tqdm.tqdm(file_list):
            data_path = os.path.join(root_path, file[11:] + ".txt") # ShapeNet55-34\shapenet_pc\04256520-9ac58aaf9989a0911f98c0761af40e04.npy

            data = np.loadtxt(data_path).astype(np.float32) # (8192, 3)
            data = farthest_point_sample(data, 1024) # (1024, 3)
            point_set = data[:, :3]
            point_set = pc_normalize(point_set) # (-1, 1)

            color = data[:, -1].astype("int")
            target = color_map[color]

            for augmentation in augmentations:
                corrupt_level = random.randint(0, 4)
                point_set_corrupt = point_set
                if augmentation == "clean":
                    pass
                elif augmentation == "rotation":
                    point_set_corrupt = corrupt_rotate(point_set_corrupt, level=corrupt_level)
                elif augmentation == "scale":
                    point_set_corrupt = corrupt_scale(point_set_corrupt, level=corrupt_level)
                elif augmentation == "jitter":
                    point_set_corrupt = corrupt_jitter(point_set_corrupt, level=corrupt_level)
                else:
                    raise NotImplementedError

                source_output_path = os.path.join(parts_dict[part], "partsegmentation/sources")
                target_output_path = os.path.join(parts_dict[part], "partsegmentation/targets")
                file_source_output_path = os.path.join(parts_dict[part], "partsegmentation/sources", file.split("/")[1] + "-" + file.split("/")[2] + "-" + augmentation + ".npy")
                file_target_output_path = os.path.join(parts_dict[part], "partsegmentation/targets", file.split("/")[1] + "-" + file.split("/")[2] + "-" + augmentation + ".npy")
                if os.path.exists(file_source_output_path):
                    print(file_source_output_path)

                if not os.path.exists(source_output_path):
                    os.makedirs(source_output_path)
                if not os.path.exists(target_output_path):
                    os.makedirs(target_output_path)

                np.save(file_source_output_path, point_set_corrupt)
                np.save(file_target_output_path, target)
                
                if part == "shuffled_train_file_list":
                    str = os.path.join("partsegmentation-" + parts_dict[part], "partsegmentation\sources", file.split("/")[1] + "-" + file.split("/")[2] + "-" + augmentation + ".npy").replace("\\", "/")
                    train_list.append(str)
                if part == "shuffled_test_file_list":
                    str = os.path.join(parts_dict[part], "partsegmentation\sources", file.split("/")[1] + "-" + file.split("/")[2] + "-" + augmentation + ".npy").replace("\\", "/")
                    test_list.append(str)
        print("{} / {} process successfully!!!".format(part.split('_')[0], "partsegmentation"))
    json.dump(train_list, open("partseg_train_list.json", 'w'))
    json.dump(test_list, open("partseg_test_list.json", 'w'))