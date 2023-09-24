import json
import math
import os
import random

import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class PairDataset(data.Dataset):
    def __init__(self, config):
        self.data_root = config.data_path
        self.subset = config.subset
        self.npoints = config.npoints

        if self.subset == "train":
            self.data_list_file = os.path.join(self.data_root, f'{self.subset}_list.json')
            print_log(f'[DATASET] Open file {self.data_list_file}', logger='Point-In-Context_Dataset')
            data_list = json.load(open(self.data_list_file))
            self.file_list = []
            for line in data_list:
                task = line.split('-', 1)[0]
                self.file_list.append({'task': task, 'file_path': line.split('-', 1)[1]})
            self.task_dict = {}
            category_dict = {}
            for idx, file in enumerate(self.file_list):
                task = file['task']
                if task == "partsegmentation":
                    file_name = file['file_path'].split('/')[-1]
                    category = file_name.split('-')[0]
                    if task not in self.task_dict:
                        category_dict[category] = [idx]
                        self.task_dict[task] = category_dict
                    else:
                        if category not in category_dict:
                            category_dict[category] = [idx]
                            self.task_dict[task] = category_dict
                        else:
                            category_dict[category].append(idx)
                            self.task_dict[task] = category_dict
                else:
                    if task not in self.task_dict:
                        self.task_dict[task] = [idx]
                    else:
                        self.task_dict[task].append(idx)
        else:
            self.data_list_file = os.path.join(self.data_root, f'{self.subset}_list.json')
            self.train_list_file = os.path.join(self.data_root, 'train_list.json')

            print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'Point-In-Context_Dataset')
            data_list = json.load(open(self.data_list_file))
            train_list = json.load(open(self.train_list_file))
            self.file_list = []
            self.train_list = []
            for line in data_list:
                task = line.split('/')[1]
                self.file_list.append({'task': task, 'file_path': line})
            for line in train_list:
                task = line.split('-', 1)[0]
                self.train_list.append({'task': task, 'file_path': line.split('-', 1)[1]})
            self.task_dict = {}
            category_dict = {}
            for idx, file in enumerate(self.train_list):
                task = file['task']
                if task == "partsegmentation":
                    file_name = file['file_path'].split('/')[-1]
                    category = file_name.split('-')[0]
                    if task not in self.task_dict:
                        category_dict[category] = [idx]
                        self.task_dict[task] = category_dict
                    else:
                        if category not in category_dict:
                            category_dict[category] = [idx]
                            self.task_dict[task] = category_dict
                        else:
                            category_dict[category].append(idx)
                            self.task_dict[task] = category_dict
                else:
                    if task not in self.task_dict:
                        self.task_dict[task] = [idx]
                    else:
                        self.task_dict[task].append(idx)
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'Point-In-Context_Dataset')

    def random_rotate_together(self, pointcloud1, pointcloud2, level=0):
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
        pointcloud1 = np.dot(pointcloud1, R)
        pointcloud2 = np.dot(pointcloud2, R)
        return pointcloud1, pointcloud2

    def random_rotate(self, pointcloud, level=0):
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

    def y_flip(self, pointcloud1, pointcloud2):
        angles = [0, 0, math.pi]
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
        pointcloud1 = np.dot(pointcloud1, R)
        pointcloud2 = np.dot(pointcloud2, R)
        return pointcloud1, pointcloud2

    def y_flip_single(self, pointcloud1):
        angles = [0, 0, math.pi]
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
        pointcloud1 = np.dot(pointcloud1, R)
        return pointcloud1

    def random_dropout_global(self, pointcloud, level=0):
        """
        Drop random points globally
        :param pointcloud: input point cloud
        :param level: severity level
        :return: corrupted point cloud
        """
        # drop_rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]
        drop_rate = [0.5, 0.75, 0.875, 0.9375, 0.96875][level]
        num_points = pointcloud.shape[0]
        # choice = random.sample(range(0, num_points), int(drop_rate * num_points))
        pointcloud[(1 - int(drop_rate * num_points)):, :] = 0
        return pointcloud

    def random_add_noise(self, pointcloud, level=0, sigma=0.2):
        """
        Randomly add noise data to point cloud
        :param pointcloud: input point cloud
        :param num_noise: number of noise points
        :return: corrupted point cloud
        """
        N, _ = pointcloud.shape
        num_noise = 100 * (level + 1)
        noise = np.clip(sigma * np.random.randn(num_noise, 3), -1, 1)
        idx = np.random.randint(0, N, num_noise)
        pointcloud[idx, :3] = pointcloud[idx, :3] + noise
        return pointcloud

    def __getitem__(self, idx):
        global target1, target2
        pointset1 = self.file_list[idx]
        task = pointset1['task']

        if self.subset == "test" and task == "registration":
            pointset1_pc = np.load(os.path.join(self.data_root, pointset1['file_path']))
        else:
            pointset1_pc = np.load(os.path.join(self.data_root, pointset1['file_path'])).astype(np.float32)

        if task == "partsegmentation":
            file_name = pointset1['file_path'].split('/')[-1]
            category = file_name.split('-')[0]
            category_dict = self.task_dict[task]
            pointset2_index = random.choice(category_dict[category])
        else:
            pointset2_index = random.choice(self.task_dict[task])

        if self.subset == "train":
            pointset2 = self.file_list[pointset2_index]
        else:
            pointset2 = self.train_list[pointset2_index]

        pointset2_pc = np.load(os.path.join(self.data_root, pointset2['file_path'])).astype(np.float32)
        corrupt_level = random.randint(0, 4)
        if task == "reconstruction":
            if self.subset == "train":
                target1 = pointset1_pc.copy()
                pointset1_pc = self.random_dropout_global(pointset1_pc, corrupt_level)
            elif self.subset == "test":
                target1 = np.load(os.path.join(self.data_root, pointset1['file_path'].replace("sources", "targets"))).astype(np.float32)
            target2 = pointset2_pc.copy()
            pointset2_pc = self.random_dropout_global(pointset2_pc, corrupt_level)
        elif task == "registration":
            if self.subset == "train":
                target1 = pointset1_pc.copy()
                target2 = pointset2_pc.copy()
                pointset1_pc, pointset2_pc = self.random_rotate_together(pointset1_pc, pointset2_pc, corrupt_level)
                target1, target2 = self.y_flip(target1, target2)
            elif self.subset == "test":
                rotation_matrix = pointset1_pc["rotation_matrix"]
                pointset1_pc = pointset1_pc["pointcloud"].astype(np.float32)
                target1 = np.load(os.path.join(self.data_root, pointset1['file_path'].replace("sources", "targets")[:-1] + 'y')).astype(np.float32)
                target2 = pointset2_pc.copy()
                pointset2_pc = np.dot(pointset2_pc, rotation_matrix)
                target2 = self.y_flip_single(target2)
        elif task == "denoising":
            if self.subset == "train":
                target1 = pointset1_pc.copy()
                pointset1_pc = self.random_add_noise(pointset1_pc, corrupt_level)
            elif self.subset == "test":
                target1 = np.load(os.path.join(self.data_root, pointset1['file_path'].replace("sources", "targets"))).astype(np.float32)
            target2 = pointset2_pc.copy()
            pointset2_pc = self.random_add_noise(pointset2_pc, corrupt_level)
        elif task == "partsegmentation":
            target1 = np.load(os.path.join(self.data_root, pointset1['file_path'].replace("sources", "targets"))).astype(np.float32)
            target2 = np.load(os.path.join(self.data_root, pointset2['file_path'].replace("sources", "targets"))).astype(np.float32)
        else:
            raise NotImplementedError()

        pointset1_pc = torch.from_numpy(pointset1_pc).float()
        pointset2_pc = torch.from_numpy(pointset2_pc).float()
        target1 = torch.from_numpy(target1).float()
        target2 = torch.from_numpy(target2).float()

        return pointset2_pc, pointset1_pc, target2, target1

    def __len__(self):
        return len(self.file_list)
