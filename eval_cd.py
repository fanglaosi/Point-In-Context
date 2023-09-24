import math
import time
import numpy as np
import torch
import argparse
from pathlib import Path
from utils.logger import *
from utils.config import *
from utils.misc import *
import os
from tools import builder
from torch.utils.data import Dataset
import json
from pytorch3d.ops import sample_farthest_points, knn_points

class PairDataset_final(Dataset):
    def __init__(self, args, eval_task, corrupt_level):
        self.data_root = args.data_path
        self.corrupt_level = corrupt_level - 1

        self.data_list_file = os.path.join(self.data_root, 'test_list.json')
        self.train_list_file = os.path.join(self.data_root, 'train_list.json')

        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'Point_Painter_Dataset')
        data_list = json.load(open(self.data_list_file))
        train_list = json.load(open(self.train_list_file))
        self.file_list = []
        self.train_list = []
        for line in data_list:
            task = line.split('/')[1]
            level = line.split('/')[3]
            if task == eval_task and int(level[5]) == corrupt_level:
                self.file_list.append({'task': task, 'file_path': line})
        for line in train_list:
            task = line.split('-', 1)[0]
            if task == eval_task:
                self.train_list.append({'task': task, 'file_path': line.split('-', 1)[1]})
        self.task_dict = {}
        for idx, file in enumerate(self.train_list):
            task = file['task']
            if task not in self.task_dict:
                self.task_dict[task] = [idx]
            else:
                self.task_dict[task].append(idx)
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'Point_Painter_Dataset')
        # print(len(self.file_list))

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
        pointset1 = self.file_list[idx]
        task = pointset1['task']

        if task == "registration":
            pointset1_pc = np.load(os.path.join(self.data_root, pointset1['file_path']))
        else:
            pointset1_pc = np.load(os.path.join(self.data_root, pointset1['file_path'])).astype(np.float32)

        pointset2_index = random.choice(self.task_dict[task])

        pointset2 = self.train_list[pointset2_index]

        pointset2_pc = np.load(os.path.join(self.data_root, pointset2['file_path'])).astype(np.float32)
        corrupt_level = self.corrupt_level
        # print(corrupt_level)
        # print(task)
        if task == "reconstruction":
            target1 = np.load(os.path.join(self.data_root, pointset1['file_path'].replace("sources", "targets"))).astype(np.float32)
            target2 = pointset2_pc.copy()
            pointset2_pc = self.random_dropout_global(pointset2_pc, corrupt_level)
        elif task == "registration":
            rotation_matrix = pointset1_pc["rotation_matrix"]
            pointset1_pc = pointset1_pc["pointcloud"].astype(np.float32)
            target1 = np.load(os.path.join(self.data_root, pointset1['file_path'].replace("sources", "targets")[:-1] + 'y')).astype(np.float32)
            target2 = pointset2_pc.copy()
            pointset2_pc = np.dot(pointset2_pc, rotation_matrix)
            target2 = self.y_flip_single(target2)
        elif task == "denoising":
            target1 = np.load(os.path.join(self.data_root, pointset1['file_path'].replace("sources", "targets"))).astype(np.float32)
            target2 = pointset2_pc.copy()
            pointset2_pc = self.random_add_noise(pointset2_pc, corrupt_level)
        else:
            raise NotImplementedError()

        pointset1_pc = torch.from_numpy(pointset1_pc).float()
        pointset2_pc = torch.from_numpy(pointset2_pc).float()
        target1 = torch.from_numpy(target1).float()
        target2 = torch.from_numpy(target2).float()

        return pointset2_pc, pointset1_pc, target2, target1

    def __len__(self):
        return len(self.file_list)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    # some args
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--loss', type=str, default='cd2', help='loss name')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')

    # dataset
    parser.add_argument('--data_path', type=str, default='data', help='')
    # comment
    parser.add_argument('--comment', type=str, default='default', help='')

    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help = 'autoresume training (interrupted by accident)')

    args = parser.parse_args()

    args.experiment_path = args.exp_name

    args.log_name = Path(args.config).stem
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)

    return args

def eval_cd(args, config, base_model, eval_task, corrupt_level):
    test_dataset = PairDataset_final(args, eval_task, corrupt_level)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.total_bs * 2,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=int(args.num_workers),
                                                  worker_init_fn=worker_init_fn)
    loss = eval(base_model, test_dataloader)
    return loss

def eval(base_model, test_dataloader):
    base_model.eval()  # set model to eval mode
    mean_loss = 0
    i = 0
    with torch.no_grad():
        for idx, (pointset1_pc, pointset2_pc, target1, target2) in enumerate(test_dataloader):
            pointset1_pc = pointset1_pc.cuda()
            pointset2_pc = pointset2_pc.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()

            _, rebuild_points, loss = base_model(pointset1_pc, pointset2_pc, target1, target2)
            rebuild_points, _ = sample_farthest_points(rebuild_points, K=target2.shape[1])
            loss = base_model.loss_func(rebuild_points, target2)
            loss = loss.mean()
            mean_loss += loss.item()*1000
            i += 1
        mean_loss /= i
    return mean_loss

def main():
    # args
    args = get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}-{args.seed}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # config
    config = get_config(args, logger = logger)
    # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)

    print_log(args.comment)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)  # seed + rank, for augmentation

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    tasks = ["reconstruction", "denoising", "registration"]
    for eval_task in tasks:
        task_loss = 0.
        print_log("------------------[{}] Tester start !!!!!!!!----------------- ".format(eval_task.upper()), logger=logger)
        for corrupt_level in range(5):
            loss = eval_cd(args, config, base_model, eval_task, corrupt_level+1)
            print_log('[TEST] corrupt_level: {} loss = {:.4f}'.format(corrupt_level+1, loss), logger=logger)
            task_loss += loss
        task_loss /= 5
        print_log('[TEST {}] loss = {:.4f}'.format(eval_task.upper(), task_loss), logger=logger)
    print_log("Done!!!!!!!!!!!!!!!!!")

if __name__ == "__main__":
    main()
