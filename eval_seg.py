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


def random_add_noise(pointcloud, level=0, sigma=0.2):
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

class PairDataset_final(Dataset):
    def __init__(self, args, eval_task):
        self.data_root = args.data_path

        self.data_list_file = os.path.join(self.data_root, 'test_list.json')
        self.train_list_file = os.path.join(self.data_root, 'train_list.json')

        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'Point_Painter_Dataset')
        data_list = json.load(open(self.data_list_file))
        train_list = json.load(open(self.train_list_file))
        self.file_list = []
        self.train_list = []
        for line in data_list:
            task = line.split('/')[1]
            if task == eval_task:
                self.file_list.append({'task': task, 'file_path': line})
        for line in train_list:
            task = line.split('-', 1)[0]
            if task == eval_task:
                self.train_list.append({'task': task, 'file_path': line.split('-', 1)[1]})
        self.task_dict = {}
        category_dict = {}

        for idx, file in enumerate(self.train_list):
            task = file['task']
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
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'Point_Painter_Dataset')
        # print(len(self.file_list))

    def __getitem__(self, idx):
        pointset1 = self.file_list[idx]
        task = pointset1['task']

        pointset1_pc = np.load(os.path.join(self.data_root, pointset1['file_path'])).astype(np.float32)


        file_name = pointset1['file_path'].split('/')[-1]
        category = file_name.split('-')[0]
        category_dict = self.task_dict[task]
        pointset2_index = random.choice(category_dict[category])


        pointset2 = self.train_list[pointset2_index]

        pointset2_pc = np.load(os.path.join(self.data_root, pointset2['file_path'])).astype(np.float32)

        # add random noise, for rebuttal
        # pointset2_pc = random_add_noise(pointset2_pc, level=0)

        if task == "partsegmentation":
            target1 = np.load(
                os.path.join(self.data_root, pointset1['file_path'].replace("sources", "targets"))).astype(np.float32)
            target2 = np.load(
                os.path.join(self.data_root, pointset2['file_path'].replace("sources", "targets"))).astype(np.float32)
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

    parser.add_argument('--resume', action='store_true', default=False, help = 'autoresume training (interrupted by accident)')

    args = parser.parse_args()

    args.experiment_path = args.exp_name

    args.log_name = Path(args.config).stem
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)

    return args

def eval_cd(args, config, base_model, eval_task):
    test_dataset = PairDataset_final(args, eval_task)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.total_bs * 2,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=int(args.num_workers),
                                                  worker_init_fn=worker_init_fn)
    shape_avg_iou = eval(base_model, test_dataloader, config)
    return shape_avg_iou

def eval(base_model, test_dataloader, config):
    color_points_raw = np.load("data/color_map_final.npy")
    color_points_raw_torch = torch.from_numpy(color_points_raw).unsqueeze(0).unsqueeze(0).cuda() # [1, 1, 50, 3]


    base_model.eval()  # set model to eval mode
    count = 0.0
    shape_ious = 0.0

    with torch.no_grad():
        for idx, (pointset1_pc, pointset2_pc, target1, target2) in enumerate(test_dataloader):
            pointset1_pc = pointset1_pc.cuda()
            pointset2_pc = pointset2_pc.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()

            gt_points, rebuild_points, loss = base_model(pointset1_pc, pointset2_pc, target1, target2)
            rebuild_points = rebuild_points.unsqueeze(2) # [B, N, 1, 3]
            gt_points = gt_points.unsqueeze(2) # [B, N, 1, 3]

            rebuild_points = torch.sum((rebuild_points - color_points_raw_torch) ** 2, dim=-1) # [B, N, 50]
            gt_points = torch.sum((gt_points - color_points_raw_torch) ** 2, dim=-1) # [B, N, 50]
            gt_points = torch.min(gt_points, dim=-1, keepdim=False)[1] # [B, N]
            gt_points = gt_points.cpu().numpy().astype(np.int32)
            gt_points = torch.from_numpy(gt_points).cuda()
            batch_shapeious = compute_overall_iou(rebuild_points, gt_points, 50)  # [b]
            batch_ious = rebuild_points.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!
            shape_ious += batch_ious.item()  # count the sum of ious in each iteration

            count += config.total_bs * 2
        shape_avg_iou = shape_ious * 1.0 / count
    return shape_avg_iou

def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred = pred.min(dim=2)[1]    # (batch_size, num_points)  the pred_class_idx of each point in each sample
    pred_np = pred.cpu().data.numpy()

    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):   # sample_idx
        part_ious = []
        for part in range(num_classes):   # class_idx! no matter which category, only consider all part_classes of all categories, check all 50 classes
            # for target, each point has a class no matter which category owns this point! also 50 classes!!!
            # only return 1 when both belongs to this class, which means correct:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            # always return 1 when either is belongs to this class:
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))

            F = np.sum(target_np[shape_idx] == part)

            if F != 0:
                iou = I / float(U)    #  iou across all points for this class
                part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious))   # each time append an average iou across all classes of this sample (sample_level!)
    return shape_ious   # [batch_size]

def main():
    # args
    args = get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}-{args.seed}_partseg.log')
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

    task = "partsegmentation"

    print_log("------------------[{}] Tester start !!!!!!!!----------------- ".format(task.upper()), logger=logger)
    for i in range(5):
        shape_avg_iou = eval_cd(args, config, base_model, task)
        print_log('[TEST {}-{}] ins_iou = {:.4f}'.format(task.upper(), i, shape_avg_iou), logger=logger)

    print_log("Done!!!!!!!!!!!!!!!!!")

if __name__ == "__main__":
    main()
