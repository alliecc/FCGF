import open3d as o3d  # prevent loading error

import sys
import logging
import json
import argparse
import numpy as np
from easydict import EasyDict as edict

import torch
from model import load_model

from lib.data_loaders import make_data_loader
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature
from lib.timer import AverageMeter, Timer

import MinkowskiEngine as ME

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])


def main(config):
    test_loader = make_data_loader(
        config, config.test_phase, 1, num_threads=config.test_num_thread, shuffle=False)

    num_feats = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=config.bn_momentum,
        conv1_kernel_size=config.conv1_kernel_size,
        normalize_feature=config.normalize_feature)
    checkpoint = torch.load(config.save_dir + '/checkpoint.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, reg_timer = Timer(), Timer(), Timer()

    test_iter = test_loader.__iter__()
    N = len(test_iter)
    n_gpu_failures = 0

    # downsample_voxel_size = 2 * config.voxel_size
    list_results_to_save = []
    for i in range(len(test_iter)):
        data_timer.tic()
        try:
            data_dict = test_iter.next()
        except ValueError:
            n_gpu_failures += 1
            logging.info(f"# Erroneous GPU Pair {n_gpu_failures}")
            continue
        data_timer.toc()
        xyz0, xyz1 = data_dict['pcd0'], data_dict['pcd1']
        T_gth = data_dict['T_gt']
        xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()
        #import pdb
        # pdb.set_trace()
        pcd0 = make_open3d_point_cloud(xyz0np)
        pcd1 = make_open3d_point_cloud(xyz1np)

        with torch.no_grad():
            feat_timer.tic()
            sinput0 = ME.SparseTensor(
                data_dict['sinput0_F'].to(device), coordinates=data_dict['sinput0_C'].to(device))
            F0 = model(sinput0).F.detach()
            sinput1 = ME.SparseTensor(
                data_dict['sinput1_F'].to(device), coordinates=data_dict['sinput1_C'].to(device))
            F1 = model(sinput1).F.detach()
            feat_timer.toc()

        # saving files to pkl
        print(i)
        dict_sample = {"pts_source": xyz0np,
                       "feat_source": F0.cpu().detach().numpy(),
                       "pts_target": xyz1np,
                       "feat_target": F1.cpu().detach().numpy()}

        list_results_to_save.append(dict_sample)

    import pickle
    path_results_to_save = "fcgf.results.pkl"
    print('Saving results to ', path_results_to_save)
    pickle.dump(list_results_to_save, open(path_results_to_save, 'wb'))
    print('Saved!')
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--test_phase', default='test', type=str)
    parser.add_argument('--test_num_thread', default=1, type=int)
    parser.add_argument('--kitti_root', type=str, default="/data/kitti/")
    args = parser.parse_args()
    config = json.load(open(args.save_dir + '/config.json', 'r'))
    config = edict(config)
    config.save_dir = args.save_dir
    config.test_phase = args.test_phase
    config.kitti_root = args.kitti_root
    config.kitti_odometry_root = args.kitti_root + '/dataset'
    config.test_num_thread = 1  # args.test_num_thread

    main(config)

# sh scripts/train_fcgf_kitti.sh --save_dir=outputs/Experiments/KITTIMapDataset-v0.3/HardestContrastiveLossTrainer/ResUNetBN2C/SGD-lr1e-1-e200-b2i1-modelnout32/2021-02-11_11-33-18/
