import numpy as np

from ltc.dataset.video_dataset import VideoDataset

import os
from os.path import join

from ltc.dataset.utils import load_segmentations
from ltc.dataset.utils import conform_temporal_sizes
import ltc.utils.logging as logging

import torch

import json

logger = logging.get_logger(__name__)


class Thumos(VideoDataset):
    def __init__(self, cfg, mode):
        super(Thumos, self).__init__(cfg, mode)
        logger.info("Constructing Thumos {} dataset with {} videos.".format(mode, self._dataset_size))

    def __getitem__(self, index: int):
        """

        :param index:
        :return: sample dict containing:
         'features': torch.Tensor [batch_size, input_dim_size, sequence_length]
         'targets': torch.Tensor [batch_size, sequence_length]
        """
        sample = {}
        targets = self._segmentations[index]
        sample['targets'] = torch.tensor(targets).long()[::self._video_sampling_rate]  # [T]

        feature_path = self._path_to_features[index]
        sample['features'] = torch.tensor(self._load_features(feature_path)).T  # [T, D]
        seq_length = sample['features'].shape[-1]
        sample['targets'] = conform_temporal_sizes(sample['targets'], seq_length)

        sample['video_name'] = self._video_names[index]

        return sample


    def _construct_loader(self):
        """
        Construct the list of features and segmentations.
        """

        # video_list_file = join(self._path_to_data,
        #                        "splits",
        #                        f"{self._mode}.split{self._cfg.DATA.CV_SPLIT_NUM}.bundle")
        # assert os.path.isfile(video_list_file), f"Video list file {video_list_file} not found."
        with open(self._path_to_data + r"/annotations/thumos14.json", 'r') as file:
            self.thumos = json.load(file)

        # with open(video_list_file, 'r') as f:
        #     list_of_videos = list(map(lambda l: l.strip(), f.readlines()))
        self._video_names = list(self.thumos['database'].keys())
        if self._mode == 'train':
            self._video_names = [s for s in self._video_names if 'validation' in s]
        else:
            self._video_names = [s for s in self._video_names if 'test' in s]

        # with open(join(self._path_to_data, "mapping.txt"), 'r') as f:
        #     lines = map(lambda l: l.strip().split(), f.readlines())
        #     action_to_idx = {action: int(str_idx) for str_idx, action in lines}
        action_to_idx = {'CricketBowling': 5, 'CricketShot': 6, 'VolleyballSpiking': 19, 'JavelinThrow': 12,
                         'Shotput': 15,
                         'TennisSwing': 17, 'GolfSwing': 9, 'ThrowDiscus': 18, 'Billiards': 2, 'CleanAndJerk': 3,
                         'LongJump': 13,
                         'Diving': 7, 'CliffDiving': 4, 'BasketballDunk': 1, 'HighJump': 11, 'HammerThrow': 10,
                         'SoccerPenalty': 16, 'BaseballPitch': 0, 'FrisbeeCatch': 8, 'PoleVault': 14,
                         'Background': 20}

        # num_videos = int(len(list_of_videos) * self._cfg.DATA.DATA_FRACTION)
        num_videos = int(len(self._video_names) * self._cfg.DATA.DATA_FRACTION)

        logger.info(f"Using {self._cfg.DATA.DATA_FRACTION * 100}% of {self._mode} data.")

        self._path_to_features = []
        self._segmentations = []
        # self._video_names = []

        # for gt_filename in list_of_videos[:num_videos]:
        #     video_id = os.path.splitext(gt_filename)[0]
        #     feat_filename = video_id + ".npy"
        #     feat_path = join(self._path_to_data, "features", feat_filename)
        #     assert os.path.isfile(feat_path), f"Feature {feat_path} not found."
        #     self._path_to_features.append(feat_path)
        #     gt_path = join(self._path_to_data, "groundTruth", gt_filename)
        #     self._segmentations.append(load_segmentations(gt_path, action_to_idx))
        #     self._video_names.append(video_id)

        for name in self._video_names[:num_videos]:
            feat_filename = name + '.npy'
            feat_path = join(self._path_to_data, "i3d_features", feat_filename)
            assert os.path.isfile(feat_path), f"Feature {feat_path} not found."
            self._path_to_features.append(feat_path)

            content = self.thumos['database'][name]
            # classes = np.ones()
            total_frames = content['fps'] * content['duration']
            classes_by_frame = np.ones(int(total_frames), dtype=int) * action_to_idx['Background']
            for anno in content['annotations']:
                id_begin = anno['segment(frames)'][0]
                id_end = anno['segment(frames)'][1]

                classes_by_frame[int(id_begin):int(id_end)] = int(anno['label_id'])

            self._segmentations.append(classes_by_frame)

        pass


'''
action_to_idx = {dict: 48} {'SIL': 0, 'add_saltnpepper': 12, 'add_teabag': 45, 'butter_pan': 17, 'crack_egg': 11, 'cut_bun': 36, 'cut_fruit': 32, 'cut_orange': 19, 'fry_egg': 13, 'fry_pancake': 29, 'peel_fruit': 34, 'pour_cereals': 1, 'pour_coffee': 5, 'pour_dough2pan': 28, 'pour_eg
f = {TextIOWrapper} <_io.TextIOWrapper name='D:\\MLdata\\data\\breakfast\\mapping.txt' mode='r' encoding='cp936'>
feat_filename = {str} 'P54_webcam02_P54_tea.npy'
feat_path = {str} 'D:\\MLdata\\data\\breakfast\\features\\P54_webcam02_P54_tea.npy'
gt_filename = {str} 'P54_webcam02_P54_tea.txt'
gt_path = {str} 'D:\\MLdata\\data\\breakfast\\groundTruth\\P54_webcam02_P54_tea.txt'
lines = {map} <map object at 0x000001BACC1142E0>
list_of_videos = {list: 1460} ['P16_cam01_P16_cereals.txt', 'P16_cam01_P16_friedegg.txt', 'P16_cam01_P16_juice.txt', 'P16_cam01_P16_milk.txt', 'P16_cam01_P16_pancake.txt', 'P16_cam01_P16_salat.txt', 'P16_cam01_P16_sandwich.txt', 'P16_cam01_P16_tea.txt', 'P16_stereo01_P16_cereals.txt', 
num_videos = {int} 1460

self.
    _cfg = {CfgNode: 14} CfgNode({'TRAIN': CfgNode({'ENABLE': True, 'DATASET': 'breakfast', 'BATCH_SIZE': 1, 'EVAL_PERIOD': 1, 'EVAL_BATCH_SIZE': 1, 'E...': 'D:\\dev\\LTContext\\output\\LTContext\\summary\\3', 'CONFIG_LOG_PATH': 'D:\\dev\\LTContext\\output\\LTContext\\config\\3'})
    _is_protocol = {bool} False
    _mode = {str} 'train'
    _path_to_data = {str} 'D:\\MLdata\\data\\breakfast\\'
    _path_to_features = {list: 1460} ['D:\\MLdata\\data\\breakfast\\features\\P16_cam01_P16_cereals.npy', 'D:\\MLdata\\data\\breakfast\\features\\P16_cam01_P16_friedegg.npy', 'D:\\MLdata\\data\\breakfast\\features\\P16_cam01_P16_juice.npy', 'D:\\MLdata\\data\\breakfast\\features\\P16_cam01_P1
    _segmentations = {list: 1460} [[0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1, 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1, 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1, 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    _video_meta = {dict: 0} {}
    _video_names = {list: 1460} ['P16_cam01_P16_cereals', 'P16_cam01_P16_friedegg', 'P16_cam01_P16_juice', 'P16_cam01_P16_milk', 'P16_cam01_P16_pancake', 'P16_cam01_P16_salat', 'P16_cam01_P16_sandwich', 'P16_cam01_P16_tea', 'P16_stereo01_P16_cereals', 'P16_stereo01_P16_coffee', 'P16_ster
    _video_sampling_rate = {int} 1

video_id = {str} 'P54_webcam02_P54_tea'
video_list_file = {str} 'D:\\MLdata\\data\\breakfast\\splits\\train.split1.bundle'
'''
