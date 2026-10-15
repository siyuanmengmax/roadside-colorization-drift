import pickle
from pathlib import Path

import numpy as np

from ..dataset import DatasetTemplate


class CarlaDataset(DatasetTemplate):
    """Minimal OpenPCDet dataset for the CARLA roadside testbed.

    Layout (root = DATA_PATH):
        points/<idx>.npy   (N,4) float32 [x,y,z,intensity]  world frame
        labels/<idx>.txt   "x y z dx dy dz heading name" per line (world frame)
        ImageSets/{train,val,test}.txt
    No infos/gt_sampling database — labels are read directly per frame.
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=training, root_path=root_path, logger=logger)
        self.root_path = Path(root_path) if root_path is not None else Path(dataset_cfg.DATA_PATH)
        split_key = 'train' if training else 'test'
        self.split = dataset_cfg.DATA_SPLIT[split_key]
        self.sample_id_list = self._load_ids(self.split)
        if logger is not None:
            logger.info(f'CarlaDataset ({self.split}): {len(self.sample_id_list)} samples '
                        f'from {self.root_path}')

    def _load_ids(self, split):
        f = self.root_path / 'ImageSets' / f'{split}.txt'
        return [x.strip() for x in f.read_text().splitlines() if x.strip()]

    def set_split(self, split):
        self.split = split
        self.sample_id_list = self._load_ids(split)

    def __len__(self):
        return len(self.sample_id_list)

    def get_lidar(self, idx):
        return np.load(self.root_path / 'points' / f'{idx}.npy').astype(np.float32)

    def get_label(self, idx):
        txt = (self.root_path / 'labels' / f'{idx}.txt').read_text().strip()
        if not txt:
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=str)
        boxes, names = [], []
        for ln in txt.split('\n'):
            v = ln.split()
            boxes.append([float(x) for x in v[:7]])
            names.append(v[7])
        return np.array(boxes, dtype=np.float32), np.array(names)

    def __getitem__(self, index):
        idx = self.sample_id_list[index]
        points = self.get_lidar(idx)
        gt_boxes, gt_names = self.get_label(idx)
        if self.training and len(gt_boxes) == 0:
            return self.__getitem__(np.random.randint(self.__len__()))
        input_dict = {
            'points': points,
            'frame_id': idx,
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
        }
        return self.prepare_data(data_dict=input_dict)

    def evaluation(self, det_annos, class_names, **kwargs):
        """Dump predictions to a pkl (world-frame boxes). Scoring with the paper's
        BEV-distance 11-point AP is done by an external script for comparability."""
        out_dir = Path(kwargs.get('output_path') or self.root_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'preds_{self.split}.pkl'
        with open(out_file, 'wb') as f:
            pickle.dump(det_annos, f)
        n_box = int(sum(len(a['score']) for a in det_annos))
        return f'CarlaDataset: dumped {len(det_annos)} frames ({n_box} boxes) -> {out_file}', {}
