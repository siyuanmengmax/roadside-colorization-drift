# PointPillars on the CARLA testbed

The supervised baseline uses [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) with a custom dataset class.

1. Install OpenPCDet and PyTorch 2.3.
2. Copy `pcdet_datasets/*.py` to `OpenPCDet/pcdet/datasets/custom/`. Register `CarlaDataset` in `OpenPCDet/pcdet/datasets/__init__.py`:
   ```python
   from .custom.carla_dataset import CarlaDataset
   __all__['CarlaDataset'] = CarlaDataset
   ```
3. Copy `cfgs/` into `OpenPCDet/tools/cfgs/`.
4. Convert the CARLA frames with `python sim_studies/t1/convert.py --condition clear` (repeat for `fog`). This writes `data/carla_pcdet/<condition>`. Set `DATA_PATH` in `cfgs/dataset_configs/carla_dataset.yaml` to that directory.
5. Train with the OpenPCDet tools:
   ```
   python train.py --cfg_file cfgs/carla_models/pointpillar_carla.yaml
   ```
   Then evaluate with `sim_studies/t1/eval_carla.py`.
6. For the zero-shot TUMTraf-I row:
   - convert the real data with `scripts/tumtraf_to_pcdet.py`;
   - run `scripts/eval_pp_zeroshot_tumtraf.py` with the clear-condition checkpoint.
