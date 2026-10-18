# roadside-colorization-drift

This repository holds the code for the paper:

> S. Meng and C. Ai, "Calibration Drift and Colorization Error in Roadside Camera–LiDAR Systems: A Saturating Law and Site-Specific Recalibration Budgets," submitted to *IEEE Transactions on Intelligent Transportation Systems*.

The paper measures the color error of a late-fused colorized LiDAR point cloud as the camera–LiDAR extrinsic drifts. It does so in a CARLA roadside testbed and on three public roadside datasets: TUMTraf-Intersection, V2X-Real, and DAIR-V2X-I.

The repository contains:

- the simulation setup;
- the scripts that rebuild every real-data experiment from the original dataset releases;
- the detection evaluation;
- the summary results (`outputs/*.json`) behind every number in the paper.

No dataset is redistributed here.

## Setup

```
conda env create -f environment.yml
conda activate rcd
```

### Data locations

Download the real datasets from their providers and accept their licences. Point the scripts to them with environment variables. The defaults are relative to the repository root.

| Dataset | Variable | Default | Used for |
|---|---|---|---|
| [TUMTraf-Intersection](https://innovation-mobility.com/en/project-providentia/a9-dataset/) (R02, sequences s01–s04) | `TUMTRAF_ROOT` | `data/tumtraf_i` | drift law, detection |
| [V2X-Real](https://mobility-lab.seas.ucla.edu/v2x-real/) (infrastructure agents) | `V2XREAL_ROOT` | `data/v2x_real` | drift law |
| [DAIR-V2X-I](https://thudair.baai.ac.cn/index) (single-infrastructure side) | `DAIR_ROOT` | `data/dair-v2x-i` | drift law |

Each adapter in `scripts/` states the directory layout it expects:

- `tumtraf_adapter.py`
- `v2xreal_adapter.py`
- `dair_adapter.py`

### Simulation (CARLA 0.9.15)

Start the CARLA server, then record the four weather conditions at Town10HD:

```
python collect_data.py --condition clear --cameras-360     # likewise rain_med, fog, night
```

Repeat with `--town Town05 --output-dir carla_data_town05 --condition clear` for the second intersection. Town10HD frames are written to `carla_data/<condition>/`, which is where the scripts look for them.

## Reproducing the paper

Run all commands from the repository root. Most scripts read and write `outputs/`.

### Calibration-drift law (Section V)

1. **Drift sweeps.** These give the drift-law and budget tables and the drift-law figure.
   ```
   python scripts/pixel_displacement_sweep.py --protocol unit --out drift_sweep_matched.json \
       --sets sim_clear sim_rain_med sim_fog sim_night sim_town05_clear real_tumtraf_day real_tumtraf_s04 real_v2xreal
   python scripts/dair_brightness.py
   python scripts/pixel_displacement_sweep.py --protocol unit --out drift_sweep_dair.json \
       --sets real_dair_W real_dair_Y1 real_dair_Y2 real_dair_Y3 real_dair_Ylow
   ```
2. **Fits, bootstrap intervals, and noise floor.**
   ```
   python scripts/drift_model_compare.py drift_sweep_matched.json drift_model_compare.json
   python scripts/drift_model_compare.py drift_sweep_dair.json drift_model_compare_dair.json
   python scripts/drift_law_bootstrap.py
   python scripts/drift_noise_floor.py
   python scripts/tumtraf_by_sequence.py
   ```
3. **Coupled rotation** (0.2° per cm).
   ```
   python scripts/pixel_displacement_sweep.py --protocol unit --out drift_sweep_matched.json \
       --deg_per_cm 0.2 --suffix _rot0.2 --sets sim_clear real_tumtraf_day real_tumtraf_s04 real_v2xreal real_dair_W real_dair_Y1
   ```
4. **Image statistic G and budget prediction** (Section V-D).
   ```
   python scripts/image_budget_predictor.py
   python scripts/budget_from_single_frame.py   # leave-one-group-out and single-frame prediction
   ```
5. **Figures.**
   - Drift-law figures (drift law, pixel displacement, Town05): `scripts/phase1_figure.py`
   - Qualitative frame: `scripts/qualitative_drift_fig.py`
   - Calibration overlay: `scripts/phase0_boxes.py`
   - Testbed schematic: `gen_testbed_fig.py`
   - Overview diagram (Fig. 1, TikZ): `figures/overview.tex` (build with `tectonic figures/overview.tex` or `pdflatex`)

### Detection (Section VI)

- **Background models for the simulated label-free protocol.**
  ```
  python sim_studies/t2/bg_subtract.py --condition clear
  ```
  Repeat for each condition. The models are written to `work/bg_models/`.
- **Simulated baselines and sensor-count ablation** (baseline and sensor-ablation tables, sensor-ablation figure).
  ```
  python scripts/sensor_ablation_ci.py
  python scripts/sensor_ablation_fig.py
  python scripts/weather_density_strata.py
  ```
- **Pedestrian point counts.**
  ```
  python sim_studies/t1/theme_a_stats.py
  ```
- **TUMTraf-I, label-free and oracle foreground** (real rows of the baseline table, quality-aware table).
  ```
  python scripts/tumtraf_detect_nonoracle.py --seq s03 --gt_filter
  python scripts/tumtraf_detect_nonoracle.py --seq s04 --gt_filter
  ```
- **Reliability statistics.** `python scripts/tumtraf_detect.py`
- **Sensitivity to the reliability scale τ.** Run the two TUMTraf commands above again with `--tau 5`, `--tau 10`, `--tau 40`, and `--tau 80`. Then run:
  ```
  python scripts/tau_sensitivity.py
  ```
- **Bootstrap block-length check** (supplementary Section S4). `python scripts/block_length_sensitivity.py`
- **Simulated quality-aware comparison.** `sim_studies/t4/t4_analysis.py` and `sim_studies/t5/t5_mechanism.py`
- **PointPillars** (supervised baseline and zero-shot TUMTraf-I row): see [`pointpillars/README.md`](pointpillars/README.md).

The detector components shared with MulDet3D are in `src/detection/` and `carla_detection.py`.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{meng2026drift,
  author = {Meng, Siyuan and Ai, Chengbo},
  title  = {Calibration Drift and Colorization Error in Roadside Camera--{LiDAR} Systems:
            A Saturating Law and Site-Specific Recalibration Budgets},
  note   = {Manuscript},
  year   = {2026}
}
```

If you also use the datasets, the simulator, or the detection components, please cite their sources:

```bibtex
@inproceedings{zimmer2023tumtraf,
  author    = {Zimmer, Walter and Cre{\ss}, Christian and Nguyen, Huu Tung and Knoll, Alois C.},
  title     = {{TUMTraf} Intersection Dataset: All You Need for Urban {3D} Camera-{LiDAR} Roadside Perception},
  booktitle = {2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)},
  pages     = {1030--1037},
  year      = {2023},
  doi       = {10.1109/ITSC57777.2023.10422289}
}

@inproceedings{xiang2024v2xreal,
  author    = {Xiang, Hao and Zheng, Zhaoliang and Xia, Xin and Xu, Runsheng and Gao, Letian and Zhou, Zewei and Han, Xu and Ji, Xinkai and Li, Mingxi and Meng, Zonglin and Jin, Li and Lei, Mingyue and Ma, Zhaoyang and He, Zihang and Ma, Haoxuan and Yuan, Yunshuang and Zhao, Yingqian and Ma, Jiaqi},
  title     = {{V2X-Real}: A Large-Scale Dataset for Vehicle-to-Everything Cooperative Perception},
  booktitle = {Computer Vision -- ECCV 2024},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  pages     = {455--470},
  year      = {2024},
  doi       = {10.1007/978-3-031-72943-0_26}
}

@inproceedings{yu2022dairv2x,
  author    = {Yu, Haibao and Luo, Yizhen and Shu, Mao and Huo, Yiyi and Yang, Zebang and Shi, Yifeng and Guo, Zhenglong and Li, Hanyu and Hu, Xing and Yuan, Jirui and Nie, Zaiqing},
  title     = {{DAIR-V2X}: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative {3D} Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {21329--21338},
  year      = {2022},
  doi       = {10.1109/CVPR52688.2022.02067}
}

@inproceedings{dosovitskiy2017carla,
  author    = {Dosovitskiy, Alexey and Ros, German and Codevilla, Felipe and Lopez, Antonio and Koltun, Vladlen},
  title     = {{CARLA}: An Open Urban Driving Simulator},
  booktitle = {Proceedings of the 1st Annual Conference on Robot Learning (CoRL)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {78},
  pages     = {1--16},
  year      = {2017}
}

@article{meng2026muldet3d,
  author  = {Meng, Siyuan and Zhang, Yanan and Raucci, David and Ai, Chengbo},
  title   = {{MulDet3D}: Multiobjective Optimization-Based Unsupervised Object Detection for Multiple Roadside {LiDARs}},
  journal = {Journal of Transportation Engineering, Part A: Systems},
  volume  = {152},
  number  = {10},
  pages   = {04026100},
  year    = {2026},
  doi     = {10.1061/JTEPBS.TEENG-9741}
}

@article{meng2027frgb3d,
  author  = {Meng, Siyuan and Parashar, Pravar and Yang, Yu-Min and Ai, Chengbo},
  title   = {{FRGB3D}: Fast Reliability-Weighted {Gaussian} Background Modeling for Roadside {LiDAR} Traffic Monitoring},
  journal = {Journal of Computing in Civil Engineering},
  volume  = {41},
  number  = {1},
  pages   = {04026129},
  year    = {2027},
  doi     = {10.1061/JCCEE5.CPENG-7710}
}

@inproceedings{lang2019pointpillars,
  author    = {Lang, Alex H. and Vora, Sourabh and Caesar, Holger and Zhou, Lubing and Yang, Jiong and Beijbom, Oscar},
  title     = {{PointPillars}: Fast Encoders for Object Detection from Point Clouds},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {12689--12697},
  year      = {2019},
  doi       = {10.1109/CVPR.2019.01298}
}

@misc{openpcdet2020,
  author       = {{OpenPCDet Development Team}},
  title        = {{OpenPCDet}: An Open-source Toolbox for {3D} Object Detection from Point Clouds},
  howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
  year         = {2020}
}
```

## Licence

The code is released under the MIT licence. The datasets remain under their own licences. Results derived from TUMTraf-Intersection are reported here only as aggregate statistics.
