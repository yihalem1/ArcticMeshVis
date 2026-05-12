# ArcticMeshVis

**High-fidelity hand-object mesh reconstruction and visualization on the Arctic dataset.**

<p align="center">
  <img src="assets/hand_object.jpg" width="80%" alt="Hand-object interaction"/>
</p>

## Overview

ArcticMeshVis is a research project (developed Feb–Jul 2023) that pushes the boundaries of mesh reconstruction and visualization for **hand-object manipulation**. It combines:

- The **FastInst** instance segmentation architecture for fast, accurate object/hand masking.
- The **MANO** parametric hand model for articulated 3D hand reconstruction.
- An **Analytical Inverse Kinematics (AIK)** module for recovering joint rotations from predicted 3D keypoints.
- A **contact map** module that estimates hand–object contact regions from reconstructed meshes.

The pipeline is tailored to the Arctic/H2O datasets and produces evaluable 3D meshes suitable for downstream rendering and analysis.

## Highlights

| | |
|---|---|
| ![3D mesh visualization](assets/mesh_3d.jpg) | **Advanced Mesh Reconstruction** — reconstructs detailed hand and object meshes from RGB input, preserving fine surface geometry. |
| ![FastInst segmentation](assets/wireframe.jpg) | **FastInst-based Segmentation** — efficient instance segmentation backbone tuned for cluttered hand-object scenes. |
| ![Robotics grasping](assets/robotics_grip.jpg) | **Contact Map Estimation** — computes pseudo contact maps via nearest-neighbor queries between hand and object vertices for grasp analysis. |
| ![VR interaction](assets/vr_interaction.jpg) | **Hand-Object Manipulation** — outputs full 3D pose + shape, ready for VR, robotics, and ergonomics applications. |

## Repository Layout

```
ArcticMeshVis/
├── AIK/                  # Analytical inverse kinematics (torch)
├── configs/              # FastInst configs (COCO / instance-seg)
├── demo/                 # Inference demo entry points
│   ├── demo.py
│   └── predictor.py
├── fastinst/             # FastInst model + GraFormer / PoseFormer
├── mano/                 # MANO hand model wrapper
├── contact_map.py        # Hand-object contact map computation
├── engine.py             # Training / evaluation engine
├── INSTALL.md
└── ugcv_environment.yml  # Conda environment spec
```

## Applications

- **Robotics** — realistic grasp synthesis and manipulation policies.
- **Virtual / Augmented Reality** — accurate hand tracking and object interaction.
- **Ergonomic Design** — quantitative analysis of how products are held and used.
- **HCI Research** — fine-grained study of bimanual and hand-object behaviors.

## Getting Started

Clone the repository and follow [INSTALL.md](INSTALL.md):

```bash
git clone https://github.com/yihalem1/ArcticMeshVis.git
cd ArcticMeshVis
conda env create -f ugcv_environment.yml
conda activate fastinst
```

Run the segmentation demo:

```bash
python demo/demo.py \
  --config-file configs/coco/instance-segmentation/fastinst_R50_ppm-fpn_x1_576.yaml \
  --input path/to/images/*.jpg \
  --output outputs/ \
  --confidence-threshold 0.5
```

Compute contact maps between reconstructed hand and object meshes:

```bash
python contact_map.py
```

## Acknowledgements

This project builds on:
- [FastInst](https://github.com/junjiehe96/FastInst) (instance segmentation)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [MANO](https://mano.is.tue.mpg.de/) hand model
- The [ARCTIC](https://arctic.is.tue.mpg.de/) and [H2O](https://taeinkwon.com/projects/h2o/) datasets

## License

See [LICENSE](LICENSE).

---

> *Images above are illustrative stock photos used for showcase purposes; actual reconstructions on the Arctic/H2O datasets are produced by running the pipeline locally.*
