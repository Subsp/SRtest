# Performance Checker Design

This directory defines a reproducible benchmark checker for comparing:

- the local SP-IE-SRGS / IE-SRGS reproduction;
- local Mip-Splatting;
- mesh-oriented baselines such as GS2Mesh;
- SuGaR-like surface Gaussian methods, including SuGaR, GOF / 2DGS-style
  methods, and the latest available 2D-SuGaR-style candidate when code and
  licenses allow it.

The checker is intentionally method-agnostic. Each method may keep its own
training, rendering, and mesh extraction code. The checker only enforces a
shared artifact contract and aggregates geometry and rendering metrics.

## Dataset Protocol

For the time-saving run, use a single-scene protocol:

- default scene set: `single_scene`;
- dataset / scene: DTU `scan24`;
- reason: one scene can exercise both geometry metrics (Chamfer-L1 from DTU
  STL ground truth) and rendering metrics (PSNR / SSIM from held-out RGB).

Keep the broader tracks below as expansion paths after the single-scene check
is stable. This avoids reporting Chamfer distance on datasets without real
geometry ground truth.

| Track | Dataset | Scenes | Metrics | Purpose |
| --- | --- | --- | --- | --- |
| Geometry + rendering core | DTU MVS | scans 24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122 | Chamfer-L1 / accuracy / completion, PSNR / SSIM | Object-centric geometry benchmark with structured-light STL ground truth. |
| Real-scene geometry supplement | Tanks and Temples | Truck, Train first; optionally intermediate scenes | F-score / precision / recall, optional Chamfer-style distances if locally computed | Real captured scenes with laser-scan ground truth and official alignment/eval tooling. |
| Rendering-only main table | Mip-NeRF 360 | bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai | PSNR / SSIM / LPIPS if available | Standard 3DGS and Mip-Splatting novel-view rendering protocol. |
| Rendering-only supplement | Deep Blending | drjohnson, playroom | PSNR / SSIM / LPIPS if available | Standard 3DGS supplement for indoor scenes. |

Default reporting:

- Single-scene table: DTU `scan24` Chamfer-L1 / accuracy / completion plus
  PSNR / SSIM.
- Optional expansion geometry table: DTU mean Chamfer-L1 plus per-scene values.
- Optional real geometry table: Tanks and Temples F-score for Truck and Train.
- Optional rendering table: Mip-NeRF 360 and Deep Blending mean PSNR / SSIM.

## Method Set

Required:

- `ie_srgs`: local `SP-IE-SRGS` reproduction.
- `mip_splatting`: local `mip-splatting`.
- `gs2mesh`: local `gs2mesh`, geometry-first.

Recommended surface-Gaussian baselines:

- `sugar`: SuGaR / surface-aligned Gaussian baseline.
- `gof`: Gaussian Opacity Fields, useful because local Mip-Splatting already
  notes its densification update.
- `two_dgs`: 2D Gaussian Splatting-style geometry baseline.
- `two_d_sugar_latest`: newest 2D-SuGaR-style method, disabled by default until
  code, license, and environment are pinned.

## Artifact Contract

For every `(method, dataset, scene)` run, the checker expects this layout by
default:

```text
benchmark_runs/
  <method_id>/
    <dataset_id>/
      <scene_id>/
        renders/                  # rendered test images, same stems as gt/
        gt/                       # copied or symlinked target images
        mesh/
          mesh.ply                # exported mesh or sampled point cloud
        metrics/
          render_metrics.json     # produced by checker.py render-metrics
          results.json            # produced by DTU/TNT eval wrappers
        manifest.json             # optional run metadata
```

The default paths can be overridden per method in
`benchmark_config.example.json`.

## Metric Definitions

Rendering:

- PSNR and SSIM are computed from paired RGB images.
- The checker uses `skimage.metrics.structural_similarity` when available.
  Otherwise it falls back to a global SSIM implementation so the layout can be
  smoke-tested without the full training environment.
- LPIPS is not computed by the standalone checker; it is collected if a method
  writes it in `render_metrics.json` or existing `results.json` files.

Geometry:

- DTU: use the self-contained `performance_checker/geometry_metrics.py` script
  against `DTU_OFFICIAL_ROOT/Points/stl/stl024_total.ply` for the single-scene
  run. It writes `accuracy`, `completion`, and `chamfer_l1`.
- Tanks and Temples: use the same self-contained geometry script if expanding
  beyond the single-scene DTU run. For formal T&T leaderboard-equivalent
  numbers, pin and document the official T&T toolbox separately.
- If a method writes `chamfer`, `chamfer_l1`, `accuracy`, `completion`,
  `precision`, `recall`, or `fscore`, the checker will normalize the keys.

## CLI

```bash
python performance_checker/checker.py plan \
  --config performance_checker/benchmark_config.example.json \
  --scene-set single_scene

python performance_checker/checker.py check-layout \
  --config performance_checker/benchmark_config.example.json \
  --scene-set single_scene

python performance_checker/checker.py render-metrics \
  --config performance_checker/benchmark_config.example.json \
  --method ie_srgs \
  --dataset dtu_core \
  --scene scan24

python performance_checker/checker.py collect \
  --config performance_checker/benchmark_config.example.json \
  --scene-set single_scene
```

Sync only the checker files to a remote machine:

```bash
REMOTE=user@server REMOTE_DIR=/path/to/SRtest \
  bash performance_checker/sync_single_scene_checker.sh
```

This sync script only copies `performance_checker/`; it does not read from or
write to any other repository.

## Run Discipline

1. Pin every external method to a commit hash in `manifest.json`.
2. Use identical train/test splits and image resolutions per dataset scene.
3. For DTU and T&T, run official culling/alignment before comparing geometry.
4. For Mip-Splatting and IE-SRGS, export a mesh before geometry evaluation.
5. Do not average render-only scenes into geometry means.
6. Report missing metrics explicitly instead of silently dropping a failed
   method/scene pair.

## Source Pointers

- Mip-Splatting project: https://niujinshuchong.github.io/mip-splatting/
- Gaussian Opacity Fields project: https://niujinshuchong.github.io/gaussian-opacity-fields/
- Mip-NeRF 360 dataset: https://jonbarron.info/mipnerf360/
- 3D Gaussian Splatting evaluation datasets: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Tanks and Temples benchmark: https://www.tanksandtemples.org/
- DTU Robot Image Data Sets: https://roboimagedata.compute.dtu.dk/
- 2D-SuGaR arXiv entry: https://arxiv.org/abs/2605.00569
