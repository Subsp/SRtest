# SP-IE-SRGS v0 Server Run

This is the first runnable routed baseline. It uses the simple 3DGS /
mip-splatting training stack, not the huge `hybrid_sdfgs/train.py` loop.

## Expected Assets

Default kitchen layout:

```text
/root/autodl-tmp/kitchen/images_2
/root/autodl-tmp/kitchen/images_8
/root/autodl-tmp/kitchen/sparse/0
/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/qwen_steps1_seed42_rcgm_aligned_images2_train244_v0/fused_priors
/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/qwen_steps1_seed42_rcgm_aligned_images2_train244_v0/usable_masks
```

## Smoke Run

```bash
cd /root/autodl-tmp/SP-IE-SRGS
ITERATIONS=200 \
TEST_ITERATIONS="200" \
SAVE_ITERATIONS="200" \
SP_SURFACE_ENABLE=0 \
REQUIRE_PRIOR_MASK=0 \
bash scripts/run_sp_ie_srgs_v0_kitchen.sh
```

## Full Default Run

```bash
cd /root/autodl-tmp/SP-IE-SRGS
bash scripts/run_sp_ie_srgs_v0_kitchen.sh
```

Useful overrides:

```bash
SR_PRIOR_NAME=qwen_steps1_seed42_rcgm_aligned_images2_train244_v0
PREPARED_SR_PRIOR_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets/prepared_sr_priors/${SR_PRIOR_NAME}
LR_ANCHOR_DIR=/root/autodl-tmp/kitchen/images_8
HR_IMAGES_SUBDIR=images_2
LLFFHOLD=8
RUN_NAME=sp_v0_qwen_routed_smoke
```

The preflight uses the same LLFF train split as `--eval` by default, so a
`train244` prior is checked against train views rather than all `images_2`
frames.

## Surface Variant

Run this only after `sp_routing_only` smoke is healthy:

```bash
cd /root/autodl-tmp/SP-IE-SRGS
RUN_NAME=sp_v0_qwen_routing_surface \
SP_SURFACE_ENABLE=1 \
SP_LAMBDA_SURFACE=1.0 \
SP_LAMBDA_DISTORTION=1000.0 \
SP_LAMBDA_DEPTH_NORMAL=0.05 \
SP_LAMBDA_NORMAL_SMOOTH=0.01 \
SP_SURFACE_RAMP_START_ITER=1000 \
SP_SURFACE_RAMP_END_ITER=5000 \
bash scripts/run_sp_ie_srgs_v0_kitchen.sh
```

The no-densification-route ablation is explicit:

```bash
RUN_NAME=sp_v0_qwen_routing_surface_no_densify_route \
SP_SURFACE_ENABLE=1 \
SP_DISABLE_DENSIFICATION_ROUTE=1 \
bash scripts/run_sp_ie_srgs_v0_kitchen.sh
```

## What This Runs

Each iteration:

```text
geometry route:
  render HR camera -> downsample to LR anchor -> update xyz/scaling/rotation/opacity
  densification statistics come only from this route
  optional surface loss uses ramped distortion/depth-normal/normal-smooth terms

appearance route:
  render HR camera -> match prepared SR prior -> update f_dc/f_rest
  optional small LR consistency updates appearance only

optimizer:
  one Adam step after routed gradients are merged
```

Outputs:

```text
${MODEL_DIR}/sp_preflight.json
${MODEL_DIR}/sp_route_audit.jsonl
${MODEL_DIR}/cfg_args
${MODEL_DIR}/point_cloud/iteration_*/point_cloud.ply
```

For v0, NPSE cache, 2DGS carriers, and residual-tetris outputs are not required
for the first smoke. Geometry evidence should still be reported from
`sp_route_audit.jsonl` surface/proxy metrics when surface is enabled, and later
from Chamfer/F-score on a geometry-GT scene.
