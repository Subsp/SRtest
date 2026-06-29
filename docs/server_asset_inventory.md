# Server Asset Inventory From Existing Scripts

This is an inspection of script assumptions, not a live server filesystem check.

## Core AutoDL Layout

Existing experiment scripts assume this server layout:

```text
/root/autodl-tmp/newsr
/root/autodl-tmp/HBSR
/root/autodl-tmp/mip-splatting
/root/autodl-tmp/kitchen
/root/autodl-tmp/kitchen/images_8
/root/autodl-tmp/kitchen/images_2
/root/autodl-tmp/kitchen/sparse/0
/root/autodl-tmp/kitchen/_hrgsrefiner_assets
/root/autodl-tmp/priors
/root/autodl-tmp/external
```

The IE-SRGS-style reproducer can run from a repo checkout and defaults to:

```text
SCENE_ROOT=/root/autodl-tmp/kitchen
INPUT_SUBDIR=images_8
GT_SUBDIR=images_2
REFERENCE_DIR=${SCENE_ROOT}/${INPUT_SUBDIR}
EXPERIMENT_ROOT=${HBSR_ROOT}/outputs/ie_srgs_repro/${SCENE_NAME}
```

## Timeline Correction

Do not read `SR_BACKEND=stablesr` in the April IE launcher as the current prior
source. The relevant timeline is:

```text
2026-03: FlashVSR / UAV / SwinIR generator experiments.
2026-04-25: direct_prior_detail_stronger_v1 local result archive.
2026-04-27: IE-SRGS-style script becomes an existing-prior adapter.
2026-06-14: SOF mainline narrows to NoSR cleanup plus prior-from-scratch /
            enhancement-SR prior wrappers.
2026-06-15: VGGTSR-mainline adds NPSE edge/trust cache and x1
            Restormer/NAFNet server setup.
2026-06-18: NPSE cache scripts assume edge_target/trust_edge outputs.
2026-06-21: SR-HF evidence and 2DGS carrier scripts become the active
            high-frequency asset path.
2026-06-23: SR-HF curve-track assets and Gaussian-layer spray are added.
2026-06-24..26: residual-tetris oracle, static, lockbox, and failure
                attribution diagnostics are the newest scripts.
```

So the reusable asset abstraction is not "StableSR prior". It is:

```text
prepared SR prior cache = fused_priors + usable_masks + aligned_references
```

## Existing-Prior Adapter

`19_run_ie_srgs_style_x8to2_scene.sh` is still useful, but only as an
existing-prior preparation pattern. Its default `SR_BACKEND=stablesr` is a
legacy naming label unless the server actually points `EXISTING_PRIOR_DIR` at
StableSR outputs.

The adapter takes an arbitrary existing prior directory:

```text
EXISTING_PRIOR_DIR=${SCENE_ROOT}/priors
```

and prepares it into:

```text
${EXPERIMENT_ROOT}/priors/${SR_BACKEND}_x8to2/priors
${EXPERIMENT_ROOT}/priors/${SR_BACKEND}_x8to2/fused_priors
${EXPERIMENT_ROOT}/priors/${SR_BACKEND}_x8to2/usable_masks
${EXPERIMENT_ROOT}/priors/${SR_BACKEND}_x8to2/manifest.json
```

Then training consumes:

```text
--external_prior_root ${PRIOR_ROOT}
--external_prior_subdir fused_priors
--external_prior_mask_subdir usable_masks
```

For SP-IE-SRGS v0, keep this adapter concept but do not hard-code StableSR.
`PRIOR_NAME`, `RAW_PRIOR_DIR`, and `PREPARED_SR_PRIOR_ROOT` should be explicit.

## SR Generator Assets

SwinIR/HAT generation scripts assume:

```text
/root/autodl-tmp/SwinIR
/root/autodl-tmp/HAT
/root/autodl-tmp/HAT/experiments/pretrained_models/HAT-L_SRx4_ImageNet-pretrain.pth
${HBSR_ROOT}/.venvs/sugar-system-py/bin/python
```

SwinIR/HAT outputs are written under:

```text
/root/autodl-tmp/priors/kitchen_swinir_x8to2_classical
/root/autodl-tmp/priors/kitchen_hat_x8to2_classical
```

with these subdirectories:

```text
priors
usable_masks
discrepancy
aligned_references
masked_priors
masked_references
fused_priors
```

FlashVSR scripts assume:

```text
/root/autodl-tmp/FlashVSR
/root/miniconda3/envs/flashvsr/bin/python
/root/autodl-tmp/hub/FlashVSR-v1.1
```

Typical FlashVSR outputs:

```text
/root/autodl-tmp/priors/kitchen_video_flashvsr_official_tiny_x8to2_93/priors
/root/autodl-tmp/priors/kitchen_video_flashvsr_official_tinylong_x8to2_93/priors
/root/autodl-tmp/priors/kitchen_video_flashvsr_seqmat_x8to2/priors
```

Upscale-A-Video scripts assume one of:

```text
/root/autodl-tmp/Upscale-A-Video
/root/autodl-tmp/HBSR/video_sr_models/Upscale-A-Video
/root/autodl-tmp/Upscale-A-Video/pretrained_models/upscale_a_video
/root/autodl-tmp/upscale_a_video
```

Typical UAV output:

```text
/root/autodl-tmp/priors/kitchen_video_uav_x8to2_chunked/priors
```

## SOF 2026-06-14 Mainline Assets

As of `SOF/MAINLINES_20260614.md`, the maintained prior path is the
enhancement-SR wrapper:

```text
SOF/scripts/run_mipsplatting_enhancement_prior_scratch_v0_kitchen.sh
```

It defaults to:

```text
WORK_ROOT=/root/autodl-tmp
SCENE_NAME=kitchen
SCENE_ROOT=${WORK_ROOT}/${SCENE_NAME}
SCENE_ASSET_ROOT=${SCENE_ROOT}/_hrgsrefiner_assets
SOURCE_IMAGES_SUBDIR=images_8
REFERENCE_IMAGES_SUBDIR=images_2
ENHANCEMENT_BACKEND=swinir
RAW_PRIOR_DIR=${SCENE_ROOT}/priors_${ENHANCEMENT_BACKEND}
PREPARED_SR_PRIOR_ROOT=${SCENE_ASSET_ROOT}/prepared_sr_priors/${ENHANCEMENT_BACKEND}_aligned_images_2_scratch_v0
SR_PRIOR_SUBDIR=fused_priors
SR_PRIOR_MASK_SUBDIR=usable_masks
SR_ANCHOR_SUBDIR=aligned_references
```

This is the better template for SP-v0 server assets than the older
IE-SRGS-style StableSR wording.

There is also a June VOSR/Qwen route in SOF scripts:

```text
RAW_PRIOR_DIR=${WORK_ROOT}/test_preds_1_vosr_same/qwen_steps1_seed42_rcgm
SOF/scripts/run_align_vosr_prior_size_v0_kitchen.sh
```

That route prepares:

```text
fused_priors
usable_masks
aligned_references
```

and prints:

```text
use SR_PRIOR_ROOT=${OUTPUT_ROOT}, SR_PRIOR_SUBDIR=fused_priors, SR_PRIOR_MASK_SUBDIR=usable_masks
```

## Newer VGGTSR-Mainline Assets

`VGGTSR-mainline/SOF` is newer than the standalone `SOF` checkout. Its git log
continues through 2026-06-26, and its server layout is documented in
`VGGTSR-mainline/SOF/SERVER_SETUP.md`:

```text
/root/autodl-tmp/newsr
/root/autodl-tmp/kitchen
/root/autodl-tmp/external/NAFNet
/root/autodl-tmp/external/Restormer
/root/autodl-tmp/external/GaussianImage
```

The default current prepared-prior name in these newer scripts is Qwen/VOSR,
not StableSR:

```text
SR_PRIOR_NAME=qwen_steps1_seed42_rcgm_aligned_images2_train244_v0
SR_DIR=${SCENE_ASSET_ROOT}/prepared_sr_priors/${SR_PRIOR_NAME}/fused_priors
```

### NPSE Edge/Trust Cache

The NPSE branch builds frame-aligned local targets from an LR render anchor, an
SR prior, and an aligned external depth prior:

```text
ANCHOR_DIR=${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0/train/ours_30000/test_preds_1
SR_DIR=${SCENE_ASSET_ROOT}/prepared_sr_priors/render_x1_restormer_aligned_images_2_scratch_v0/fused_priors
DEPTH_PRIOR_DIR=${SCENE_ASSET_ROOT}/depth_prior_aligned_gs2mesh/render_x1_depthprior_images_2_train_gs2mesh_aligned_v0/aligned_depth
OUTPUT_ROOT=${SCENE_ASSET_ROOT}/npse_cache/render_x1_restormer_depthprior_npse_v0
```

Later scripts often assume the inspected full cache:

```text
${SCENE_ASSET_ROOT}/npse_cache/render_x1_restormer_depthprior_npse_yellow_fidelity_nogate_full_v0
```

Useful subdirectories are:

```text
edge_target
trust_edge
continuous_target        # after materialize_npse_continuous_targets_v0.py
trust_continuous         # after materialize_npse_continuous_targets_v0.py
residual_npse
edge_type
debug_overlay
npz
```

`run_mipsplatting_nosr_layerfreq_cleanup_v0_kitchen.sh` already has switches to
consume these assets:

```text
PRIOR_EDGE_DIR
PRIOR_EDGE_MASK_DIR
PRIOR_EDGE_ANCHOR_DIR
LAMBDA_PRIOR_EDGE
LAMBDA_PRIOR_EDGE_SHAPE
PRIOR_EDGE_CONTRAST_WEIGHT
PRIOR_LOCAL_DIR
PRIOR_LOCAL_MASK_DIR
LAMBDA_PRIOR_LOCAL
LAMBDA_PRIOR_LOCAL_SURFACE
```

### SR-HF Evidence And 2DGS Carriers

The newer high-frequency route treats the prepared SR prior as input and builds
an explicit evidence cache:

```text
SR_DIR=${SCENE_ASSET_ROOT}/prepared_sr_priors/qwen_steps1_seed42_rcgm_aligned_images2_train244_v0/fused_priors
LR_DIR=${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0/train/ours_30000/test_preds_1
MASK_DIR=${SCENE_ASSET_ROOT}/npse_cache/render_x1_restormer_depthprior_npse_yellow_fidelity_nogate_full_v0/trust_edge
OUTPUT_ROOT=${SCENE_ASSET_ROOT}/sr_hf_evidence/<evidence_name>
```

Important outputs:

```text
effective_hf_carrier_rgb
effective_hf_weight
effective_hf_score
geometry_carrier_rgb
texture_carrier_rgb
primitives
primitive_overlay
```

Default evidence names vary across scripts
(`qwen_steps1_seed42_rcgm_aligned_images2_train244_v0_sr_hf_evidence_v0`,
`qwen_vosr_sr_hf_effective_8view_v0`,
`qwen_vosr_sr_hf_effective_verywide_8view_v0`), so server runs should pass
`EVIDENCE_ROOT` explicitly.

The 2DGS carrier route requires the external GaussianImage repo and writes
carrier assets:

```text
EXTERNAL_REPO_ROOT=/root/autodl-tmp/external/GaussianImage
TARGET_DIR=${EVIDENCE_ROOT}/effective_hf_carrier_rgb
MASK_DIR=${EVIDENCE_ROOT}/effective_hf_weight
ANCHOR_DIR=${SCENE_ASSET_ROOT}/kitchen_mip_vanilla_images8_v1/mip30k_rerun_check_directsrc_r1_v0/train/ours_30000/test_preds_1
OUTPUT_ROOT=${SOF_ROOT}/output/2dgs_sr_hf_evidence_carrier/<carrier_name>
```

Useful outputs:

```text
evidence_target
evidence_render
evidence_alpha
evidence_primitives
primitives
```

### Curve Tracks, Spray, And Residual Tetris

After evidence/carrier construction, the newest scripts are mostly offline
analysis or post-hoc injection:

```text
${SCENE_ASSET_ROOT}/sr_hf_curve_tracks/<track_name>/sr_hf_curve_tracks_v0.npz
${SOF_ROOT}/output/mipsplatting_sr_hf_curve_spray_v0/kitchen/<run_name>
${SOF_ROOT}/output/mipsplatting_2dgs_posterior_spray_v0/kitchen/<run_name>
${SOF_ROOT}/output/mipsplatting_2dgs_posterior_integrated_v0/kitchen/<run_name>
${SOF_ROOT}/output/residual_tetris_oracle_v0/<run_name>
${SOF_ROOT}/output/residual_tetris_static_v1/<run_name>
${SOF_ROOT}/output/residual_tetris_level1_lockbox_v0/<run_name>
${SOF_ROOT}/output/residual_tetris_failure_attribution_v2/<run_name>
/root/autodl-tmp/check/residual_tetris_*/<run_name>
```

These are valuable diagnostics for whether an SR high-frequency signal can be
represented on the Gaussian layer. They are not required for the first routed
SP-IE-SRGS training loop.

## Geometry And Mesh Assets

2DGS baseline scripts assume:

```text
/root/autodl-tmp/2d-gaussian-splatting
/root/autodl-tmp/kitchen/images_8
/root/autodl-tmp/kitchen/sparse/0
```

SuGaR scripts assume:

```text
/root/autodl-tmp/SuGaR
${HBSR_ROOT}/.venvs/sugar-system-py/bin/python
/root/autodl-tmp/kitchen/sparse/0
```

The helper creates a scene alias under:

```text
${HBSR_ROOT}/outputs/sugar_scene_aliases/${SCENE_NAME}_${IMAGES_SUBDIR}
```

and maps:

```text
images -> ${SCENE_ROOT}/${IMAGES_SUBDIR}
```

These assets are useful for geometry validation and mesh extraction, but they
are not required for the first SP routing sanity check.

## Recommended SP-v0 Use

Preferred kitchen sanity run should use explicit prepared-prior inputs:

```text
SCENE_ROOT=/root/autodl-tmp/kitchen
INPUT_SUBDIR=images_8
GT_SUBDIR=images_2
SCENE_ASSET_ROOT=/root/autodl-tmp/kitchen/_hrgsrefiner_assets
PREPARED_SR_PRIOR_ROOT=${SCENE_ASSET_ROOT}/prepared_sr_priors/<prior_name>
SR_PRIOR_SUBDIR=fused_priors
SR_PRIOR_MASK_SUBDIR=usable_masks
SR_ANCHOR_SUBDIR=aligned_references
```

SP-v0 should first check for already prepared current-mainline priors:

```text
${PREPARED_SR_PRIOR_ROOT}/manifest.json
${PREPARED_SR_PRIOR_ROOT}/${SR_PRIOR_SUBDIR}
${PREPARED_SR_PRIOR_ROOT}/${SR_PRIOR_MASK_SUBDIR}
${PREPARED_SR_PRIOR_ROOT}/${SR_ANCHOR_SUBDIR}
```

If valid, consume them directly. If missing, run the current preparation wrapper
for that prior source, for example:

```text
SOF/scripts/run_mipsplatting_enhancement_prior_scratch_v0_kitchen.sh
SOF/scripts/run_align_vosr_prior_size_v0_kitchen.sh
```

Use the April `19_run_ie_srgs_style_x8to2_scene.sh` only as a reference for
generic existing-prior preparation, not as evidence that current work uses
StableSR.

For SP-v0, the optional newer assets should be explicit knobs:

```text
NPSE_CACHE_ROOT=<optional>
PRIOR_EDGE_DIR=${NPSE_CACHE_ROOT}/edge_target
PRIOR_EDGE_MASK_DIR=${NPSE_CACHE_ROOT}/trust_edge
PRIOR_LOCAL_DIR=${NPSE_CACHE_ROOT}/continuous_target
PRIOR_LOCAL_MASK_DIR=${NPSE_CACHE_ROOT}/trust_continuous
EVIDENCE_ROOT=<optional SR-HF evidence cache>
CARRIER_ROOT=<optional 2DGS carrier cache>
```

Do not make the first SP-v0 route depend on:

```text
FlashVSR
Upscale-A-Video
SwinIR/HAT generation
SuGaR
2d-gaussian-splatting
GaussianImage
residual-tetris outputs
```

unless running ablations or geometry/mesh evaluation. The first SP route should
only require the scene, COLMAP sparse data, and a prepared SR prior cache. Add
NPSE edge/local assets as an optional second switch, and leave 2DGS carrier /
residual-tetris assets for diagnostics or later post-hoc injection experiments.
