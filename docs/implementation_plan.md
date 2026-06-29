# SP-IE-SRGS v0 Implementation Plan

## Objective

Validate the smallest useful version of source-routed IE-SRGS:

```text
IE/SR prior must not directly update geometry or densification.
Geometry is controlled by LR consistency plus surface regularization.
```

v0 keeps the existing Gaussian schema. Do not add residual SH yet.

## Base Assets

- Base repo: `mip-splatting/hybrid_sdfgs`
- IE-style prior pipeline:
  - `hybrid_sdfgs/train.py`
  - `hybrid_sdfgs/tools/prepare_existing_sr_priors.py`
  - `hybrid_sdfgs/exp_scripts/19_run_ie_srgs_style_x8to2_scene.sh`
- Surface assets already present in this base:
  - `gaussian_renderer.render(..., return_aux=True)` / SOF rasterizer paths
  - `hybrid_sdfgs/blocks/sof_regularization_block.py`

## v0 Routing

Parameter routes:

```text
geometry:   xyz, scaling, rotation, opacity
appearance: f_dc, f_rest
```

Training step:

```text
1. Geometry route forward at target render resolution.
2. Downsample render to LR observation and compute LR loss.
3. Add ramped surface loss from target-resolution aux maps.
4. Backward geometry route.
5. Collect densification stats only from geometry-route viewspace grads.
6. Save geometry grads and clear grads.
7. Appearance route forward with geometry requires_grad disabled.
8. Compute SR/IE prior loss plus small LR consistency.
9. Backward appearance route.
10. Audit that geometry grad is zero/None for appearance route.
11. Restore routed grads into the single Gaussian optimizer.
12. Run exactly one optimizer step.
```

Default knobs:

```text
lambda_lr_app = 0.05
lambda_surface = 1.0 when --sp_surface_enable is set
lambda_distortion = 1000.0
lambda_depth_normal = 0.05
lambda_smoothness = 0.01
surface_ramp_start_iter = 1000
surface_ramp_end_iter = 5000
routing_step_mode = one_step
```

The first server smoke is `sp_routing_only` with surface disabled. The complete
v0 method is `sp_routing_surface`, enabled explicitly after the routing-only
audit passes.

## Ablations

Minimum table:

```text
vanilla_lr
ie_srgs_style_prior
ie_srgs_style_prior_surface_all_params
sp_routing_only
sp_routing_surface
sp_routing_surface_no_densify_route
```

## Success Criteria

v0 succeeds if:

```text
1. grad audit proves appearance route does not update geometry.
2. densification stats are consumed only after geometry backward.
3. SP-routing-only keeps image metrics close to IE-style prior.
4. SP-routing-surface improves depth/normal/distortion proxies.
5. At least one geometry-GT scene improves Chamfer/F-score.
```

Kitchen is only a sanity scene if no GT geometry is available.
