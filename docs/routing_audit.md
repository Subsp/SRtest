# Routing Audit

The v0 audit must be treated as method evidence, not just debugging output.

Expected appearance-route audit:

```text
[ROUTE-AUDIT] appearance
  geometry: xyz=norm:0.000e+00 scaling=norm:0.000e+00 rotation=norm:0.000e+00 opacity=norm:0.000e+00
  appearance: f_dc=norm:... f_rest=norm:...
```

Expected densification audit:

```json
{
  "densify_grad_source": "geometry",
  "densify_consumed": 1,
  "mean_norm_xy": 0.0,
  "mean_norm_z": 0.0
}
```

Expected appearance viewspace audit:

```json
{
  "densify_grad_source": "appearance_unused",
  "densify_consumed": 0,
  "has_grad": 1
}
```

Required invariants:

```text
add_densification_stats is called once per iteration at most.
add_densification_stats is called only after geometry loss backward.
appearance-route viewspace gradients are never consumed for densification.
```

For the `sp_routing_surface_no_densify_route` ablation, geometry-route
densification consumption is disabled on purpose and should be visible as
`densify_consumed: 0`.
