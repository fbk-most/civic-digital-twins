<!-- SPDX-License-Identifier: Apache-2.0 -->

# `dt_model` package layout

`civic_digital_twins.dt_model` is organised in three layers — engine, model,
simulation. See the root [README](../../README.md#conceptual-overview) for
the conceptual overview and the [design docs](../../docs/design/) for the
full reference.

Besides the layer subpackages, this package hosts two top-level modules with
distinct, deliberate roles:

- **`dt_model.axes`** is the *canonical home* of the cross-cutting axis
  vocabulary: `Axis`, `AxisRole`, the role constants (`DOMAIN`, `PARAMETER`,
  `ENSEMBLE`), the `TIME_AXIS` singleton, and axis set operations.  Axes are
  shared by all three layers, so they live above them; every internal module
  imports axis symbols from here.  User code can equivalently use the
  top-level re-exports
  (`from civic_digital_twins.dt_model import Axis, TIME_AXIS`).
- **`dt_model.graph`** is a *curated user façade* over the engine-owned
  `engine.frontend.graph`: it re-exports only the node builders meant for
  use in model definitions.  Internal code imports the engine module
  directly — the façade is intentionally narrow and is not widened to serve
  internal needs.

A new top-level module should fit one of these two roles: canonical home for
cross-cutting vocabulary, or curated façade over a layer-owned module.
