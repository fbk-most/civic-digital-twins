<!-- SPDX-License-Identifier: Apache-2.0 -->

# CDT Seminar — Axes in v0.12.0

Slide deck and companion code for the **Civic Digital Twins seminar on axes**,
the main addition of v0.12.0: multi-domain, typed axes, shaped indexes,
operators along named axes, reading results by name, and `@config`.

The running example is a small city air-quality model: three zones, seven
Euro classes, 24 hours and a 16 × 16 spatial grid, with an entry fee
graduated by Euro class and NOx dispersion by diffusion.

---

## Directory contents

| File | Description |
|------|-------------|
| `seminar.md` | Marp slide deck (primary deliverable) |
| `seminar_axes.py` | Runnable companion script — mirrors the slide sections |
| `fig_city.png` | Map of the city: zones on the grid, and the plant (generated) |
| `fig_zone_time.png` | Inflow by zone and hour, no fee vs fee (generated) |
| `fig_euro_class.png` | Fleet mix by Euro class, per zone (generated) |
| `fig_exposure.png` | Daily NOx exposure maps, no fee vs fee, and the reduction (generated) |
| `fig_config.png` | Exposure under three `DispersionModel.Config` choices (generated) |
| `fig_attribution.png` | Concentration at 08:00 — traffic vs industry (generated; backup, not used in the slides) |

All files marked *generated* are produced by `seminar_axes.py`.

---

## Prerequisites

### Python environment

The repository uses [uv](https://docs.astral.sh/uv/) for dependency
management; `uv run` resolves everything automatically.  `matplotlib` comes
with the dev dependencies:

```bash
uv sync --dev
```

`xarray` is optional: if it is installed, the script also demonstrates
`LabeledArray.to_xarray()`; otherwise that step is skipped.

### Marp CLI (for rendering slides to HTML / PDF)

[Marp CLI](https://github.com/marp-team/marp-cli) requires Node.js ≥ 18.

```bash
# Install globally
npm install -g @marp-team/marp-cli
marp --version

# ... or run it without installing
npx @marp-team/marp-cli --version
```

### VS Code live preview (optional)

Install the
[Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
extension, open `seminar.md`, and click the preview icon (top-right).

---

## Running the companion script

From the **repository root**:

```bash
uv run python docs/axis-seminar/seminar_axes.py
```

It runs the five sections (§1–§5), prints what each slide shows — axis
reprs, error messages, declared vs returned axis orders, KPIs — and writes
the six PNG figures into `docs/axis-seminar/`.  Runtime: a few seconds.

| Section | Slides | Prints / produces |
|---------|--------|-------------------|
| §1 Defining axes | Part 1 | Axis reprs; `(name, role)` identity |
| §2 Indexes with a shape | Parts 0, 2 | `fig_city`; `axes=` on values vs formulas (inferred axes, verification); shape classes; shaped `DistributionIndex` |
| §3 Computing along axes | Part 3 | `sum()` without `axis=`; `AxesInferenceWarning`; `.broadcast()` and its size-1 pitfall; registered functions and `axis=-1` |
| §4 Reading results | Parts 1, 4 | `layout` vs `layout_of`; a `trip length` sweep showing all three axis roles (§4f); declared vs returned order; `.sel()`; KPIs; `fig_zone_time`, `fig_euro_class`, `fig_exposure`, `fig_attribution` |
| §5 `@config` | Part 5 | `default_config()`; three configs; `ConfigTypeMismatchError`; `fig_config` |

---

## Keeping slides and script in sync

The ensembles use fixed seeds (`np.random.default_rng(42)` and `(0)`), so
every run prints the same numbers.  If the model or its data change, rerun
the script and update the slides that quote its output:

- **"Results: inflow by zone and fleet mix"** — total NOx and its 5th–95th
  percentile range (`KPIs` block of §4).
- **"Results: exposure across the city"** — peak exposure (`KPIs` block of §4).
- **"The ordering trap"** — declared vs returned dims and the `raw[0]` values (§4b, §4d).
- **"`.broadcast()`"** and **"Registered functions"** — the printed outputs of §3c and §3d.
- **"`result.layout` vs `result.layout_of(index)`"** — the layout table (§4a).
- **"Axes and their roles"** — the `res_sweep.layout` table (§4f).

Code on the slides is copied from the script; when you change a model in
`seminar_axes.py`, update the matching slide.  Slides never show `import`
statements.

---

## Rendering the slides

Run from `docs/axis-seminar/` (or pass the paths explicitly).

### HTML

```bash
marp seminar.md --html --allow-local-files --output seminar.html
```

### PDF

```bash
marp seminar.md --pdf --allow-local-files --output seminar.pdf
```

PDF and image export need a headless Chrome/Chromium (Marp finds a local
Chrome automatically; otherwise set `CHROME_PATH`).

### PowerPoint (PPTX)

```bash
marp seminar.md --pptx --allow-local-files --output seminar.pptx
```

### Slide images (for checking overflow)

```bash
marp seminar.md --images png --allow-local-files --output /tmp/slides/slide.png
```

---

## Slide deck structure

| # | Part | Slides | Key content |
|---|------|--------|-------------|
| — | Introduction | 2 | Title; what changes in v0.12.0, outline |
| 0 | Running example | 3 | City, zones, grid; pipeline; why time alone is not enough |
| 1 | Defining axes | 3 | Roles and the layout rule; `DomainAxis` and the type lattice; `(name, role)` identity |
| 2 | Indexes with a shape | 6 | `axes=` on values (declared); formulas (inferred, then checked); `FIXED_AXES` shape classes; the shapes of every index in the example; structural contract check; shaped `DistributionIndex` |
| 3 | Computing along axes | 7 | Named reductions; outer products and `AxesInferenceWarning`; zones → grid; `shift`/`cumulative`; `laplacian` and boundaries; `.broadcast()`; registered functions |
| 4 | Reading results | 5 | `layout` vs `layout_of`; the ordering trap; `labeled()`/`.sel()`; results (inflow, fleet mix, exposure) |
| 5 | `@config` | 2 | `Config` vs `Inputs`; three configs side by side |
| 6 | Wrap-up | 5 | Summary; migration checklist v0.11 → v0.12; roadmap; named shapes — the mechanism needs tuning; Twindex preview |
| — | Appendix | 1 | bdt-mobility review notes |
| — | Closing | 1 | Thank you |

Plus one section-header slide per part.

---

## Notes

- **Slide prose is single long lines** — keep each paragraph on one line in
  `seminar.md`; wrapped lines render as narrower text.
- **Technical slides** — marked with a "⚙ technical" badge via
  `<!-- _class: technical -->` (combinable: `dense technical`); they can be
  skipped for a non-developer audience.
- **Dense slides** — slides with a lot of code or bullets use
  `<!-- _class: dense -->` (smaller font, defined in the front-matter style).
  Render to images after editing to check nothing overflows.
- **The model is illustrative** — synthetic data, a toy diffusion scheme;
  the numbers show how the framework behaves, not real air quality.
- **Shape classes are hand-written** — the seminar does not use
  `named_shape()`: Pyright rejects its results as `Inputs`/`Outputs`
  annotations.  The last Wrap-up slide discusses why the mechanism needs
  tuning.
