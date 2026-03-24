# CDT Modelling Seminar — Bologna Mobility

Slide deck and companion code for the **Civic Digital Twins modelling seminar**,
illustrating IO Contracts, Modularity, and Model Variants through the Bologna
city-centre road pricing (ZTL) case study.

---

## Directory contents

| File | Description |
|------|-------------|
| `seminar.md` | Marp slide deck (primary deliverable) |
| `seminar_bologna.py` | Runnable companion script — mirrors the slide sections |
| `kpi_inflow.png` | Bar chart of inflow KPI means (generated) |
| `kpi_uncertainty.png` | KPI uncertainty range chart, 5th–95th pct (generated) |
| `kpi_traffic_ts.png` | Circulating-traffic timeseries with ensemble spread (generated) |
| `kpi_emissions_ts.png` | NOx-emissions timeseries with ensemble spread (generated) |
| `kpi_summary.txt` | Slide-ready KPI stats table — copy numbers from here into the slide (generated) |

All files marked *generated* are produced by `seminar_bologna.py` and are **not**
committed to the repository.  Run the script once before presenting to refresh them.

---

## Prerequisites

### Python environment

The repository uses [uv](https://docs.astral.sh/uv/) for dependency management.
No manual `pip install` is needed — `uv run` resolves everything automatically.

```bash
# Verify uv is available
uv --version
```

### Marp CLI (for rendering slides to HTML / PDF)

[Marp CLI](https://github.com/marp-team/marp-cli) requires Node.js ≥ 18.

```bash
# Install Node.js via your system package manager or https://nodejs.org

# Install Marp CLI globally
npm install -g @marp-team/marp-cli

# Verify
marp --version
```

Alternatively, install it locally inside the repo:

```bash
npm install --save-dev @marp-team/marp-cli
npx marp --version
```

### VS Code live preview (optional)

Install the
[Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
extension.  Open `seminar.md` and click the preview icon (top-right) for an
instant rendered view that updates as you edit.

---

## Running the companion script

The script must be run from the **repository root**:

```bash
uv run python docs/seminar/seminar_bologna.py
```

It executes all five sections (§1–§4b), prints a summary to stdout, and writes
the four PNG plots and `kpi_summary.txt` into `docs/seminar/`.

Expected runtime: ~30–60 s (200-scenario ensemble in §1, two 50-scenario
evaluations in §4b).

### What the script produces

| Output | Section | Content |
|--------|---------|---------|
| `kpi_inflow.png` | §1 | Bar chart of mean inflow KPIs |
| `kpi_uncertainty.png` | §1 | Range chart — mean + 5th/95th pct for all scalar KPIs |
| `kpi_traffic_ts.png` | §1 | Traffic timeseries heatmap (200-scenario ensemble) |
| `kpi_emissions_ts.png` | §1 | NOx timeseries heatmap (200-scenario ensemble) |
| `kpi_summary.txt` | §1 | Stats table formatted for the slide |

---

## Updating KPI numbers and plots before a presentation

The KPI numbers in the slides and all PNG files are generated from a stochastic
ensemble run and should be refreshed before each presentation to ensure they
reflect the current model.

1. **Run the script** from the repository root:

   ```bash
   uv run python docs/seminar/seminar_bologna.py
   ```

2. **Check `kpi_summary.txt`** — it contains a stats table in the same format
   as the slide:

   ```
   Bologna ZTL pricing — KPI summary
   200-scenario ensemble · cost_threshold ~ Uniform(4 €, 11 €)
   ==================================================================
   KPI                                   Mean        5th – 95th pct
   ──────────────────────────────────────────────────────────────────
   Base inflow                  168,139 veh/d                     —
   Modified inflow              ...
   ```

3. **Update the two KPI slides** in `seminar.md`:

   - **"KPI output — uncertainty matters"** (Part 1) — the Markdown table
     near the top of Part 1.
   - **"Bologna: what the model tells us"** (Part 5) — the ASCII art table
     near the bottom of the file.

   Copy the Mean and 5th–95th pct values directly.  The commentary line
   ("A ~17 % reduction … ~2× uncertainty range") may also need updating if
   the model parameters change.

4. **The PNG files are referenced by filename** — no slide edits needed for
   the images; they are picked up automatically once regenerated.

5. **Commit** the updated `seminar.md` and the four PNG files together.

---

## Rendering the slides

### HTML (for presenting from a browser)

```bash
marp docs/seminar/seminar.md --html --output docs/seminar/seminar.html
```

Open `seminar.html` in any modern browser.  Use arrow keys or click to
advance slides.  Full-screen with **F** (Chrome/Firefox) or the browser's
built-in full-screen shortcut.

### PDF

```bash
marp docs/seminar/seminar.md --pdf --output docs/seminar/seminar.pdf
```

PDF export requires a headless Chrome/Chromium or Puppeteer installation.
If `marp --pdf` fails, install Puppeteer:

```bash
npm install -g puppeteer
```

or use the `--allow-local-files` flag if images are not embedded:

```bash
marp docs/seminar/seminar.md --pdf --allow-local-files
```

### PowerPoint (PPTX)

```bash
marp docs/seminar/seminar.md --pptx --output docs/seminar/seminar.pptx
```

---

## Slide deck structure

| # | Part | Slides | Key content |
|---|------|--------|-------------|
| 0 | Motivating example | 3 | Bologna ZTL scenario, KPIs, pipeline overview |
| 1 | CDT Framework Basics | 7 | Indexes, DistributionIndex, simulation, KPI uncertainty, flat-model problem |
| 2 | IO Contracts | 7 | Inputs/Outputs dataclasses, InflowModel formulas, `__init__` pattern, three-level access, Expose, InputsContractWarning |
| 3 | Modularity | 7 | Constructor wiring, TrafficModel, EmissionsModel, BolognaModel root, KPI reading, full-picture diagram |
| 4 | Model Variants | 7 | Motivation, LinearTrafficModel vs SolverTrafficModel, registered functions, FtsTrafficModel, ModelVariant selector, contract enforcement, future dynamic selection |
| 5 | Putting it all together | 5 | KPI results with uncertainty ranges, timeseries plots, summary table, roadmap |

---

## Notes

- **Slide prose is single long lines** — Marp's default `breaks: true` converts
  every newline into a `<br>`.  Do not wrap prose across multiple lines inside
  `seminar.md`.
- **No `import` statements in slides** — imports appear only in the companion
  script.
- **Legend language** — `kpi_traffic_ts.png` and `kpi_emissions_ts.png` use
  Italian legend labels ("Riferimento" = baseline, "Modificato (mediana)" =
  modified-scenario mean) because they are generated by the shared Bologna
  plotting helper.  This will be addressed when issue #133 is resolved.
- **KPI numbers are stochastic** — each script run samples a fresh 200-scenario
  ensemble; numbers will vary slightly between runs.  This is expected and
  illustrates the framework's uncertainty propagation in action.