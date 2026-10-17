# Titration by a co-replicating reservoir

Solver, result tables and figure scripts for

> Tal, S. *Titration by a co-replicating reservoir: an exact stochastic analysis of
> iteron-based plasmid copy-number control.* Journal of Theoretical Biology.

Every result in the paper is computed; no new experimental data were generated. This
repository contains the model, the scripts that generate each result table, the tables
themselves, and the script that draws the three figures. Measured values taken from the
literature are tabulated with the source of each in `data/p1_parameters.csv`.

## Layout

| path | contents |
|---|---|
| `iteron_cme.py` | the model: stationary solution of the chemical master equation for copy number and total initiator, with the iteron reservoir at conditional equilibrium |
| `regenerate.py` | regenerates the result tables in `data/` |
| `verify.py` | recomputes the inexpensive tables and compares them against the shipped copies |
| `make_figs.py` | draws Figures 1 to 3 from the tables in `data/` |
| `data/` | every result table used by the paper or by the figures |
| `figures/` | the three figures, as vector PDF and as raster PNG |

## Running it

Tested on Python 3.11 with the versions in `requirements.txt`.

```
pip install -r requirements.txt
python verify.py                 # about a minute: recomputes and compares
python regenerate.py --list      # every target, and which are slow
python regenerate.py fast        # the deterministic tables, seconds
python regenerate.py all         # adds the master-equation sweeps, hours
python make_figs.py              # redraws the figures from data/
```

`verify.py` snapshots the shipped tables into `reference/` the first time it runs, then
recomputes and compares against that snapshot. `reference/` is therefore generated, not
distributed, and is listed in `.gitignore`. To check a slow table, regenerate it by name
and run `verify.py` again; it compares any slow table it finds in both directories.

`make_figs.py` reads only the saved tables, so redrawing takes seconds and runs no
solver. It carries fallback style definitions, so the figures reproduce without any
external style package.

## The model

The state is the pair (plasmid copy number `n`, total initiator `x_T`). Initiator binding
to the iterons is fast relative to replication and dilution, so the bound amount is at
conditional equilibrium given the state, and the stationary distribution is obtained by
solving `Q pi = 0` on a truncated two-dimensional state space with a sparse direct
solver.

The truncation is not a free parameter. Every quoted result was checked for independence
of the state-space ceilings `nmax` and `xTmax`, and the boundary mass is reported
alongside each number; `data/boundary_check.csv` and `data/cv2xf_convergence.csv` are
those checks. This matters because the copy-number distribution can carry a tail heavy
enough that a second moment computed at too low a ceiling is an artifact of where the
tail was cut.

Processes represented: bursty initiator synthesis, autorepression of the initiator gene,
binding to `n*m` iterons, replication fired by free initiator and suppressed by origin
pairing, degradation of free and of bound initiator, and dilution of both the plasmid
pool and the initiator by growth.

## Which table backs which claim

| table | used for |
|---|---|
| `decoupling_check.csv`, `fig1c_collapse.csv` | Figure 1a, 1c: the decoupling and its collapse onto the capacity ratio |
| `fig1b_signreversal.csv` | Figure 1b: a co-scaling against a fixed reservoir, opposite in sign |
| `closed_loop_gain.csv`, `gain_with_sn.csv` | composition of reservoir and autorepression, and its generalization to copy-number-dependent synthesis |
| `no_autorep_control.csv`, `handcuff_convergence.csv`, `boundary_check.csv` | Figure 2a, 2b: no stationary state without origin pairing |
| `fig2c_pmf.csv`, `loss_rate_sweep.csv` | Figure 2c, 2d: narrowing distributions and the segregational loss rate |
| `clustering_test.csv` | whether spatial restriction of pairing partners destroys the restoring force |
| `control_khc_sensitivity.csv`, `control_hc_loaded.csv` | the two controls on the loss-rate result |
| `optimum_invariant.csv`, `optimum_robustness*.csv`, `optimum_measured_kd.csv` | Figure 3: the optimum, its crossover with affinity, and what is conserved at it |
| `abeles_comparison.csv`, `abeles_retention.csv` | the incompatibility comparison |
| `p1_parameters.csv`, `p1_parameter_gains.csv` | measured P1 parameters, with the source of each value |
| `timescale_separation.csv` | binding residence time against generation time |

## What can be regenerated, and what is shipped only

`regenerate.py` covers nine of the twenty-three tables. `verify.py` checks exactly those
nine, which is why it reports nine rather than twenty-three.

| table | status |
|---|---|
| `decoupling_check`, `fig1b_signreversal`, `fig1c_collapse`, `gain_with_sn`, `timescale_separation` | regenerable, seconds (`regenerate.py fast`) |
| `clustering_test`, `handcuff_convergence`, `loss_rate_sweep`, `no_autorep_control` | regenerable, slow (`regenerate.py all`) |
| `p1_parameters`, `p1_parameter_gains` | not computed. Measured values collected from the literature, with the source of each in the `source` column |
| `abeles_comparison`, `abeles_retention` | derived from the published incompatibility measurements together with the model, by the procedure described in the paper |
| `boundary_check`, `cv2xf_convergence`, `fig2c_pmf`, `closed_loop_gain`, `control_khc_sensitivity`, `control_hc_loaded`, `optimum_invariant`, `optimum_measured_kd`, `optimum_robustness`, `optimum_robustness_ext` | computed with `iteron_cme.solve` but shipped without a driver script |

For the last group the deposited tables are the record, and that is a real limitation
of this deposit rather than a presentational one. Each table carries the parameters
that were *swept* as columns, but not the fixed ones held in common across its rows,
and those differ between tables: the optimum sweeps, for example, were run without the
origin-pairing term that `loss_rate_sweep` uses. Reproducing one of these tables from
`iteron_cme.solve` therefore requires reading the corresponding section of the paper
for the fixed parameter set. `verify.py` does not check them.

## Reproducibility, including the limits

* The sweeps are deterministic. There is no Monte Carlo and no random seed, so a rerun
  reproduces the shipped tables to solver tolerance rather than approximately.
* Columns obtained by finite differencing a numerically solved root reproduce to about
  `1e-8` rather than to machine precision, because the root finder sets the floor.
  `verify.py` applies a looser tolerance to those columns and states which they are; the
  closed-form columns beside them agree to about `1e-15`.
* Figures regenerate with identical content but are not byte-identical to the deposited
  PNGs, because `bbox_inches="tight"` crops to the rendered text extent and that shifts
  the canvas by a few pixels across font versions.
* `regenerate.py all` takes a few hours on one core. The clustering and replication
  sweeps dominate.

## Citation

Cite the paper above. This repository is archived on Zenodo; the DOI of the archived
version appears on the Zenodo record for the release. `CITATION.cff` carries the
machine-readable metadata.

## License

MIT, see `LICENSE`.
