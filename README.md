# Co-replicating iteron arrays filter plasmid copy-number noise without controlling copy number

Code and result tables for

> S. Tal, *Co-replicating iteron arrays filter plasmid copy-number noise without controlling copy number* (2026).

Every result in the paper is computed from the model; no experimental data were generated.
The repository holds the model, one script that computes every result table, one script
that draws every data figure, the tables, and the figures.

## Files

| path | contents |
|---|---|
| `iteron_cme.py` | the model: stationary solution of the chemical master equation for copy number and total initiator, with iteron binding at conditional equilibrium (Results, Model formulation; Methods) |
| `binding_cme3d.py` | the master equation with explicit binding kinetics on (n, x_T, bound), used to validate the fast-binding reduction (Results; Supplementary Note S2) |
| `lna.py` | the linear-noise approximation of the same model (Results; Methods, Linear-noise approximation) |
| `cell_cycle.py` | the same model with discrete cell division and binomial partitioning instead of continuous dilution (Supplementary Note S4) |
| `compute_tables.py` | computes every table in `data/` |
| `make_figures.py` | draws Figs. 2–5 and Supplementary Figs. S1–S4 into `figures/` from the tables |
| `data/` | the result tables |
| `figures/` | all figures as vector PDF and 600-dpi PNG; Fig. 1 is a schematic with no data behind it |

## Running it

Tested with Python 3.11 and the package versions in `requirements.txt`.

```
pip install -r requirements.txt
python compute_tables.py --list        # every target, its cost, and what it is for
python compute_tables.py fast          # the deterministic tables, seconds
python compute_tables.py all           # everything, several hours on 12 cores
python compute_tables.py optimum_grid  # a single table
python make_figures.py                 # redraws every figure from data/, seconds
```

The matched-mean sweeps run in parallel (`--jobs N`, default: number of cores minus two,
at most 12). Targets that read another table (for example `copy_number_pmf` reads the synthesis
rates in `loss_rate_sweep`) are listed after it, so `all` runs them in a valid order.
`binding_validation` (1–2 h) requires `pypardiso` (MKL PARDISO); SciPy's default sparse
solver cannot factor the three-dimensional lattice at these sizes. `binding_assemble`
recomputes only the comparison step of that table from its stored explicit-binding columns,
in seconds.

## Which table backs which result

Figure, equation, table, and note numbers refer to the manuscript and Supplementary Information.

| table | used for |
|---|---|
| `decoupling_check` | Fig. 2a |
| `sign_reversal` | Fig. 2b, Eq. (10) |
| `gain_collapse` | Fig. 2c, Eq. (9): 76 combinations of array size and affinity |
| `closed_loop_gain` | Results, Eq. (11): 18 parameter sets |
| `gain_with_sn` | Results, Eq. (12): 24 parameter sets |
| `boundary_check` | Fig. 3a, Results: autorepression without handcuffing |
| `handcuff_convergence` | Fig. 3a, Results, Methods: with handcuffing |
| `no_autorep_control` | Supplementary Fig. S1, Results, Supplementary Note S1 |
| `clustering_test` | Results (Model formulation): pairing partners restricted to clusters, or capped |
| `loss_rate_sweep` | Fig. 3c, Fig. 5a, Results: the matched-mean sweep (mean copy number five) |
| `copy_number_pmf` | Fig. 3b (and single-copy n = 1 loss decomposition) |
| `control_pairing_strength`, `control_hc_loaded` | Results: the two controls on the mechanism |
| `khc_sensitivity`, `khc_sensitivity_summary` | Supplementary Fig. S3, Results, Supplementary Note S3 |
| `lna_comparison` | Fig. 4, Results, Methods (Linear-noise approximation) |
| `optimum_grid`, `optimum_summary` | Fig. 5b,c, Results |
| `optimum_p1` | Fig. 5d, Results |
| `optimum_ceiling_check` | Results: the optima re-solved on a larger state space |
| `binding_validation`, `binding_reduced_ref` | Supplementary Fig. S2, Results, Methods (Time-scale separation), Supplementary Note S2 |
| `timescale_separation` | Methods (Time-scale separation) |
| `abeles_retention`, `abeles_comparison` | Methods (Comparison with the incompatibility data) |
| `cell_cycle_check` | Supplementary Fig. S4, Results, Supplementary Note S4: discrete division |

## The model

The state is the pair (plasmid copy number `n`, total initiator `x_T`). Initiator binding
to the `n m` iterons is fast relative to replication and dilution, so the bound amount is
at conditional equilibrium given the state, and every propensity that depends on free
initiator is averaged over that equilibrium. The stationary distribution solves `Q pi = 0`
on a truncated two-dimensional lattice, by a sparse direct solver; there is no sampling.

Processes: bursty initiator synthesis with gene dosage `n`, autorepression, removal of free
and bound initiator, replication fired by free initiator and divided by the handcuffing
factor `1 + K_hc E[p^2 | x_T] (n - 1)`, and dilution of plasmids. The handcuffed initiation
rate is written as a ratio of conditional means; this is the only closure in the reduced
model, and the explicit-binding comparison bounds its effect at 2.4e-4 relative.

Without handcuffing (for copy-number-proportional synthesis) the copy-number distribution
has no stationary form, so any moment computed on a truncated lattice reflects where the
lattice was cut. Every table reports the probability in the boundary states (`edge_n`,
`edge_T`), and `handcuff_convergence`, `boundary_check`, and `optimum_ceiling_check` test
the ceilings directly. All results with handcuffing use the baseline parameters of Table 1
(`BASE` in `compute_tables.py`); the tables that deliberately omit handcuffing or
autorepression (`boundary_check`, `no_autorep_control`) say so in their docstrings.

## Reproducibility

* The calculations are deterministic. A rerun reproduces the tables to the tolerance of the
  matched-mean search (mean copy number matched to 1e-7) and of the sparse solver.
* Columns obtained by finite differencing a numerical root (`g_num`, `g_co`, `g_numeric`)
  reproduce to about 1e-8.
* `binding_validation` was computed with the synthesis rates of an earlier run of
  `loss_rate_sweep`, which differ from the present ones by at most 1.4e-6 relative; the
  reduced model on those grids reproduces the present `loss_rate_sweep` to 1.5e-6.
* Redrawn figures have identical content but are not byte-identical to the deposited files.

## Citation

Cite the paper above; `CITATION.cff` carries the metadata.

## License

MIT, see `LICENSE`.
