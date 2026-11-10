# Co-replicating iteron arrays filter plasmid copy-number noise without controlling copy number

Code and result tables for

> S. Tal, *Co-replicating iteron arrays filter plasmid copy-number noise without controlling copy number*, submitted to Physical Review E (2026).

Every result in the paper is computed from the model; no experimental data were generated.
The repository holds the model, one script that computes every result table, one script
that draws every data figure, the tables, and the figures.

## Files

| path | contents |
|---|---|
| `iteron_cme.py` | the model: stationary solution of the chemical master equation for copy number and total initiator, with iteron binding at conditional equilibrium (Sec. II, Appendix A) |
| `binding_cme3d.py` | the master equation with explicit binding kinetics on (n, x_T, bound), used to validate the fast-binding reduction (Sec. III F, Supplemental Sec. S2) |
| `lna.py` | the linear-noise approximation of the same model (Sec. III D, Appendix C) |
| `cell_cycle.py` | the same model with discrete cell division and binomial partitioning instead of continuous dilution (Supplemental Sec. S4) |
| `compute_tables.py` | computes every table in `data/` |
| `make_figures.py` | draws Figs. 2-5 and S1-S4 into `figures/` from the tables |
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
`binding_validation` (1-2 h) requires `pypardiso` (MKL PARDISO); SciPy's default sparse
solver cannot factor the three-dimensional lattice at these sizes. `binding_assemble`
recomputes only the comparison step of that table from its stored explicit-binding columns,
in seconds.

## Which table backs which result

Figure and section numbers refer to the manuscript.

| table | used for |
|---|---|
| `decoupling_check` | Fig. 2(a) |
| `sign_reversal` | Fig. 2(b), Eq. (10) |
| `gain_collapse` | Fig. 2(c), Eq. (9): 76 combinations of array size and affinity |
| `closed_loop_gain` | Sec. III A, Eq. (11): 18 parameter sets |
| `gain_with_sn` | Sec. III A, Eq. (12): 24 parameter sets |
| `boundary_check` | Fig. 3(a), Sec. III B: autorepression without handcuffing |
| `handcuff_convergence` | Fig. 3(a), Sec. III B, Appendix A: with handcuffing |
| `no_autorep_control` | Fig. S1, Sec. III B, Supplemental Sec. S1 |
| `clustering_test` | Sec. II C: pairing partners restricted to clusters, or capped |
| `loss_rate_sweep` | Fig. 3(c), Fig. 5(a), Sec. III C: the matched-mean sweep (mean copy number five) |
| `copy_number_pmf` | Fig. 3(b) |
| `control_pairing_strength`, `control_hc_loaded` | Sec. III C: the two controls on the mechanism |
| `khc_sensitivity`, `khc_sensitivity_summary` | Fig. S3, Sec. III C and E, Supplemental Sec. S3 |
| `lna_comparison` | Fig. 4, Sec. III B and D, Appendix C |
| `optimum_grid`, `optimum_summary` | Fig. 5(b,c), Sec. III E |
| `optimum_p1` | Fig. 5(d), Sec. III E |
| `optimum_ceiling_check` | Sec. III E: the optima re-solved on a larger state space |
| `binding_validation`, `binding_reduced_ref` | Fig. S2, Sec. III F, Appendix B, Supplemental Sec. S2 |
| `timescale_separation` | Appendix B |
| `abeles_retention`, `abeles_comparison` | Appendix D |
| `cell_cycle_check` | Fig. S4, Sec. III C and E, Supplemental Sec. S4: discrete division |

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

Without handcuffing the copy-number distribution has no stationary form, so any moment
computed on a truncated lattice reflects where the lattice was cut. Every table reports the
probability in the boundary states (`edge_n`, `edge_T`), and `handcuff_convergence`,
`boundary_check` and `optimum_ceiling_check` test the ceilings directly. All results with
handcuffing use the baseline parameters of Table I (`BASE` in `compute_tables.py`); the
tables that deliberately omit handcuffing or autorepression (`boundary_check`,
`no_autorep_control`) say so in their docstrings.

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
