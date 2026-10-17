"""Regenerate the cheap result tables and compare them against the shipped copies.

    python verify.py

Only the deterministic/root-finding targets are rerun, so this takes about a minute.
The master-equation sweeps are checked by `regenerate.py <name>` followed by a rerun of
this script, which then finds the regenerated file in data/ and compares it with the
reference copy in reference/.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regenerate  # noqa: E402

DATA = os.path.join(HERE, "data")
REF = os.path.join(HERE, "reference")
TOL = 1e-9          # relative, default for every numeric column
# Columns holding a central-difference logarithmic derivative of a numerically solved
# root reproduce only to the root-finder tolerance, not to machine precision. The
# closed-form columns beside them (g_co_pred, g_pred, g_analytic) are held to TOL and
# agree to ~1e-15, which is what establishes that the method reproduces.
FD_COLUMNS = {"g_co", "g_fix", "g_numeric"}
FD_TOL = 1e-6


def compare(name, fresh):
    ref_path = os.path.join(REF, name + ".csv")
    if not os.path.exists(ref_path):
        return name, None, "no reference copy"
    ref = pd.read_csv(ref_path)
    if list(fresh.columns) != list(ref.columns):
        return name, False, f"columns differ: {list(fresh.columns)} vs {list(ref.columns)}"
    if len(fresh) != len(ref):
        return name, False, f"row count {len(fresh)} vs {len(ref)}"
    worst, where = 0.0, ""
    for c in ref.columns:
        # pandas >= 3 uses an extension StringDtype that np.issubdtype cannot read
        if not pd.api.types.is_numeric_dtype(ref[c]):
            if not (fresh[c].astype(str) == ref[c].astype(str)).all():
                return name, False, f"non-numeric column {c} differs"
            continue
        a = fresh[c].to_numpy(float)
        b = ref[c].to_numpy(float)
        denom = np.where(np.abs(b) > 0, np.abs(b), 1.0)
        rel = np.abs(a - b) / denom
        tol_c = FD_TOL if c in FD_COLUMNS else TOL
        m = float(np.nanmax(rel))
        if m > tol_c:
            return name, False, f"column {c!r} off by {m:.2e} (tolerance {tol_c:g})"
        if m > worst:
            worst, where = m, c
    return name, True, f"worst relative deviation {worst:.2e} in {where!r}"


def main():
    if not os.path.isdir(REF):
        # first run: the shipped data/ IS the reference
        os.makedirs(REF, exist_ok=True)
        for f in os.listdir(DATA):
            if f.endswith(".csv"):
                shutil.copy(os.path.join(DATA, f), os.path.join(REF, f))
        print(f"created reference/ from the shipped data/ ({len(os.listdir(REF))} files)")

    results = []
    for name, fn in regenerate.FAST.items():
        results.append(compare(name, fn()))
    # any slow table already regenerated into data/ is compared as it stands
    for name in regenerate.SLOW:
        p = os.path.join(DATA, name + ".csv")
        if os.path.exists(p) and os.path.exists(os.path.join(REF, name + ".csv")):
            results.append(compare(name, pd.read_csv(p)))

    bad = 0
    for name, ok, msg in results:
        tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        bad += (ok is False)
        print(f"  {tag}  {name:24s} {msg}")
    print(f"\n{len(results)} tables checked, {bad} failed (tolerance {TOL:g} relative)")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
