"""Per-group sequence predictability, to give the 'predictability privilege'
claim direct evidence.

The paper argues that disadvantaged-IMD and disabled students receive higher
recall because their click patterns are 'more standardised / template-like /
predictable'. The reviewer correctly noted this causal mechanism is asserted with
zero measurement. This script measures it.

For each protected group we compute two predictability signals over the training
histories:

  1. Next-item conditional entropy H(next | current): build the transition
     distribution P(next_item | current_item) restricted to each group's sessions,
     then average the Shannon entropy weighted by how often each current_item
     occurs. LOWER entropy = more predictable = the 'template-like' claim.

  2. Top-1 transition mass: for each current_item, the probability of its single
     most likely successor, averaged. HIGHER = more predictable.

If disadvantaged/disabled groups show lower entropy and higher top-1 mass, that is
direct evidence for predictability privilege. If not, the paper should soften the
claim -- either way this script tells you the truth.

Output: results/group_predictability.csv

Usage:
    python -m experiments.run_predictability
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import log2

import numpy as np
import pandas as pd

from src.eval.fairness import load_sequences
from src.utils import paths

ATTRS = {
    "gender": ["F", "M"],
    "age_binary": ["35+", "0-35"],
    "disability": ["Y", "N"],
    "imd_binary": ["disadvantaged", "advantaged"],
}


def _predictability_for_sessions(seqs):
    """Given a list of item-sequences, return (mean conditional entropy,
    mean top-1 transition mass)."""
    # transitions[current][next] = count
    transitions = defaultdict(Counter)
    for seq in seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            transitions[a][b] += 1

    if not transitions:
        return float("nan"), float("nan")

    entropies = []
    top1_masses = []
    weights = []
    for current, succ in transitions.items():
        total = sum(succ.values())
        if total == 0:
            continue
        probs = np.array([c / total for c in succ.values()])
        # Shannon entropy in bits.
        ent = -np.sum(probs * np.log2(probs))
        entropies.append(ent)
        top1_masses.append(probs.max())
        weights.append(total)  # weight by how often this state is visited

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    mean_entropy = float(np.sum(np.array(entropies) * weights))
    mean_top1 = float(np.sum(np.array(top1_masses) * weights))
    return mean_entropy, mean_top1


def main():
    paths.ensure_dirs()
    sequences = load_sequences()

    # sequences has a 'seq' column (list of item indices) per session, plus groups.
    if "seq" not in sequences.columns:
        raise SystemExit("expected a 'seq' column in sequences.parquet")

    rows = []
    for attr, values in ATTRS.items():
        if attr not in sequences.columns:
            continue
        for val in values:
            sub = sequences[sequences[attr] == val]
            if len(sub) == 0:
                continue
            seqs = [list(s) for s in sub["seq"].tolist()]
            ent, top1 = _predictability_for_sessions(seqs)
            rows.append({
                "attribute": attr,
                "group": val,
                "n_sessions": len(sub),
                "mean_cond_entropy_bits": ent,
                "mean_top1_transition_mass": top1,
            })
            print(f"  {attr}={val:>14}  n={len(sub):6d}  "
                  f"H(next|cur)={ent:.3f} bits  top1_mass={top1:.3f}")

    df = pd.DataFrame(rows)
    out = paths.RESULTS_DIR / "group_predictability.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(df.to_string(index=False))

    # Interpretation helper: for each attribute, is the first-listed (focus) group
    # more predictable (lower entropy, higher top1) than the second?
    print("\n--- Predictability-privilege check (focus vs other) ---")
    for attr, values in ATTRS.items():
        d = df[df["attribute"] == attr]
        if len(d) < 2:
            continue
        f = d[d["group"] == values[0]]
        o = d[d["group"] == values[1]]
        if len(f) and len(o):
            f, o = f.iloc[0], o.iloc[0]
            more_pred = (f["mean_cond_entropy_bits"] < o["mean_cond_entropy_bits"]
                         and f["mean_top1_transition_mass"] > o["mean_top1_transition_mass"])
            verdict = "SUPPORTS privilege" if more_pred else "does NOT cleanly support"
            print(f"  {attr}: {values[0]} vs {values[1]} -> "
                  f"entropy {f['mean_cond_entropy_bits']:.3f} vs {o['mean_cond_entropy_bits']:.3f}, "
                  f"top1 {f['mean_top1_transition_mass']:.3f} vs {o['mean_top1_transition_mass']:.3f} "
                  f"=> {verdict}")


if __name__ == "__main__":
    main()