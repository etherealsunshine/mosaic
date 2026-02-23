# LM Losses Under Simplex vs Logit Optimization (Mosaic Experiment)

## Summary
Inspired by https://github.com/escalante-bio/mosaic/issues/35#issuecomment-3910165459

 When a language-model-style sequence prior (ProteinMPNN inverse-folding loss, optionally ESM PLL) is included in the objective, optimizing in unconstrained logit space can produce worse optimization dynamics than simplex-native optimization.

In this run, I compared three optimization geometries on the same objective and initialization:
- `simplex` (projected simplex updates; Mosaic-style)
- `logit` (unconstrained logits, mapped via softmax for loss evaluation)
- `mirror` (entropic mirror descent / exponentiated-gradient on simplex)

## Hypothesis
Unconstrained logit optimization may:
- weaken the practical influence of sequence-prior terms
- underperform on the combined objective relative to simplex-native updates

## Experimental Setup
- Target: PDL1 target chain sequence (from a fixed known binder-target complex 4ZQK PDB ID) 
- Initialization:
- `init_pssm`: empirical binder PSSM sampled from fixed-structure ProteinMPNN
- `pos_mask`: interface mask (only interface positions optimized)
- Structure model: `Boltz2`
- Sequence prior term: `InverseFoldingSequenceRecovery` (ProteinMPNN-based)
- Objective:
- `L_total = L_struct + L_LM` (with fixed weights across all arms)
- Same seed(s), same mask, same initialization, same loss weights across arms
- Only optimizer geometry changes

## Metrics Tracked
Per step:
- `L_struct`
- `L_LM`
- `L_total`
- `H` (mean entropy over binder positions)
- `p_max_mean` (mean max amino-acid probability per position; higher = more peaked)

## Main Result
The simplex arm consistently outperformed logit and mirror in this setup.

Observed behavior:
- `simplex` reduced `L_total` and `L_struct` substantially more than `logit`/`mirror`
- `simplex` became less peaked over time (`p_max_mean` decreased)
- `logit` and `mirror` remained more peaked (`p_max_mean` stayed high / increased slightly)
- `L_LM` did not show dramatic improvement in any arm, but `simplex` was not worse and often slightly better

Representative late-step comparison (same run horizon):
- `simplex`: lower `L_total`, lower `L_struct`, lower `p_max_mean`
- `logit`: higher `L_total`, higher `L_struct`, higher `p_max_mean`
- `mirror`: similar to `logit` in this experiment

## Interpretation
We can say that unconstrained/logit-space optimization can have worse optimization dynamics for this mixed structure + sequence-prior objective.

Somethings I'd consider:
- This experiment starts from an existing binder-derived empirical PSSM (refinement setting), not de novo random initialization. I think this might reduce headroom for `L_LM` improvement. It's also worth noting that the  conclusion here is about optimization dynamics and geometry, not a universal claim about LM collapse in all settings.

## Caveats
- ProteinMPNN-based LM term is stochastic , so variance across seeds is expected.

## Reproducibility Notes
Core script:
- `run_lm_simplex_vs_logit.py`

Inputs used:
- `binder_empirical_pssm.npy`
- `binder_interface_mask.npy`
will put out code on reproducing this soon

Recommended execution:
```bash
python3 -u run_lm_simplex_vs_logit.py > final_log.log 2>&1
