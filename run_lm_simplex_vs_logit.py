# run_lm_simplex_vs_logit.py
import gc
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

import mosaic.losses.structure_prediction as sp
from mosaic.models.boltz2 import Boltz2
from mosaic.structure_prediction import TargetChain
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.optimizers import projection_simplex

# Optional: reduce JAX OOM issues
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# -----------------------------
# Config (edit these)
# -----------------------------
TARGET_SEQUENCE = "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
INIT_PSSM_PATH = "binder_empirical_pssm.npy"
MASK_PATH = "binder_interface_mask.npy"
OUT_DIR = "lm_vs_simplex_runs/run_oom_safe"

SEED = 230
USE_ESMC = False


@dataclass
class RunConfig:
    steps: int = 3
    lr: float = 0.13
    w_struct: float = 1.0
    w_mpnn: float = 1.0
    w_esmc: float = 0.0
    w_entropy: float = 0.0
    grad_clip: float = 1.0
    eps: float = 1e-8
    # OOM-safe toggles
    compute_lm_grad_metrics: bool = False
    lm_grad_every: int = 10


def build_components(target_sequence: str, binder_len: int, use_esmc: bool = False):
    model = Boltz2()
    features, _ = model.binder_features(
        binder_length=binder_len,
        chains=[TargetChain(target_sequence, use_msa=False)],
    )

    struct_terms = (
        1.5 * sp.BinderTargetContact()
        + 0.75 * sp.WithinBinderContact()
        + 0.05 * sp.BinderTargetPAE()
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.IPTMLoss()
        + 0.10 * sp.PLDDTLoss()
    )

    # Reduce sampling_steps for memory
    L_struct = model.build_loss(
        loss=struct_terms,
        features=features,
        recycling_steps=1,
        sampling_steps=2,
    )

    mpnn = load_mpnn_sol()
    L_mpnn = model.build_loss(
        loss=InverseFoldingSequenceRecovery(
            mpnn=mpnn,
            temp=jnp.array(0.10),
            num_samples=2,        # reduced
            jacobi_iterations=3,  # reduced
        ),
        features=features,
        recycling_steps=1,
        sampling_steps=2,
    )

    L_esmc = None
    if use_esmc:
        from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc
        esmc = load_esmc("esmc_300m")
        L_esmc = ESMCPseudoLikelihood(esm=esmc, stop_grad=True)

    return L_struct, L_mpnn, L_esmc


def _entropy(p, eps=1e-8):
    return -(p * jnp.log(p + eps)).sum(-1).mean()


def _norm(x, eps=1e-12):
    return jnp.sqrt((x * x).sum() + eps)

def make_plots(out_dir: str):
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    out = Path(out_dir)
    df = pd.read_csv(out / "step_logs_all.csv")

    # Mean over seeds
    g = (
        df.groupby(["arm", "step"], as_index=False)
        .agg(
            L_struct=("L_struct", "mean"),
            L_LM=("L_LM", "mean"),
            L_total=("L_total", "mean"),
            H=("H", "mean"),
            p_max_mean=("p_max_mean", "mean"),
            step_improve=("step_improve", "mean"),
        )
    )

    metrics = ["L_struct", "L_LM", "L_total", "H", "p_max_mean", "step_improve"]
    for m in metrics:
        plt.figure(figsize=(6, 4))
        for arm in ["simplex", "logit"]:
            d = g[g["arm"] == arm]
            plt.plot(d["step"], d[m], label=arm)
        plt.title(f"{m} (mean over seeds)")
        plt.xlabel("step")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"plot_{m}.png", dpi=160)
        plt.close()

    # Also dump final-step summary table
    final = g.sort_values("step").groupby("arm").tail(1)
    final.to_csv(out / "final_step_summary.csv", index=False)
    print("Saved plots + summary to", out)

def run_arm(
    arm: str,
    init_pssm: jnp.ndarray,
    pos_mask: jnp.ndarray,
    L_struct,
    L_mpnn,
    L_esmc,
    cfg: RunConfig,
    key=jax.random.key(230),
    verbose=True,
):
    assert arm in ("simplex", "logit")
    mask = pos_mask.astype(jnp.float32)[:, None]
    ref_pssm = init_pssm.astype(jnp.float32)

    var = jnp.log(ref_pssm + 1e-5) if arm == "logit" else ref_pssm
    logs = []
    prev_total = None

    def to_pssm(v):
        return jax.nn.softmax(v, axis=-1) if arm == "logit" else v

    def masked_pssm(v):
        p = to_pssm(v)
        p_eff = mask * p + (1.0 - mask) * ref_pssm
        return p_eff / (p_eff.sum(-1, keepdims=True) + cfg.eps)

    for t in range(cfg.steps):
        t0 = time.time()
        k = jax.random.fold_in(key, t)

        def f_struct(v):
            return L_struct(masked_pssm(v), key=jax.random.fold_in(k, 1))[0]

        def f_mpnn(v):
            return L_mpnn(masked_pssm(v), key=jax.random.fold_in(k, 2))[0]

        def f_esmc(v):
            if L_esmc is None:
                return 0.0
            return L_esmc(masked_pssm(v), key=jax.random.fold_in(k, 3))[0]

        def f_ent(v):
            return _entropy(masked_pssm(v), cfg.eps)

        def f_total(v):
            return (
                cfg.w_struct * f_struct(v)
                + cfg.w_mpnn * f_mpnn(v)
                + cfg.w_esmc * f_esmc(v)
                + cfg.w_entropy * f_ent(v)
            )

        t1 = time.time()
        Ls = float(f_struct(var))
        t2 = time.time()
        Lm = float(f_mpnn(var) + f_esmc(var))
        t3 = time.time()
        H = float(f_ent(var))
        Lt = float(f_total(var))
        t4 = time.time()

        g_total = jax.grad(f_total)(var)
        t5 = time.time()

        # Disabled by default for OOM safety
        if cfg.compute_lm_grad_metrics and (t % cfg.lm_grad_every == 0):
            g_lm = jax.grad(
                lambda v: cfg.w_mpnn * f_mpnn(v) + cfg.w_esmc * f_esmc(v)
            )(var)
        else:
            g_lm = jnp.zeros_like(var)
        t6 = time.time()

        gn_total = float(_norm(g_total))
        gn_lm = float(_norm(g_lm))
        grad_ratio = gn_lm / (gn_total + 1e-12)
        cos_lm_total = float(
            jnp.vdot(g_lm.ravel(), g_total.ravel())
            / ((_norm(g_lm) * _norm(g_total)) + 1e-12)
        )
        p_max_mean = float(masked_pssm(var).max(-1).mean())
        step_improve = 0.0 if prev_total is None else float(prev_total - Lt)
        prev_total = Lt

        logs.append(
            dict(
                step=t, arm=arm, L_struct=Ls, L_LM=Lm, H=H, L_total=Lt,
                grad_norm_total=gn_total, grad_norm_LM=gn_lm,
                grad_ratio_LM_to_total=grad_ratio, grad_cos_LM_total=cos_lm_total,
                p_max_mean=p_max_mean, step_improve=step_improve,
                dt_struct=t2 - t1, dt_lm=t3 - t2, dt_vals=t4 - t3,
                dt_grad_total=t5 - t4, dt_grad_lm=t6 - t5, dt_step=time.time() - t0,
            )
        )

        if verbose:
            print(
                f"[{arm}] step {t:03d} Ltot={Lt:8.4f} Ls={Ls:8.4f} Llm={Lm:8.4f} "
                f"pmax={p_max_mean:0.4f} dt(step)={time.time()-t0:0.2f}s "
                f"(struct {t2-t1:0.2f}s lm {t3-t2:0.2f}s grad {t5-t4:0.2f}s)"
            )

        if cfg.grad_clip is not None and gn_total > cfg.grad_clip:
            g_total = g_total * (cfg.grad_clip / (gn_total + 1e-12))

        var = var - cfg.lr * g_total
        if arm == "simplex":
            var = jnp.array(
                projection_simplex(np.array(var, dtype=np.float64), z=1.0),
                dtype=jnp.float32,
            )

        gc.collect()

    return masked_pssm(var), logs


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    init_pssm = jnp.array(np.load(INIT_PSSM_PATH), dtype=jnp.float32)
    pos_mask = jnp.array(np.load(MASK_PATH), dtype=jnp.float32)
    binder_len = int(init_pssm.shape[0])

    cfg = RunConfig(
        steps=15,
        lr=0.13,
        w_struct=1.0,
        w_mpnn=1.0,
        w_esmc=0.0,
        w_entropy=0.0,
        grad_clip=1.0,
        compute_lm_grad_metrics=False,
    )

    print("Building losses...")
    L_struct, L_mpnn, L_esmc = build_components(
        target_sequence=TARGET_SEQUENCE,
        binder_len=binder_len,
        use_esmc=USE_ESMC,
    )

    seeds = [230, 231, 232]
    all_logs = []

    for seed in seeds:
        key = jax.random.key(seed)
        run_dir = out / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Seed {seed}: simplex ===")
        p_simplex, log_simplex = run_arm(
            "simplex", init_pssm, pos_mask, L_struct, L_mpnn, L_esmc, cfg, key=key, verbose=True
        )
        np.save(run_dir / "final_pssm_simplex.npy", np.array(p_simplex))

        print(f"\n=== Seed {seed}: logit ===")
        p_logit, log_logit = run_arm(
            "logit", init_pssm, pos_mask, L_struct, L_mpnn, L_esmc, cfg, key=key, verbose=True
        )
        np.save(run_dir / "final_pssm_logit.npy", np.array(p_logit))

        # tag logs with seed
        for row in log_simplex:
            row["seed"] = seed
        for row in log_logit:
            row["seed"] = seed

        seed_df = pd.DataFrame(log_simplex + log_logit)
        seed_df.to_csv(run_dir / "step_logs.csv", index=False)
        all_logs.extend(log_simplex + log_logit)

    pd.DataFrame(all_logs).to_csv(out / "step_logs_all.csv", index=False)

    with open(out / "config.json", "w") as f:
        json.dump(
            {
                "seeds": seeds,
                "binder_len": binder_len,
                "use_esmc": USE_ESMC,
                "cfg": asdict(cfg),
            },
            f,
            indent=2,
        )

    print("Done:", out)
    make_plots(OUT_DIR)




if __name__ == "__main__":
    main()
