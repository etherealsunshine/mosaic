OUT_DIR = "lm_vs_simplex_runs/run_oom_safe_final"

def make_plots(out_dir: str, csv_path: str = "final_ever_trust.csv"):
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Build aggregation dict only for columns that actually exist
    agg_spec = {
        "L_struct": ("L_struct", "mean"),
        "L_LM": ("L_LM", "mean"),
        "L_total": ("L_total", "mean"),
        "p_max_mean": ("p_max_mean", "mean"),
    }
    if "H" in df.columns:
        agg_spec["H"] = ("H", "mean")
    if "step_improve" in df.columns:
        agg_spec["step_improve"] = ("step_improve", "mean")

    g = df.groupby(["arm", "step"], as_index=False).agg(**agg_spec)

    metrics = [m for m in ["L_struct", "L_LM", "L_total", "H", "p_max_mean", "step_improve"] if m in g.columns]

    for m in metrics:
        plt.figure(figsize=(6, 4))
        for arm in ["simplex", "logit", "mirror"]:
            d = g[g["arm"] == arm]
            if len(d) == 0:
                continue
            plt.plot(d["step"], d[m], label=arm)
        plt.title(f"{m} (mean over seeds)")
        plt.xlabel("step")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"plot_{m}.png", dpi=160)  # dpi, not dpsi
        plt.close()

    final = g.sort_values("step").groupby("arm").tail(1)
    final.to_csv(out / "final_step_summary.csv", index=False)
    g.to_csv(out / "step_logs_grouped_mean.csv", index=False)

    print("Saved plots + summary to", out)

make_plots(OUT_DIR)
