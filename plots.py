OUT_DIR = "lm_vs_simplex_runs/run_oom_safe"


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

make_plots(OUT_DIR)