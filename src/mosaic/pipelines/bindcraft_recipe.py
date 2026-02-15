from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from mosaic.optimizers import gradient_MCMC, simplex_APGM


StageKind = Literal["logits_apgm", "simplex_apgm", "gradient_mcmc"]


@dataclass(frozen=True, slots=True)
class BindCraftStageConfig:
    """
    Single optimization stage in a BindCraft-style recipe.
    """

    name: str
    kind: StageKind
    n_steps: int
    stepsize: float | None = None
    momentum: float = 0.0
    scale: float = 1.0
    max_gradient_norm: float | None = 1.0
    update_loss_state: bool = False
    serial_evaluation: bool = False
    sample_loss: bool = False
    proposal_temp: float = 0.01
    temp: float = 0.001
    max_path_length: int = 2
    detailed_balance: bool = False


@dataclass(frozen=True, slots=True)
class BindCraftRecipeConfig:
    """
    Ordered stage list that defines a full optimization campaign.
    """

    stages: tuple[BindCraftStageConfig, ...] = field(default_factory=tuple)
    fix_loss_key_for_mcmc: bool = True

    @staticmethod
    def kickoff_default() -> "BindCraftRecipeConfig":
        """
        Minimal default schedule to start integrating a staged optimizer.
        """
        return BindCraftRecipeConfig(
            stages=(
                BindCraftStageConfig(
                    name="logits_warmup",
                    kind="logits_apgm",
                    n_steps=50,
                    stepsize=1.5,
                    momentum=0.2,
                    scale=1.0,
                    max_gradient_norm=1.0,
                ),
                BindCraftStageConfig(
                    name="simplex_refine",
                    kind="simplex_apgm",
                    n_steps=40,
                    stepsize=0.5,
                    momentum=0.0,
                    scale=1.1,
                    max_gradient_norm=1.0,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class BindCraftStageResult:
    name: str
    kind: StageKind
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class BindCraftRunResult:
    final_pssm: Float[Array, "N 20"]
    best_pssm: Float[Array, "N 20"]
    stage_results: tuple[BindCraftStageResult, ...]


def _ensure_pssm(x: Float[Array, "N 20"]) -> Float[Array, "N 20"]:
    x = jnp.asarray(x)
    if x.ndim != 2 or x.shape[-1] != 20:
        raise ValueError(f"Expected x to have shape (N, 20); got {x.shape}.")
    return x


def _run_apgm_stage(
    *,
    loss_function,
    x,
    stage: BindCraftStageConfig,
    key,
    logspace: bool,
):
    if stage.stepsize is None:
        raise ValueError(f"Stage '{stage.name}' requires `stepsize` for APGM.")

    return simplex_APGM(
        loss_function=loss_function,
        x=x,
        n_steps=stage.n_steps,
        stepsize=stage.stepsize * np.sqrt(x.shape[0]),
        momentum=stage.momentum,
        scale=stage.scale,
        max_gradient_norm=stage.max_gradient_norm,
        update_loss_state=stage.update_loss_state,
        logspace=logspace,
        serial_evaluation=stage.serial_evaluation,
        sample_loss=stage.sample_loss,
        key=key,
    )


def _run_mcmc_stage(
    *,
    loss_function,
    x,
    stage: BindCraftStageConfig,
    key,
    fix_loss_key: bool,
):
    seq = jnp.argmax(x, axis=-1)
    seq = gradient_MCMC(
        loss=loss_function,
        sequence=seq,
        temp=stage.temp,
        proposal_temp=stage.proposal_temp,
        max_path_length=stage.max_path_length,
        steps=stage.n_steps,
        alphabet_size=20,
        key=key,
        detailed_balance=stage.detailed_balance,
        fix_loss_key=fix_loss_key,
        serial_evaluation=stage.serial_evaluation,
    )
    pssm = jax.nn.one_hot(seq, 20)
    return pssm, pssm


def run_bindcraft_recipe(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    recipe: BindCraftRecipeConfig,
    key=None,
) -> BindCraftRunResult:
    """
    Run a staged, BindCraft-style optimization recipe using Mosaic optimizers.
    """
    if key is None:
        key = jax.random.key(np.random.randint(0, 10000))

    current_x = _ensure_pssm(x)
    best_x = current_x
    stage_results: list[BindCraftStageResult] = []

    for i, stage in enumerate(recipe.stages):
        stage_key = jax.random.fold_in(key, i)
        start = time()

        if stage.kind == "logits_apgm":
            current_x, best_x = _run_apgm_stage(
                loss_function=loss_function,
                x=jnp.log(current_x + 1e-5),
                stage=stage,
                key=stage_key,
                logspace=True,
            )
        elif stage.kind == "simplex_apgm":
            current_x, best_x = _run_apgm_stage(
                loss_function=loss_function,
                x=current_x,
                stage=stage,
                key=stage_key,
                logspace=False,
            )
        elif stage.kind == "gradient_mcmc":
            current_x, best_x = _run_mcmc_stage(
                loss_function=loss_function,
                x=current_x,
                stage=stage,
                key=stage_key,
                fix_loss_key=recipe.fix_loss_key_for_mcmc,
            )
        else:
            raise ValueError(f"Unknown stage kind: {stage.kind}")

        stage_results.append(
            BindCraftStageResult(
                name=stage.name,
                kind=stage.kind,
                elapsed_s=time() - start,
            )
        )

    return BindCraftRunResult(
        final_pssm=current_x,
        best_pssm=best_x,
        stage_results=tuple(stage_results),
    )
