"""Training loop with early stopping, full seeding, and deterministic execution.

The loss is plain MSE on scaled GHI, as the spec prescribes, and early stopping watches
that same validation loss.

One honest caveat is logged alongside it. Roughly half of every 24 hour target window is
night, when GHI is near zero and trivially predictable. Plain MSE therefore spends about
half its signal on hours no model can get wrong, which dilutes the quantity early stopping
is watching. Rather than silently changing the stopping criterion away from what the spec
asks for, this module additionally records a daylight-masked validation MAE in W/m^2 every
epoch. The stopping decision stays as specified, and the report can show what the diluted
loss was hiding. That is the honest way to surface a design concern without quietly
overriding the design.

Determinism is not aspirational here. The spec requires metrics stable to three decimal
places across two runs, and that requires all of: seeded Python, NumPy and torch RNGs, a
seeded DataLoader generator, zero dataloader workers, and CPU execution. cuDNN's LSTM
kernels are not deterministic, so CPU is a correctness choice, not only a convenience.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import model as model_module
from src.config import Config, configure_logging
from src.preprocess import ScalerBundle, load_features, load_scalers
from src.windowing import SplitWindows, build_upperbound_decoders
from src.windowing import load as load_windows

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "lstm"
UPPER_BOUND_NAME = "lstm_upperbound"


@dataclass(frozen=True)
class TrainingArtifacts:
    """Result of one training run.

    Attributes:
        model: the network with its best weights restored.
        history: per-epoch losses, columns epoch, train_loss, val_loss, val_masked_mae_wm2.
        best_epoch: one-based epoch index whose validation loss was lowest.
        best_val_loss: that lowest scaled-space validation MSE.
        n_parameters: trainable parameter count.
    """

    model: model_module.LSTMForecaster
    history: pd.DataFrame
    best_epoch: int
    best_val_loss: float
    n_parameters: int


def seed_everything(seed: int) -> torch.Generator:
    """Seed every random source the pipeline touches and enable deterministic kernels.

    Args:
        seed: the single seed from config.

    Returns:
        A seeded torch Generator to hand to the training DataLoader, so that shuffle
        order is reproducible rather than drawn from global state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    LOGGER.info("seeded all RNGs with %d and enabled deterministic algorithms", seed)
    return generator


def _tensors(
    encoder: np.ndarray, decoder: np.ndarray, target: np.ndarray
) -> TensorDataset:
    """Wrap the three window arrays as a float32 TensorDataset."""
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(encoder, dtype="float32")),
        torch.from_numpy(np.ascontiguousarray(decoder, dtype="float32")),
        torch.from_numpy(np.ascontiguousarray(target, dtype="float32")),
    )


def to_physical(
    scaled: np.ndarray, bundle: ScalerBundle, clip_at_zero: bool
) -> np.ndarray:
    """Inverse transform scaled predictions back to W/m^2.

    Args:
        scaled: model output of shape (samples, horizon) in scaled units.
        bundle: the fitted scalers, whose target_scaler owns the inverse transform.
        clip_at_zero: whether to apply the zero physical floor.

    Returns:
        Predictions in W/m^2, same shape.

    WHY clipping is applied, and why it is fair. Irradiance cannot be negative, and the
    same floor was already applied to the GHI inputs in preprocess, so clipping holds the
    model to the physics its own inputs obey rather than granting it a post-hoc favour.
    Leaving it off would also make the comparison asymmetric in the other direction:
    persistence is composed of real observations and can never go negative, so an unclipped
    model would be charged for an error the baseline is structurally incapable of making.
    """
    shape = scaled.shape
    physical = bundle.target_scaler.inverse_transform(
        scaled.reshape(-1, 1).astype("float64")
    ).reshape(shape)
    if clip_at_zero:
        physical = np.clip(physical, 0.0, None)
    return physical


def masked_mae(
    predictions_physical: np.ndarray,
    target_actual: np.ndarray,
    threshold: float,
) -> float:
    """Mean absolute error over daylight hours only.

    Args:
        predictions_physical: predictions in W/m^2, shape (samples, horizon).
        target_actual: observed GHI in W/m^2, same shape.
        threshold: daylight mask threshold in W/m^2. Hours at or below it are excluded.

    Returns:
        MAE in W/m^2 over the masked hours, or NaN when the mask selects nothing.
    """
    mask = target_actual > threshold
    if not mask.any():
        return float("nan")
    return float(np.abs(predictions_physical[mask] - target_actual[mask]).mean())


@torch.no_grad()
def predict_scaled(
    model: model_module.LSTMForecaster,
    encoder: np.ndarray,
    decoder: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Run the model over a split in evaluation mode.

    Args:
        model: a trained network.
        encoder: encoder inputs, shape (samples, encoder_hours, features).
        decoder: decoder inputs, shape (samples, horizon, calendar_features).
        batch_size: minibatch size for inference.

    Returns:
        Scaled predictions of shape (samples, horizon).
    """
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(encoder), batch_size):
        stop = start + batch_size
        batch_encoder = torch.from_numpy(
            np.ascontiguousarray(encoder[start:stop], dtype="float32")
        )
        batch_decoder = torch.from_numpy(
            np.ascontiguousarray(decoder[start:stop], dtype="float32")
        )
        outputs.append(model(batch_encoder, batch_decoder).numpy())
    return np.concatenate(outputs, axis=0)


def train_model(
    config: Config,
    splits: dict[str, SplitWindows],
    bundle: ScalerBundle,
    decoders: dict[str, np.ndarray] | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    announce: bool = True,
) -> TrainingArtifacts:
    """Fit the forecaster with early stopping on validation loss.

    Args:
        config: loaded pipeline configuration.
        splits: windowed tensors keyed by split name.
        bundle: fitted scalers, used only to report a masked validation MAE in W/m^2.
        decoders: optional replacement decoder inputs keyed by split name. The optional
            perfect-forecast upper bound of spec section 13 uses this to hand the model
            tomorrow's observed weather. The primary model must never use it.
        max_epochs: override for the configured epoch cap. Gate 1 uses a small value.
        patience: override for the configured early stopping patience.
        announce: whether to print the parameter count at startup.

    Returns:
        The trained model with best weights restored, plus its loss history.

    Raises:
        ValueError: if the training or validation split holds no windows.
    """
    training = config.training
    epochs = int(training["max_epochs"] if max_epochs is None else max_epochs)
    stop_patience = int(
        training["early_stopping_patience"] if patience is None else patience
    )
    batch_size = int(training["batch_size"])
    device = torch.device(str(training["device"]))
    threshold = float(config.evaluation["daylight_threshold"])
    clip = bool(config.evaluation["clip_predictions_at_zero"])

    train_decoder = (
        splits["train"].decoder if decoders is None else decoders["train"]
    )
    val_decoder = splits["val"].decoder if decoders is None else decoders["val"]

    if splits["train"].n_samples == 0 or splits["val"].n_samples == 0:
        raise ValueError(
            f"training needs both splits populated, got {splits['train'].n_samples} train "
            f"and {splits['val'].n_samples} validation windows"
        )

    generator = seed_everything(config.seed)

    train_loader = DataLoader(
        _tensors(splits["train"].encoder, train_decoder, splits["train"].target),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
        drop_last=False,
    )

    network = model_module.build(
        config,
        n_encoder_features=splits["train"].encoder.shape[2],
        n_decoder_features=train_decoder.shape[2],
        announce=announce,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        network.parameters(), lr=float(training["learning_rate"])
    )

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(network.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        network.train()
        running_loss = 0.0
        running_count = 0
        for batch_encoder, batch_decoder, batch_target in train_loader:
            optimizer.zero_grad()
            prediction = network(batch_encoder.to(device), batch_decoder.to(device))
            loss = criterion(prediction, batch_target.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_target)
            running_count += len(batch_target)
        train_loss = running_loss / running_count

        val_scaled = predict_scaled(
            network, splits["val"].encoder, val_decoder, batch_size
        )
        val_loss = float(
            np.mean((val_scaled - splits["val"].target.astype("float64")) ** 2)
        )
        val_mae = masked_mae(
            to_physical(val_scaled, bundle, clip),
            splits["val"].target_actual,
            threshold,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_masked_mae_wm2": val_mae,
            }
        )
        LOGGER.info(
            "epoch %3d: train MSE %.6f, val MSE %.6f, val daylight MAE %.2f W/m^2",
            epoch,
            train_loss,
            val_loss,
            val_mae,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(network.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= stop_patience:
                LOGGER.info(
                    "early stopping at epoch %d, no improvement for %d epochs",
                    epoch,
                    stop_patience,
                )
                break

    # Restore the best weights so the returned model is the one validation selected,
    # not whatever the last epoch happened to leave behind.
    network.load_state_dict(best_state)
    LOGGER.info(
        "restored best weights from epoch %d with val MSE %.6f", best_epoch, best_val_loss
    )

    return TrainingArtifacts(
        model=network,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        n_parameters=model_module.count_parameters(network),
    )


def save_artifacts(
    config: Config,
    artifacts: TrainingArtifacts,
    splits: dict[str, SplitWindows],
    model_name: str = MODEL_NAME,
) -> None:
    """Persist the checkpoint, the resolved config, and the loss history.

    Args:
        config: loaded pipeline configuration.
        artifacts: the training result.
        splits: windows, used to record the realised input widths.
        model_name: label used in the checkpoint filename for non-primary variants.
    """
    def variant(key: str, pattern: str) -> Path:
        """Path for this model's copy of a configured artefact.

        The primary model keeps the exact filenames the spec names. Any other variant gets
        its name prefixed, so the upper bound cannot overwrite the headline checkpoint.
        Deriving all three paths through one rule keeps them from drifting apart.
        """
        path = config.path(key)
        return path if model_name == MODEL_NAME else path.with_name(pattern)

    checkpoint_path = variant("checkpoint", f"{model_name}_best.pt")
    torch.save(artifacts.model.state_dict(), checkpoint_path)
    LOGGER.info("wrote checkpoint to %s", checkpoint_path)

    config_path = variant("checkpoint_config", f"{model_name}_best_config.json")
    record = {
        "model_name": model_name,
        "config": config.raw,
        "config_source": str(config.source),
        "n_parameters": artifacts.n_parameters,
        "best_epoch": artifacts.best_epoch,
        "best_val_loss_scaled_mse": artifacts.best_val_loss,
        "epochs_run": int(artifacts.history["epoch"].max()),
        "encoder_shape": list(splits["train"].encoder.shape[1:]),
        "decoder_shape": list(splits["train"].decoder.shape[1:]),
        "n_train_windows": splits["train"].n_samples,
        "n_val_windows": splits["val"].n_samples,
        "n_test_windows": splits["test"].n_samples,
    }
    config_path.write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    LOGGER.info("wrote resolved config to %s", config_path)

    loss_path = variant("loss_log", f"{model_name}_loss_log.csv")
    artifacts.history.to_csv(loss_path, index=False)
    LOGGER.info("wrote loss history to %s", loss_path)


def run(
    config: Config,
    splits: dict[str, SplitWindows] | None = None,
    bundle: ScalerBundle | None = None,
    upper_bound: bool = False,
) -> TrainingArtifacts:
    """Train a model, persist everything, and write its predictions.

    Args:
        config: loaded pipeline configuration.
        splits: optional pre-loaded windows.
        bundle: optional pre-loaded scalers.
        upper_bound: when True, train the spec section 13 perfect-forecast variant, which
            receives tomorrow's observed weather as decoder input. This is not a forecast
            and is labelled `lstm_upperbound` everywhere it appears.

    Returns:
        The training result.
    """
    windows = load_windows(config) if splits is None else splits
    scalers = load_scalers(config) if bundle is None else bundle

    model_name = UPPER_BOUND_NAME if upper_bound else MODEL_NAME
    decoders: dict[str, np.ndarray] | None = None
    if upper_bound:
        decoders = build_upperbound_decoders(
            config, load_features(config), scalers, windows
        )

    artifacts = train_model(config, windows, scalers, decoders=decoders)
    save_artifacts(config, artifacts, windows, model_name=model_name)

    batch_size = int(config.training["batch_size"])
    clip = bool(config.evaluation["clip_predictions_at_zero"])
    for split in ("val", "test"):
        decoder = windows[split].decoder if decoders is None else decoders[split]
        scaled = predict_scaled(
            artifacts.model, windows[split].encoder, decoder, batch_size
        )
        physical = to_physical(scaled, scalers, clip)
        path = config.prediction_path(model_name, split)
        np.save(path, physical)
        LOGGER.info(
            "wrote %s %s predictions of shape %s to %s",
            model_name,
            split,
            physical.shape,
            path,
        )

    return artifacts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m src.train`."""
    parser = argparse.ArgumentParser(
        description="Train the LSTM forecaster with early stopping on validation loss."
    )
    parser.add_argument(
        "--config", default=None, help="path to config.yaml, defaults to project root"
    )
    parser.add_argument(
        "--upper-bound",
        action="store_true",
        help=(
            "train the spec section 13 perfect-forecast variant, which is given tomorrow's "
            "observed weather. Not a forecast, never the headline model."
        ),
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load(args.config)
    artifacts = run(config, upper_bound=args.upper_bound)

    final = artifacts.history.iloc[artifacts.best_epoch - 1]
    print(
        f"Training complete: {artifacts.n_parameters:,} parameters, "
        f"{int(artifacts.history['epoch'].max())} epochs run, "
        f"best epoch {artifacts.best_epoch} with validation MSE "
        f"{artifacts.best_val_loss:.6f} and daylight MAE "
        f"{final['val_masked_mae_wm2']:.2f} W/m^2."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
