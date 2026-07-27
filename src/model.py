"""The forecasting network: an LSTM encoder with an MLP head.

Architecture, and the reasoning behind each piece:

    Encoder. A two layer LSTM over the 24 hours of past-observed input. Recurrence is
    the natural fit here because yesterday's irradiance trajectory carries the state a
    forecast depends on, cloudiness persistence above all, and an LSTM summarises that
    trajectory into a fixed vector without being told which lags matter.

    Fusion. The encoder's final hidden state is concatenated with the flattened
    decoder-known calendar features. The hidden state answers "what has the sky been
    doing", and the calendar block answers "which 24 hours am I being asked about".
    Concatenation keeps those two questions separable, which matters for interpretation:
    the calendar path can learn the clear-sky envelope while the recurrent path learns
    the departure from it.

    Head. One hidden layer of 128 ReLU units to a 24 dimensional output. All 24 hours are
    emitted at once, direct multi-output rather than recursive. WHY direct: a recursive
    decoder would feed its own hour-one prediction into hour two, compounding its error
    across the horizon, and it would need a GHI input for tomorrow that by construction
    does not exist. Direct output sidesteps both problems.

The model is deliberately small. The training set holds roughly 35000 windows, and the
research question is whether a compact model can beat persistence, not how large a model
can be made to fit.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from src.config import Config

LOGGER = logging.getLogger(__name__)


class LSTMForecaster(nn.Module):
    """LSTM encoder plus MLP head producing a 24 hour GHI forecast.

    Args:
        n_encoder_features: channel count of the encoder input.
        n_decoder_features: channel count of the decoder-known input.
        horizon_hours: number of hours to predict, which is also the decoder length.
        hidden_size: LSTM hidden state width.
        num_layers: number of stacked LSTM layers.
        dropout: dropout probability applied between LSTM layers.
        head_hidden: width of the MLP head's single hidden layer.

    Shapes:
        encoder input  (batch, encoder_hours, n_encoder_features)
        decoder input  (batch, horizon_hours, n_decoder_features)
        output         (batch, horizon_hours), in scaled GHI units
    """

    def __init__(
        self,
        n_encoder_features: int,
        n_decoder_features: int,
        horizon_hours: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        head_hidden: int,
    ) -> None:
        super().__init__()
        self.horizon_hours = horizon_hours

        self.encoder = nn.LSTM(
            input_size=n_encoder_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch applies this between stacked layers, not after the last one, which
            # is the placement the spec asks for.
            dropout=dropout if num_layers > 1 else 0.0,
        )

        fusion_width = hidden_size + horizon_hours * n_decoder_features
        self.head = nn.Sequential(
            nn.Linear(fusion_width, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, horizon_hours),
        )

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """Produce a 24 hour scaled GHI forecast.

        Args:
            encoder_input: past-observed features, shape (batch, encoder_hours, features).
            decoder_input: known calendar features for the target window,
                shape (batch, horizon_hours, calendar_features).

        Returns:
            Scaled GHI predictions of shape (batch, horizon_hours).
        """
        _, (hidden_state, _) = self.encoder(encoder_input)
        # hidden_state has shape (num_layers, batch, hidden_size). The last layer's state
        # is the encoder's summary of the input sequence.
        summary = hidden_state[-1]
        known = decoder_input.flatten(start_dim=1)
        return self.head(torch.cat([summary, known], dim=1))


def count_parameters(module: nn.Module) -> int:
    """Total number of trainable parameters.

    Args:
        module: any torch module.

    Returns:
        Count of elements across all parameters that require gradients.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build(
    config: Config,
    n_encoder_features: int,
    n_decoder_features: int,
    announce: bool = True,
) -> LSTMForecaster:
    """Construct the model from config and report its size.

    Args:
        config: loaded pipeline configuration.
        n_encoder_features: channel count of the encoder input, from the realised windows.
        n_decoder_features: channel count of the decoder input, from the realised windows.
        announce: whether to print the parameter count, which the spec asks for at startup.

    Returns:
        An initialised LSTMForecaster on the CPU.

    The feature widths are passed in rather than read from config because Gate 0 may drop a
    weather column, which changes the encoder width. Deriving the width from the actual
    windows keeps the model and the data in agreement by construction.
    """
    settings = config.model
    model = LSTMForecaster(
        n_encoder_features=n_encoder_features,
        n_decoder_features=n_decoder_features,
        horizon_hours=int(config.windowing["horizon_hours"]),
        hidden_size=int(settings["hidden_size"]),
        num_layers=int(settings["num_layers"]),
        dropout=float(settings["dropout"]),
        head_hidden=int(settings["head_hidden"]),
    )

    total = count_parameters(model)
    LOGGER.info(
        "model built: encoder %d features, decoder %d features, %d trainable parameters",
        n_encoder_features,
        n_decoder_features,
        total,
    )
    if announce:
        print(f"Model parameter count: {total:,} trainable parameters.")
    return model
