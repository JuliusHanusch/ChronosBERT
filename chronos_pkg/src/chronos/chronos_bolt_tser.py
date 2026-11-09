import torch
import torch.nn as nn
import torch.nn.functional as F

from chronos.chronos_bolt import ChronosBoltModelForForecasting


class ChronosBoltModelForTSER(ChronosBoltModelForForecasting):
    def __init__(
        self,
        config,
        dropout_head=0.1,
        loss_type="mse",
    ):
        super().__init__(config)

        self.dropout = nn.Dropout(dropout_head)
        self.loss_type = loss_type.lower()

        if self.loss_type not in {"mse", "mae"}:
            raise ValueError(
                f"Unsupported loss_type: {loss_type}. "
                "Choose 'mse', 'rmse', 'mae', or 'l1'."
            )

        # Infer the encoder hidden dimension from the config.
        candidates = [
            getattr(config, "hidden_size", None),
            getattr(config, "d_model", None),
        ]

        chronos_config = getattr(config, "chronos_config", None)
        if chronos_config is not None:
            candidates.extend([
                getattr(chronos_config, "hidden_size", None),
                getattr(chronos_config, "d_model", None),
            ])

        hidden_size = next(
            (int(value) for value in candidates if value is not None),
            None,
        )

        if hidden_size is None:
            raise ValueError(
                "Could not determine hidden size from the BOLT config. "
            )

        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, context, mask, labels=None):
        encoded = self.encode(context, mask)

        hidden_states = encoded[0]
        patch_mask = encoded[4]

        # Pool patch-level representations.
        if (
            torch.is_tensor(patch_mask)
            and patch_mask.ndim == 2
            and patch_mask.shape == hidden_states.shape[:2]
        ):
            weights = patch_mask.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            pooled = (
                hidden_states * weights.unsqueeze(-1)
            ).sum(dim=1) / weights.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
        else:
            pooled = hidden_states.mean(dim=1)

        logits = self.regressor(
            self.dropout(pooled)
        ).squeeze(-1)

        outputs = {"logits": logits}

        if labels is not None:
            labels = labels.to(
                device=logits.device,
                dtype=logits.dtype,
            ).reshape(-1)

            if self.loss_type in {"mae", "l1"}:
                loss = F.l1_loss(logits, labels)
            else:
                loss = F.mse_loss(logits, labels)

                if self.loss_type == "rmse":
                    loss = torch.sqrt(loss + 1e-12)

            outputs["loss"] = loss

        return outputs



from chronos.chronos_bolt import ChronosBoltModelForForecasting


class ChronosBoltModelForTSERMultivariate(
    ChronosBoltModelForForecasting
):
    def __init__(
        self,
        config,
        n_channels: int,
        dropout_head: float = 0.1,
        loss_type: str = "mse",
    ):
        super().__init__(config)

        self.n_channels = n_channels
        self.dropout_head = dropout_head
        self.loss_type = loss_type.lower()

        if self.loss_type not in {"mse", "rmse", "mae", "l1"}:
            raise ValueError(
                f"Unsupported loss_type: {loss_type}. "
                "Choose 'mse', 'rmse', 'mae', or 'l1'."
            )

        # Infer the encoder hidden dimension from the config.
        candidates = [
            getattr(config, "hidden_size", None),
            getattr(config, "d_model", None),
        ]

        chronos_config = getattr(config, "chronos_config", None)
        if chronos_config is not None:
            candidates.extend([
                getattr(chronos_config, "hidden_size", None),
                getattr(chronos_config, "d_model", None),
            ])

        hidden_size = next(
            (
                int(value)
                for value in candidates
                if value is not None
            ),
            None,
        )

        if hidden_size is None:
            raise ValueError(
                "Could not determine hidden size from the "
                "Chronos Bolt config."
            )

        self.hidden_size = hidden_size

        # Fuse the pooled representations of all channels.
        self.fusion = nn.Sequential(
            nn.Linear(
                n_channels * hidden_size,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_size, 1),
        )

    def encode_channel(
        self,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode(context, mask)

        hidden_states = encoded[0]
        patch_mask = encoded[4]

        # Pool valid patch representations.
        if (
            torch.is_tensor(patch_mask)
            and patch_mask.ndim == 2
            and patch_mask.shape == hidden_states.shape[:2]
        ):
            weights = patch_mask.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            pooled = (
                hidden_states * weights.unsqueeze(-1)
            ).sum(dim=1) / weights.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)

        else:
            # Fallback if no compatible patch mask is available.
            pooled = hidden_states.mean(dim=1)

        return pooled

    def forward(
        self,
        context: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):

        if context.ndim != 3:
            raise ValueError(
                "context must have shape [batch, channels, "
                f"time], but received {tuple(context.shape)}."
            )

        if mask.shape != context.shape:
            raise ValueError(
                "mask must have the same shape as context."
            )

        batch_size, n_channels, _ = context.shape

        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, "
                f"but received {n_channels}."
            )

        representations = []

        # Encode each channel independently using Chronos Bolt.
        for channel in range(n_channels):
            representation = self.encode_channel(
                context[:, channel, :],
                mask[:, channel, :],
            )

            representations.append(representation)

        # [B, C, H] -> [B, C * H]
        combined = torch.cat(
            representations,
            dim=-1,
        )

        # Predict the scalar regression target.
        logits = self.fusion(combined).squeeze(-1)

        outputs = {"logits": logits}

        if labels is not None:
            labels = labels.to(
                device=logits.device,
                dtype=logits.dtype,
            ).reshape(-1)

            if self.loss_type in {"mae", "l1"}:
                loss = F.l1_loss(logits, labels)
            else:
                loss = F.mse_loss(logits, labels)

                if self.loss_type == "rmse":
                    loss = torch.sqrt(loss + 1e-12)

            outputs["loss"] = loss

        return outputs
