from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chronos_bolt import (
    ChronosBoltModelForForecasting,
    ChronosBoltConfig,
)


class ChronosBoltModelForSimilarity(
    ChronosBoltModelForForecasting
):

    def __init__(
        self,
        config: ChronosBoltConfig,
        embedding_dim: int = 128,
        dropout_head: float = 0.1,
        pooling: str = "mean",
        temperature: float = 0.1,
    ):
        super().__init__(config)

        self.pooling = pooling
        self.temperature = temperature

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than zero."
            )

        hidden_size = getattr(
            self.config, "hidden_size", None
        ) or getattr(
            self.config, "d_model", None
        )

        if hidden_size is None:
            raise ValueError(
                "Cannot determine encoder hidden size from config."
            )

        self.dropout = nn.Dropout(dropout_head)

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_size, embedding_dim),
        )

    def encode_series(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        (
            hidden_states,
            _,
            _,
            _,
            attention_mask,
            _,
        ) = super().encode(
            context=context,
            mask=mask,
        )

        # hidden_states: [batch, patches, hidden_size]
        # attention_mask: [batch, patches]

        # Exclude the REG token if enabled.
        if self.chronos_config.use_reg_token:
            hidden_states = hidden_states[:, :-1, :]
            attention_mask = attention_mask[:, :-1]

        if self.pooling == "mean":
            weights = attention_mask.unsqueeze(-1).to(
                dtype=hidden_states.dtype
            )

            pooled = (hidden_states * weights).sum(dim=1)

            pooled = pooled / weights.sum(
                dim=1
            ).clamp_min(1.0)

        elif self.pooling == "cls":
            # Bolt does not use a BERT-style CLS token.
            # This option uses the first patch representation.
            pooled = hidden_states[:, 0, :]

        else:
            raise ValueError(
                f"Unknown pooling method: {self.pooling}"
            )

        # Projection head
        embedding = self.projection(
            self.dropout(pooled)
        )

        # Normalize for cosine similarity.
        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding

    def forward(
        self,
        context_1: torch.Tensor,
        context_2: torch.Tensor,
        mask_1: Optional[torch.Tensor] = None,
        mask_2: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):

        z1 = self.encode_series(
            context=context_1,
            mask=mask_1,
        )

        z2 = self.encode_series(
            context=context_2,
            mask=mask_2,
        )

        similarity = F.cosine_similarity(
            z1,
            z2,
            dim=-1,
        )

        loss = None

        if labels is not None:
            target = labels.to(
                device=similarity.device,
                dtype=similarity.dtype,
            ).reshape_as(similarity)

            loss = F.binary_cross_entropy_with_logits(
                similarity / self.temperature,
                target,
            )

        return {
            "loss": loss,
            "embeddings_1": z1,
            "embeddings_2": z2,
            "similarity": similarity,
        }