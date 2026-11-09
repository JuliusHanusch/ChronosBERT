import torch
import torch.nn as nn

from .chronos_bolt import (
    ChronosBoltModelForForecasting,
    ChronosBoltConfig,
)


class ChronosBoltModelForClassification(ChronosBoltModelForForecasting):
    def __init__(
        self,
        config: ChronosBoltConfig,
        num_labels: int = 2,
        dropout_head: float = 0.1,
        pooling="mean",

    ):
        super().__init__(config)

        self.num_labels = num_labels

        hidden_size = getattr(
            self.config, "hidden_size", None
        ) or getattr(self.config, "d_model", None)

        if hidden_size is None:
            raise ValueError(
                "Cannot determine encoder hidden size from config."
            )

        self.dropout = nn.Dropout(dropout_head)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, context, mask=None, labels=None):
        (
            hidden_states,
            _,
            _,
            _,
            attention_mask,
            _,
        ) = self.encode(
            context=context,
            mask=mask,
        )

        # Exclude REG token if enabled.
        if self.chronos_config.use_reg_token:
            hidden_states = hidden_states[:, :-1, :]
            attention_mask = attention_mask[:, :-1]

        # Mean-pool only valid patches.
        weights = attention_mask.unsqueeze(-1).to(
            dtype=hidden_states.dtype
        )

        pooled = (hidden_states * weights).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0)

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.long())

        return {"loss": loss, "logits": logits}



