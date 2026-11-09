# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>, Caner Turkmen <atturkm@amazon.com>, Lorenzo Stella <stellalo@amazon.com>
# Original source:
# https://github.com/autogluon/autogluon/blob/f57beb26cb769c6e0d484a6af2b89eab8aee73a8/timeseries/src/autogluon/timeseries/models/chronos/pipeline/chronos_bolt.py

import copy
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Config,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
)
from transformers import BertConfig, PreTrainedModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder

from transformers.models.bert.modeling_bert import BertEncoder

from transformers.models.roberta.modeling_roberta import (
    RobertaEncoder
)
from transformers.utils import ModelOutput

#from .base import BaseChronosPipeline, ForecastType

#colab
import chronos.chronos as chronos
from chronos.base import BaseChronosPipeline, ForecastType

logger = logging.getLogger(__file__)


@dataclass
class ChronosBoltConfig:
    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False
    individual_dropout: float = 0.0
    masking_prob: float = 0.0
    unchanged_patch_prob: float = 0.0
    patch_masking_prob: float = 0.0
    patch_perturbation_prob: float = 0.0
    patch_perturbation_scale_strength: float = 0.0
    patch_perturbation_noise_strength: float = 0.0
    model_id: str = "T5"  # "RoBERTa" or "T5"
    debug_patching: bool = False
    debug_batch_idx: int = 0
    debug_max_patches: int = 3


@dataclass
class ChronosBoltOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    """
    See, also, RevIN. Apply standardization along the last dimension.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(
        self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class ChronosBoltModelForForecasting(PreTrainedModel):
    config_class = BertConfig
    _keys_to_ignore_on_load_missing = [  # type: ignore
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]  # type: ignore
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]  # type: ignore

    def __init__(self, config):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        super().__init__(config)

        # Naming difference in BERT models
        self.model_dim = config.hidden_size
        # Chronos-owned logic
        self.chronos_config = ChronosBoltConfig(**config.chronos_config)
        self.debug_patching = self.chronos_config.debug_patching
        self.debug_batch_idx = self.chronos_config.debug_batch_idx
        self.debug_max_patches = self.chronos_config.debug_max_patches


        # Only decoder_start_id (and optionally REG token)
        config.mask_token_id = 1
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1
            config.mask_token_id = 2

        config.vocab_size = 3 if self.chronos_config.use_reg_token else 2 # Added one for [MASK] token
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size)

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.intermediate_size,
            out_dim=config.hidden_size,
            act_fn_name=config.hidden_act,
            dropout_p=config.hidden_dropout_prob,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm()

        
        #print("model_id_chronos_config", self.chronos_config.model_id)

        encoder_config = copy.deepcopy(config)
        if self.chronos_config.model_id == "FacebookAI/roberta-base":
            roberta_config = AutoConfig.from_pretrained("FacebookAI/roberta-base") # TODO allow custom configs
            for key, value in encoder_config.__dict__.items():
                if hasattr(roberta_config, key):
                    setattr(roberta_config, key, value)
            roberta_config.hidden_size = config.hidden_size
            roberta_config.num_attention_heads = config.num_attention_heads
            roberta_config.num_hidden_layers = config.num_hidden_layers
            roberta_config.intermediate_size = config.intermediate_size
            roberta_config.hidden_dropout_prob = config.hidden_dropout_prob
            roberta_config.hidden_act = config.hidden_act
            num_patches = (
                config.chronos_config["context_length"]
                // config.chronos_config["input_patch_stride"]
            )

            roberta_config.max_position_embeddings = num_patches + int(config.chronos_config.get("use_reg_token", False))
            roberta_config.layer_norm_eps = config.layer_norm_eps
            roberta_config.is_decoder = False
            self.encoder = RobertaEncoder(roberta_config)

        elif "bert" in self.chronos_config.model_id.lower(): # TODO BERT Base
            bert_config = AutoConfig.from_pretrained(self.chronos_config.model_id.lower())
            for key, value in encoder_config.__dict__.items():
                if hasattr(bert_config, key):
                    setattr(bert_config, key, value)
            bert_config.hidden_size = config.hidden_size
            bert_config.num_attention_heads = config.num_attention_heads
            bert_config.num_hidden_layers = config.num_hidden_layers
            bert_config.intermediate_size = config.intermediate_size
            bert_config.hidden_dropout_prob = config.hidden_dropout_prob
            bert_config.hidden_act = config.hidden_act
            num_patches = (
                config.chronos_config["context_length"]
                // config.chronos_config["input_patch_stride"]
            )

            bert_config.max_position_embeddings = num_patches + int(config.chronos_config.get("use_reg_token", False))
            bert_config.layer_norm_eps = config.layer_norm_eps
            bert_config.is_decoder = False
            self.encoder = BertEncoder(bert_config)

        else:
            encoder_config.is_decoder = False
            encoder_config.use_cache = False
            encoder_config.is_encoder_decoder = False
            self.encoder = T5Stack(encoder_config, self.shared)

        self.position_embeddings = nn.Embedding(
            bert_config.max_position_embeddings,
            bert_config.hidden_size,
        )

        self.bert_embedding_layer_norm = nn.LayerNorm(
            bert_config.hidden_size,
            eps=bert_config.layer_norm_eps,
        )

        self.bert_embedding_dropout = nn.Dropout(
            bert_config.hidden_dropout_prob,
        )



        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.hidden_size,
            h_dim=config.intermediate_size,
            out_dim=self.num_quantiles * self.chronos_config.input_patch_size,
            act_fn_name=config.hidden_act,
            dropout_p=config.hidden_dropout_prob,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        
        self.patch_perturbation = self.chronos_config.patch_perturbation_prob * self.chronos_config.masking_prob
        self.patch_masking = self.chronos_config.patch_masking_prob * self.chronos_config.masking_prob
        self.patch_preservation = self.chronos_config.unchanged_patch_prob * self.chronos_config.masking_prob
        # Need to add up to 1.0 except when masking prob == 0
        assert int((self.patch_perturbation + self.patch_masking + self.patch_preservation)*100) == int(self.chronos_config.masking_prob * 100)

    def _init_weights(self, module):
        
        if isinstance(self.config, T5Config):
            T5PreTrainedModel._init_weights(self, module)
        elif isinstance(self.config, BertConfig):
            BertPreTrainedModel._init_weights(self, module)

        #print("Config:", print(type(self.config).__name__))

        if isinstance(self.config, T5Config):
            factor = self.config.initializer_factor
            #print("Config: Confirmed T5")

        if isinstance(self.config, BertConfig):
            factor = self.config.initializer_range = 0.05
            #print("Config: Confirmed Bert")


        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if (
                hasattr(module.hidden_layer, "bias")
                and module.hidden_layer.bias is not None
            ):
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if (
                hasattr(module.residual_layer, "bias")
                and module.residual_layer.bias is not None
            ):
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.intermediate_size) ** -0.5)
            )
            if (
                hasattr(module.output_layer, "bias")
                and module.output_layer.bias is not None
            ):
                module.output_layer.bias.data.zero_()

    def encode(
        self, context: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]: # TODO If necessary pass mask and apply all iterations only to patches inside mask --> Remove train_attention_mask
        if self.debug_patching:
            print("\n=== RAW CONTEXT ===")
            print("context.shape:", context.shape)
            print("context[0, :20]:", context[0, :20])

        mask = (
            mask.to(context.dtype)
            if mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]

        # scaling
        context, loc_scale = self.instance_norm(context)

        if self.debug_patching:
            print("\n=== SCALED CONTEXT ===")
            print("context.shape:", context.shape)
            print("mean (sample 0):", context[0].mean().item())
            print("std  (sample 0):", context[0].std().item())

        target = self.patch(context)

        # the scaling op above is done in 32-bit precision,
        # then the context is moved to model's dtype
        context = context.to(self.dtype)
        orig_mask = mask.clone()

        individual_mask = torch.ones_like(mask)
        if self.training and self.chronos_config.individual_dropout > 0.0:
            # mask out individual values
            individual_mask = torch.bernoulli(torch.full_like(mask, 1 - self.chronos_config.individual_dropout))
            mask = mask * individual_mask
        mask = mask.to(self.dtype)
        
        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        if self.debug_patching:
            print("\n=== PATCHED CONTEXT ===")
            print("patched_context.shape:", patched_context.shape)
            print("patched_mask.shape:", patched_mask.shape)

            b = self.chronos_config.debug_batch_idx
            k = self.chronos_config.debug_max_patches

            print("\nFirst patches (values):")
            for i in range(k):
                print(f"patch[{i}]:", patched_context[b, i])

            print("\nFirst patches (mask):")
            for i in range(k):
                print(f"mask[{i}]:", patched_mask[b, i])

        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        # concat context and mask along patch dim
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        if self.debug_patching:
            print("\n=== PATCH + MASK CONCAT ===")
            print("patched_context.shape:", patched_context.shape)
            print("expected last dim:", self.chronos_config.input_patch_size * 2)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = (
            patched_mask.sum(dim=-1) > 0
        )  # (batch_size, patched_seq_length)

        inputs_embeds = self.input_patch_embedding(patched_context)

        if self.debug_patching:
            print("\n=== PATCH EMBEDDINGS ===")
            print("inputs_embeds.shape:", inputs_embeds.shape)
            print("embed[0, 0, :8]:", inputs_embeds[0, 0, :8])

        # Mask out entire patches during training
        patch_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.bool)


        if self.training and self.chronos_config.masking_prob > 0.0:
            # selected MLM/MAE target patches: loss is computed here
            prediction_patch_mask = (
                torch.rand(inputs_embeds.shape[:-1], device=inputs_embeds.device)
                < self.chronos_config.masking_prob
            )
            


            # BERT-style split among selected patches
            u = torch.rand(inputs_embeds.shape[:-1], device=inputs_embeds.device)

            mask_replace = prediction_patch_mask & (u < self.chronos_config.patch_masking_prob)

            perturb_replace = (
                prediction_patch_mask
                & (u >= self.chronos_config.patch_masking_prob)
                & (u < self.chronos_config.patch_masking_prob + self.chronos_config.patch_perturbation_prob)
            )

            # unchanged_replace = (
            #     prediction_patch_mask
            #     & ~(mask_replace | perturb_replace)
            # )

            # 1) replace with [MASK]
            mask_input_ids = torch.full(
                (1, 1),
                self.config.mask_token_id,
                device=inputs_embeds.device,
            )
            mask_embeds = self.shared(mask_input_ids)

            inputs_embeds = torch.where(
                mask_replace.unsqueeze(-1),
                mask_embeds,
                inputs_embeds,
            )

            # 2) perturb selected patches
            noise = torch.randn_like(inputs_embeds) * self.chronos_config.patch_perturbation_noise_strength

            scalar = 1 + (
                torch.rand(
                    inputs_embeds.size(0),
                    inputs_embeds.size(1),
                    1,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                ) * 2 - 1
            ) * self.chronos_config.patch_perturbation_scale_strength

            perturbed_embeds = (inputs_embeds + noise) * scalar

            inputs_embeds = torch.where(
                perturb_replace.unsqueeze(-1),
                perturbed_embeds,
                inputs_embeds,
            )

        else: 
            prediction_patch_mask = torch.zeros(
                inputs_embeds.shape[:-1],
                dtype=torch.bool,
                device=inputs_embeds.device,
            )




        train_attention_mask = None
        if self.training:
            patch_mask_flat = prediction_patch_mask.repeat_interleave(
                self.chronos_config.input_patch_size,
                dim=1,
            )

            # assert patch_mask_flat.shape == orig_mask.shape, "Shape mismatch"
            # assert torch.all(~patch_mask_flat | orig_mask.bool()) == torch.all(
            #     patch_mask_flat & orig_mask.bool() == patch_mask_flat
            # ), "Invalid positions selected"

            # valid_per_sample = orig_mask.bool().sum(dim=1)

            # print("Valid positions per sample:")
            # print("  min:", valid_per_sample.min().item())
            # print("  mean:", valid_per_sample.float().mean().item())
            # print("  max:", valid_per_sample.max().item())

            train_attention_mask = patch_mask_flat & orig_mask.bool()

            # print("Selected patches:", prediction_patch_mask.sum().item())
            # print("Selected valid positions:", train_attention_mask.sum().item())
            # print("Mask check passed:", torch.all(train_attention_mask <= orig_mask.bool()).item())

            # print("Selected patches:", prediction_patch_mask.sum().item())
            # print("Total patches:", prediction_patch_mask.numel())
            # print("Valid original positions:", orig_mask.bool().sum().item())
            # print("Selected positions before validity mask:", patch_mask_flat.sum().item())
            # print("Selected valid positions:", train_attention_mask.sum().item())

            # align in case patch padding changed length
            train_attention_mask = train_attention_mask[..., -context.shape[-1]:]


        if self.chronos_config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=inputs_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            inputs_embeds = torch.cat([inputs_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask.to(self.dtype),
                    torch.ones_like(reg_input_ids).to(self.dtype),
                ],
                dim=-1,
            )

        seq_len = inputs_embeds.size(1)

        position_ids = torch.arange(
            seq_len,
            device=inputs_embeds.device,
        ).unsqueeze(0)

        token_type_ids = torch.zeros(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )

        inputs_embeds = (
            inputs_embeds
            + self.position_embeddings(position_ids)
        )

        inputs_embeds = self.bert_embedding_layer_norm(inputs_embeds)
        inputs_embeds = self.bert_embedding_dropout(inputs_embeds)
        
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            attention_mask.shape,
        )

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            return_dict=True,
        )

        # TODO attention_mask=attention_mask,
        if self.debug_patching:
            print("\n=== ENCODER OUTPUT ===")
            print("hidden_states.shape:", encoder_outputs[0].shape)


        return encoder_outputs[0], target, loc_scale, inputs_embeds, attention_mask, train_attention_mask


    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBoltOutput:
        batch_size = context.size(0)

        hidden_states, target, loc_scale, inputs_embeds, attention_mask, train_attention_mask = self.encode(
            context=context, mask=mask
        )
        # Hidden States = Roberta(inputs_embeds)


        num_patches = hidden_states.size(1)
        patch_size = self.chronos_config.input_patch_size

        quantile_preds = self.output_patch_embedding(hidden_states)
        quantile_preds = quantile_preds.view(
            batch_size,
            num_patches,
            self.num_quantiles,
            patch_size,
        )
        quantile_preds = quantile_preds.permute(0, 2, 1, 3).contiguous()
        quantile_preds = quantile_preds.view(
            batch_size,
            self.num_quantiles,
            num_patches * patch_size,
        )

        quantile_preds = quantile_preds[..., -context.shape[-1]:]

        loss = None
        if target is not None:

            target_scaled = target.reshape((quantile_preds.shape[0],quantile_preds.shape[-1])).unsqueeze(1) # type: ignore
            target_scaled = target_scaled.to(quantile_preds.device)
            target_mask = (~torch.isnan(target_scaled))
            target_scaled[~target_mask] = 0.0

            #print("target_scaled", target_scaled.mean())
            #print("target_scaled shape",target_scaled.shape)            
            #print("quantile_preds shape", quantile_preds.mean())
            #print("quantile_preds", quantile_preds.shape)


            loss = (
                2
                * torch.abs(
                    (target_scaled - quantile_preds)
                    * (
                        (target_scaled <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)
                    )
                )
            )

            # -----------------------------
            # APPLY TRAIN ATTENTION MASK
            # -----------------------------
            #effective_mask = target_mask.float().expand_as(loss)
            effective_mask = target_mask.float()

            if train_attention_mask is not None:
                train_attention_mask = train_attention_mask[..., -loss.shape[-1]:]
                effective_mask = effective_mask * train_attention_mask.unsqueeze(1).float()

            loss = loss.mean(dim=1)
            loss = loss * effective_mask.squeeze(1)
            loss = loss.sum() / effective_mask.sum().clamp_min(1.0)
            #loss = (loss * effective_mask).sum() / effective_mask.sum().clamp_min(1.0)

        quantile_preds_unscaled = self.instance_norm.inverse(
            quantile_preds.reshape(batch_size, -1),
            loc_scale,
        ).reshape_as(quantile_preds)
        return ChronosBoltOutput(
            loss=loss,
            quantile_preds=quantile_preds_unscaled,
        )


class ChronosBoltPipeline(BaseChronosPipeline):
    forecast_type: ForecastType = ForecastType.QUANTILES
    default_context_length: int = 2048

    def __init__(self, model: ChronosBoltModelForForecasting):
        super().__init__(inner_model=model)  # type: ignore
        self.model = model

    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]

    @torch.no_grad()
    def embed(
        self, context: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get encoder embeddings for the given time series.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.

        Returns
        -------
        embeddings, loc_scale
            A tuple of two items: the encoder embeddings and the loc_scale,
            i.e., the mean and std of the original time series.
            The encoder embeddings are shaped (batch_size, num_patches + 1, d_model),
            where num_patches is the number of patches in the time series
            and the extra 1 is for the [REG] token (if used by the model).
        """
        context_tensor = self._prepare_and_validate_context(context=context)
        model_context_length = self.model.config.chronos_config["context_length"]

        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        context_tensor = context_tensor.to(
            device=self.model.device,
            dtype=torch.float32,
        )
        embeddings, loc_scale, *_ = self.model.encode(context=context_tensor)
        return embeddings.cpu(), (
            loc_scale[0].squeeze(-1).cpu(),
            loc_scale[1].squeeze(-1).cpu(),
        )

    def predict(  # type: ignore[override]
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
    ) -> torch.Tensor:
        """
        Get forecasts for the given time series.

        Refer to the base method (``BaseChronosPipeline.predict``)
        for details on shared parameters.
        Additional parameters
        ---------------------
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. False by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        torch.Tensor
            Forecasts of shape (batch_size, num_quantiles, prediction_length)
            where num_quantiles is the number of quantiles the model has been
            trained to output. For official Chronos-Bolt models, the value of
            num_quantiles is 9 for [0.1, 0.2, ..., 0.9]-quantiles.

        Raises
        ------
        ValueError
            When limit_prediction_length is True and the prediction_length is
            greater than model's trainig prediction_length.
        """
        context_tensor = self._prepare_and_validate_context(context=context)

        model_context_length = self.model.config.chronos_config["context_length"]
        model_prediction_length = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # We truncate the context here because otherwise batches with very long
        # context could take up large amounts of GPU memory unnecessarily.
        if context_tensor.shape[-1] > model_context_length:
            context_tensor = context_tensor[..., -model_context_length:]

        # TODO: We unroll the forecast of Chronos Bolt greedily with the full forecast
        # horizon that the model was trained with (i.e., 64). This results in variance collapsing
        # every 64 steps.
        context_tensor = context_tensor.to(
            device=self.model.device,
            dtype=torch.float32,
        )
        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor,
                ).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]

            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32, device="cpu"
        )

    def predict_quantiles(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refer to the base method (``BaseChronosPipeline.predict_quantiles``).
        """
        # shape (batch_size, prediction_length, len(training_quantile_levels))
        predictions = (
            self.predict(context, prediction_length=prediction_length, **predict_kwargs)
            .detach()
            .swapaxes(1, 2)
        )

        training_quantile_levels = self.quantiles

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            # no need to perform intra/extrapolation
            quantiles = predictions[
                ..., [training_quantile_levels.index(q) for q in quantile_levels]
            ]
        else:
            # we rely on torch for interpolating quantiles if quantiles that
            # Chronos Bolt was trained on were not provided
            if min(quantile_levels) < min(training_quantile_levels) or max(
                quantile_levels
            ) > max(training_quantile_levels):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that Chronos-Bolt was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which Chronos-Bolt "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            # TODO: this is a hack that assumes the model's quantiles during training (training_quantile_levels)
            # made up an equidistant grid along the quantile dimension. i.e., they were (0.1, 0.2, ..., 0.9).
            # While this holds for official Chronos-Bolt models, this may not be true in the future, and this
            # function may have to be revised.
            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions,
                q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype),
                dim=-1,
            ).permute(1, 2, 0)
        # NOTE: the median is returned as the mean here
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean
    
    

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        print("From Pretrained")
        """
        Load the model, either from a local path S3 prefix or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel`` from ``transformers``.
        """

        if str(pretrained_model_name_or_path).startswith("s3://"):
            return BaseChronosPipeline.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        if class_ is None:
            logger.warning(f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecasting")
            class_ = ChronosBoltModelForForecasting

        model = class_.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(model=model)
