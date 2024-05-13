"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import string
import random
import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


def flatten_text(text):
    x = np.array(text)
    return x.flatten().tolist(), x.shape


@registry.register_model("blip2_qformer2")
class Blip2Qformer2(Blip2Base):
    """
    BLIP2 Q-Former model.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_qformer2", "flant5xl")
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }


    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        prompt=None,
    ):
        """
        """
        super().__init__()

        # ------------------------------ Vision Encoder ------------------------------ #

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # --------------------------------- Q-Former --------------------------------- #

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.prompt = None


    def forward(self, samples):
        image = samples["image"]
        prompt = samples["text_input"]
        captions = samples["caption"]
        text_class = samples["text_class"]
        text_match = samples["text_match"]
        itc_targets = samples["text_class_targets"]
        itm_targets = samples["text_match_targets"]

        # encode image
        image_embeds, image_atts = self._encode_vision(image)  # [64, 514, 1408], [64, 514]
        query_tokens, query_atts = self._get_query_tokens(image)  # [64, 32, 768], [64, 32]
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True)
        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)

        # encode pos_text
        text_class_flat, (batch_size, n_samples, n_class) = flatten_text(text_class)

        text_tokens_itc = self._get_qformer_text_tokens(text_class_flat, image)
        text_output_itc = self.Qformer.bert(
            text_tokens_itc.input_ids,
            attention_mask=text_tokens_itc.attention_mask,
            return_dict=True)
        text_feats_itc = F.normalize(self.text_proj(text_output_itc.last_hidden_state[:, 0, :]), dim=-1)
        text_feats_itc = text_feats_itc.reshape(batch_size, n_samples, n_class, text_feats_itc.shape[-1])

        # encode image and text
        text_match_flat, (_, n_match) = flatten_text(text_match)

        image_embeds_itc = image_embeds.repeat_interleave(n_match, 0)
        image_atts_itc = image_atts.repeat_interleave(n_match, 0)
        
        text_match_tokens = self._get_qformer_text_tokens(text_match_flat, image)
        query_tokens_itm, query_atts_itm = self._get_query_tokens(image_embeds_itc)

        itm_output = self.Qformer.bert(
            text_match_tokens.input_ids, #[192, 32]
            query_embeds=query_tokens_itm, #[192, 32, 768]
            attention_mask=torch.cat([query_atts_itm, text_match_tokens.attention_mask], dim=1), #[192, 64]
            encoder_hidden_states=image_embeds_itc, #[192, 514, 1408]
            encoder_attention_mask=image_atts_itc, #[192, 514]
            return_dict=True,
        )

        # text-contrastive learning
        # [64, 2, 256], [64, 3, 256]
        # [64, 2, 1, 256] + [64, 2, 3, 256] > [64, 2, 4, 256]
        # [64, 32, 256], [64, 2, 4, 256] > [64, 2, 4, 32]
        # [64, 2, 4, 32] > [64, 2, 4] > [128, 4]
        sim = torch.einsum('ijk,ilmk->ilmj', image_feats, text_feats_itc)
        sim, _ = sim.max(-1)
        sim = sim / self.temp

        sim = sim.reshape(-1, sim.shape[-1])
        itc_targets = itc_targets.reshape(-1)        
        loss_itc = F.cross_entropy(sim, itc_targets, label_smoothing=0.1)

        # per-sample ITM
        # [192, 32, x]
        vl_embeds = itm_output.last_hidden_state[:, :query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeds)
        logits = vl_output.mean(dim=1)
        logits = logits.reshape(batch_size * n_match, 2)
        itm_targets = itm_targets.reshape(-1)
        loss_itm = F.cross_entropy(logits, itm_targets)  # [192, 2], [192]

        # LM Loss
        text_tokens = self._get_qformer_text_tokens(captions, image)
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=torch.cat([query_atts, text_tokens.attention_mask], dim=1),
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        # lm_loss = 0
        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def itm(
        self,
        samples,
    ):
        image = samples["image"]
        text_match = samples["text_match"]
        text_match_flat, (batch_size, n_match) = flatten_text(text_match)

        image_embeds, image_atts = self._encode_vision(image)  # [64, 514, 1408], [64, 514]
        image_embeds_itm = image_embeds.repeat_interleave(n_match, 0)
        image_atts_itm = image_atts.repeat_interleave(n_match, 0)
        
        query_tokens, query_atts = self._get_query_tokens(image_embeds_itm)  # [64, 32, 768], [64, 32]
        text_tokens = self._get_qformer_text_tokens(text_match_flat, image_embeds_itm)

        # encode image and text
        itm_output = self.Qformer.bert(
            text_tokens.input_ids, #[192, 32]
            query_embeds=query_tokens, #[192, 32, 768]
            attention_mask=torch.cat([query_atts, text_tokens.attention_mask], dim=1), #[192, 64]
            encoder_hidden_states=image_embeds_itm, #[192, 514, 1408]
            encoder_attention_mask=image_atts_itm, #[192, 514]
            return_dict=True,
        )

        # [192, 32, x] > [192, 32, 2] > [192, 2]
        vl_embeds = itm_output.last_hidden_state[:, : query_tokens.size(1), :]
        vl_output = self.itm_head(vl_embeds)
        logits = vl_output.mean(dim=1)
        logits = logits.reshape(batch_size, n_match, 2)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def itc(
        self,
        samples,
    ):
        image = samples["image"]
        text_class = samples["text_class"]
        text_class_flat, (batch_size, n_samples, n_class) = flatten_text(text_class)

        image_embeds, image_atts = self._encode_vision(image)  # [64, 514, 1408], [64, 514]
        query_tokens, query_atts = self._get_query_tokens(image)  # [64, 32, 768], [64, 32]
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True)
        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)

        # encode text
        text_tokens = self._get_qformer_text_tokens(text_class_flat, image)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True)
        text_feats = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        text_feats = text_feats.reshape(batch_size, n_samples, n_class, text_feats.shape[-1])

        # [64, 32, 256], [64, 50, 256] > [64, 50, 32] > [64, 50]
        sim = torch.einsum('ijk,ilmk->ilmj', image_feats, text_feats)
        sim, _ = sim.max(-1)
        sim = sim / self.temp
        return torch.softmax(sim, dim=-1)

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=True,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        image = samples["image"]

        num_beams = 1 # if use_nucleus_sampling else num_beams
        image_embeds, image_atts = self._encode_vision(image)
        image_embeds, image_atts = self._beam_repeat(num_beams, image_embeds, image_atts)
        query_tokens, query_atts = self._get_query_tokens(image_embeds)

        input_ids = torch.LongTensor(image_embeds.size(0), 1).fill_(self.tokenizer.bos_token_id).to(image.device)
        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text


    def forward_image(self, image):
        image_embeds, image_atts = self._encode_vision(image)
        query_tokens, _ = self._get_query_tokens(image)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True)
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True)
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_embeds, text_ids, text_atts):
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        query_tokens, query_atts = self._get_query_tokens(image)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=torch.cat([query_atts, text_atts], dim=1),
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        raise NotImplemented

    # ---------------------------------------------------------------------------- #
    #                                 Core Q-Former                                #
    # ---------------------------------------------------------------------------- #

    def _get_query_tokens(self, image):
        """Get the learned query tokens."""
        query_tokens = self.query_tokens.expand(image.size(0), -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=query_tokens.device)
        return query_tokens, query_atts
    
    def _get_qformer_text_tokens(self, prompt, image):
        """Tokenize Q-Former inputs."""
        text_output = self.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt"
        ).to(image.device)
        return text_output

    def _fold_time_into_batch(self, image):
        T = 1
        if image.dim() == 5:
            B, T, C, H, W = image.size()
            image = image.reshape(B*T, C, H, W)
        return image, T

    def _encode_vision(self, image):
        """Encode image/video to be used as Q-Former input."""
        # TODO possible to encode video here? if image.dim() == 5:
        with self.maybe_autocast():
            if image.dim() == 5:
                B, T, C, H, W = image.size()
                image = image.reshape(B*T, C, H, W)
                image_embeds = self.ln_vision(self.visual_encoder(image))
                _, L, C = image_embeds.size()
                image_embeds = image_embeds.reshape(B, T*L, C)
            else:
                image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device)
        return image_embeds, image_atts

    def _beam_repeat(self, num_beams, *xs):
        return [x.repeat_interleave(num_beams, dim=0) for x in xs] if num_beams > 1 else xs

    def _encode_qformer(self, image_embeds, image_atts, query_tokens, query_atts, text_tokens=None):
        """Encode image, query, and text tokens with the Q-Former."""
        # image_embeds: (batch, patch, features)

        # encode q former
        # if self.qformer_text_input and text_tokens is not None:
        #     # with text
        #     query_output = self.Qformer.bert(
        #         text_tokens.input_ids,
        #         attention_mask=torch.cat([query_atts, text_tokens.attention_mask], dim=1),
        #         query_embeds=query_tokens,
        #         encoder_hidden_states=image_embeds,
        #         encoder_attention_mask=image_atts,
        #         return_dict=True,
        #     )
        # else:
        # without text
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        return query_output

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
