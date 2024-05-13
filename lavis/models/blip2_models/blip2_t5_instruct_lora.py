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

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.blip2_models.blip2_t5_instruct2 import Blip2T5Instruct2
from transformers.modeling_outputs import BaseModelOutput


@registry.register_model("blip2_t5_instruct_lora")
class Blip2T5InstructLoRA(Blip2T5Instruct2):
    """
    BLIP2 T5 model.
    Supported model types:
        - flant5xl
        - flant5xxl
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5_instruct_lora", "flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2/blip2_instruct_flant5xl_lora.yaml",
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl_lora.yaml",
    }

    def __init__(self, *a, llm_lora=None, **kw):
        super().__init__(*a, **kw)
        self.t5_model = apply_peft(self.t5_model, **(llm_lora or {}))

    @classmethod
    def from_config(cls, cfg, **kw):
        return super().from_config(
            cfg, 
            llm_lora=cfg.get("llm_lora", {})
        )

TARGET_MODULES = {
    'attn': ['q','v'],
    'ffn': ["wi", "wo", "wi_1", "wi_0"],
    'all': ['q', 'v', "wi", "wo", "wi_1", "wi_0"],
}
def apply_peft(t5_model, llm_lora_apply='attn', llm_lora_r=8, llm_lora_alpha=8):
    if not llm_lora_apply:
        return t5_model

    print(f"applying llm lora on {llm_lora_apply}")
    if isinstance(llm_lora_apply, str):
        llm_lora_apply = TARGET_MODULES[llm_lora_apply]
    t5_model = get_peft_model(t5_model, LoraConfig(
        r=llm_lora_r,
        lora_alpha=llm_lora_alpha,
        target_modules=llm_lora_apply,
        task_type="SEQ_2_SEQ_LM",
    ))
    t5_model.print_trainable_parameters()
    return t5_model
