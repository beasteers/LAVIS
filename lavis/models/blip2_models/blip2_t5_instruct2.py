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

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


@registry.register_model("blip2_t5_instruct2")
class Blip2T5Instruct2(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - flant5xl
        - flant5xxl
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5_instructn", "flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2/blip2_instruct_flant5xl.yaml",
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl.yaml",
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
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
        qformer_text_input=True,
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
            num_query_token, self.visual_encoder.num_features
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        # ---------------------------------- T-5 LLM --------------------------------- #

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        # -------------------------------- Parameters -------------------------------- #

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

        self.qformer_text_input = qformer_text_input

    def forward(self, samples):
        image = samples["image"]
        prompt = samples["text_input"]

        # encode Q-former
        image_embeds, image_atts = self._encode_vision(image)
        query_tokens, query_atts = self._get_query_tokens(image)
        text_tokens = self._get_qformer_text_tokens(prompt, image)
        inputs_t5, atts_t5 = self._encode_qformer_t5(image_embeds, image_atts, query_tokens, query_atts, text_tokens)

        # encode few-shot
        fs_embeds, fs_atts = self.prepare_few_shot_embeds(self._sample_few_shot(samples.get('few_shot_samples')))
        fs_embeds, fs_atts = ([fs_embeds], [fs_atts]) if fs_embeds is not None else ([], [])

        with self.maybe_autocast(dtype=torch.bfloat16):
            # encode t5 query
            input_embeds, inputs_atts = self._get_t5_text_tokens(prompt, image, truncate=True)
            inputs_embeds = torch.cat([*fs_embeds, inputs_t5, input_embeds], dim=1)
            inputs_atts = torch.cat([*fs_atts, atts_t5, inputs_atts], dim=1)

            # encode targets
            targets, target_atts = self._get_t5_text_targets(samples['text_output'], image)

            # input_embeds = [few shot, images, text query]
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_atts,
                decoder_attention_mask=target_atts,
                return_dict=True,
                labels=targets,
            )
            lm_loss = outputs.loss

        # lm_loss = 0
        return {"loss": lm_loss}

    def _sample_few_shot(self, samples):
        if samples is None or not len(samples): 
            return []
        n = min(self.num_few_shot_examples, len(samples))
        this_n_fs = random.choices(list(range(n + 1)), [1 - self.few_shot_prob] + [self.few_shot_prob / n] * n)[0]
        return random.sample(samples, this_n_fs)

    def prepare_few_shot_embeds(self, samples, sample=False):
        if samples is None or not len(samples):
            return None, None

        images = torch.stack([d['image'] for d in samples], dim=0)
        text_input = [d['text_input'] for d in samples]

        image_embeds, image_atts = self._encode_vision(image)
        query_tokens, query_atts = self._get_query_tokens(images)
        text_tokens = self._get_qformer_text_tokens(text_input, images)
        inputs_t5, atts_t5 = self._encode_qformer_t5(image_embeds, image_atts, query_tokens, query_atts, text_tokens)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_embeds, inputs_atts = self._get_t5_text_tokens(text_input, image, truncate=True)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            inputs_atts = torch.cat([atts_t5, inputs_atts], dim=1)

        if this_n_fs > 1:
            inputs_embeds = inputs_embeds.reshape(inputs_embeds.size(0) // this_n_fs, inputs_embeds.size(1) * this_n_fs, inputs_embeds.size(2))
            inputs_atts = inputs_atts.reshape(inputs_atts.size(0) // this_n_fs, inputs_atts.size(1) * this_n_fs)

        return inputs_embeds, inputs_atts

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        prompt = samples.get('prompt', self.prompt)
        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        image_embeds, image_atts = self._encode_vision(image)
        query_tokens, query_atts = self._get_query_tokens(image)
        text_tokens = self._get_qformer_text_tokens(prompt, image)
        inputs_t5, atts_t5 = self._encode_qformer_t5(image_embeds, image_atts, query_tokens, query_atts, text_tokens)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds, inputs_atts = self._get_t5_text_tokens(prompt, image, truncate=True)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            inputs_atts = torch.cat([atts_t5, inputs_atts], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text

    def _format_question(self, samples, prompt=""):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                        for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = [
                        prompt.format(samples["text_input"][i], " ".join(
                            f"({string.ascii_lowercase[j]}) {ch}" 
                            for j, ch in enumerate(samples["choices"][i])))
                        for i in range(len(samples["text_input"]))]
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]
        samples["prompt"] = text_input
        return samples

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        samples = self._format_question(samples, prompt)
        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if self._apply_lemmatizer or samples.get("apply_lemmatizer", True):
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if isinstance(candidates[0], list):
            results = []
            for i in range(samples["image"].size(0)):
                sample_i = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                    **({k: [samples[k][i]] for k in ['text_input', 'context', 'history', 'caption'] if k in samples})
                }

                all_losses = self._predict_class(sample_i, candidates[i], n_segments)
                this_result = torch.argsort(all_losses, dim=-1)
                results.append(this_result)

            results = torch.cat(results, dim=0)
            return results

        all_losses = self._predict_class(samples, candidates, n_segments)
        this_result = torch.argsort(all_losses, dim=-1)
        return this_result

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - prompt: the instruction
            candidates:
                (list): A list of candidate class names;
            n_segments:
                (int): Split the candidates into n_segments and predict one by one. This is useful when the number of candidates is too large.
        Returns:
            output_class: predicted class index
        """

        image = samples["image"]
        # prompt = samples["prompt"]
        prompt = [samples['text_input'][i] for i in range(len(image))]
        bs = image.size(0)
        n_cands = len(candidates)

        # Encode Q-Former
        image_embeds, image_atts = self._encode_vision(image)
        query_tokens, query_atts = self._get_query_tokens(image)
        text_tokens = self._get_qformer_text_tokens(prompt, image)
        inputs_t5, atts_t5 = self._encode_qformer_t5(image_embeds, image_atts, query_tokens, query_atts, text_tokens)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_embeds, inputs_atts = self._get_t5_text_tokens(text_input, image)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            inputs_atts = torch.cat([atts_t5, inputs_atts], dim=1)

            output_tokens = self.t5_tokenizer(candidates, padding="longest", return_tensors="pt").to(image.device)
            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_atts,
            )

            all_losses = []
            for n in range(n_segments):
                # compute segment bounds
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)
                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                
                # compute the loss for part of an encoder output
                loss = self._decode_segment_loss(encoder_outputs, input_atts, output_tokens, start_i, end_i)
                all_losses.append(loss)

        all_losses = torch.cat(all_losses, dim=-1)
        return all_losses

    def _decode_segment_loss(self, encoder_outputs, input_atts, output_tokens, start_i, end_i):
        seg_len = end_i - start_i
        bs = encoder_outputs.last_hidden_state.shape[0]
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state.repeat_interleave(seg_len, dim=0))
        inputs_atts = inputs_atts.repeat_interleave(seg_len, dim=0)
        output_token_ids = output_tokens.input_ids[start_i:end_i].repeat(bs, 1)
        targets = output_token_ids.masked_fill(output_token_ids == self.t5_tokenizer.pad_token_id, -100)
        output_atts = output_tokens.attention_mask[start_i:end_i].repeat(bs, 1)

        output = self.t5_model(
            encoder_outputs=encoder_outputs,
            attention_mask=input_atts,
            decoder_attention_mask=output_atts,
            return_dict=True,
            labels=targets,
            reduction="none",
        )
        loss = outputs.loss.reshape(bs, seg_len)
        return loss
        

    # ---------------------------------------------------------------------------- #
    #                                 Core Q-Former                                #
    # ---------------------------------------------------------------------------- #

    def _fold_time_into_batch(self, image):
        T = 1
        if image.dim() == 5:
            B, T, C, H, W = image.size()
            image = image.reshape(B*T, C, H, W)
        return image, T

    def _get_query_tokens(self, image):
        """Get the learned query tokens."""
        query_tokens = self.query_tokens #.expand(image.size(0), -1, -1)
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

    def _get_t5_text_tokens(self, prompt, image, truncate=False):
        """Tokenize Q-Former inputs."""
        input_tokens = self.t5_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
            truncation=truncate,
            max_length=self.max_txt_len if truncate else None,
        ).to(image.device)
        input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        input_atts = input_tokens.attention_mask
        return input_embeds, input_atts

    def _get_t5_text_targets(self, text_output, image):
        """Tokenize Q-Former inputs."""
        output_tokens = self.t5_output_tokenizer(
            text_output,
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
            return_tensors="pt",
        ).to(image.device)
        targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100)
        target_atts = output_tokens.attention_mask
        return targets, target_atts

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
    
    def _encode_qformer(self, image_embeds, image_atts, query_tokens, query_atts, text_tokens=None, proj_t5=False):
        """Encode image, query, and text tokens with the Q-Former."""
        # add batch size
        query_tokens = query_tokens.expand(image_embeds.size(0), -1, -1)
        query_atts = query_atts.expand(image_embeds.size(0), -1)

        # encode q former
        if self.qformer_text_input and text_tokens is not None:
            # with text
            query_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=torch.cat([query_atts, text_tokens.attention_mask], dim=1),
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            # without text
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        if proj_t5:
            inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=inputs_t5.device)
            return inputs_t5, atts_t5, query_output
        return query_output

    def _proj_t5(self, query_output):
        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=inputs_t5.device)
        return inputs_t5, atts_t5

    def _encode_qformer_t5(self, image_embeds, image_atts, query_tokens, query_atts=None, text_tokens=None):
        """Encode frame(s) with the Q-Former and project them for the language model."""
        if image_embeds.dim() == 4 and not self.qformer_video_input:
            # We have image embeddings of shape (batch, time, patch, features)
            # encode each frame with Q-Former separately i.e. (batch, patch, features)
            inputs_t5, atts_t5, query_outputs = [], [], []
            for j in range(image_embeds.size(1)):
                frame_inputs_t5, frame_atts_t5, query_output = self._encode_qformer(image_embeds[:,j], image_atts[:,j], query_tokens, query_atts, text_tokens, proj_t5=True)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
        else:
            # We have image/video embeddings of shape (batch, time * patch, features)
            # encode all frames with Q-Former together i.e. (batch, time * patch, features)
            frame_inputs_t5, frame_atts_t5, query_output = self._encode_qformer(image_embeds, image_atts, query_tokens, query_atts, text_tokens, proj_t5=True)
        return frame_inputs_t5, frame_atts_t5

    def _lemmatize(self, answers):
        return [
            " ".join(
                token.lemma_ if token.pos_ in ["NOUN", "VERB"] else token.text 
                for token in self.lemmatizer(answer))
            for answer in answers
        ]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy
                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                import spacy.cli
                spacy.cli.download("en_core_web_sm")
                self._lemmatizer = spacy.load("en_core_web_sm")

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg, **kw):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        num_few_shot_examples = cfg.get("num_few_shot_examples", 0)
        few_shot_prob = cfg.get("few_shot_prob", 0.0)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_prob=few_shot_prob,
            qformer_text_input=qformer_text_input,
            **kw
        )

        model.load_checkpoint_from_config(cfg)

        return model
