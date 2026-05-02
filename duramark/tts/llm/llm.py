# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from duramark.tts.utils.common import IGNORE_ID
from duramark.tts.transformer.label_smoothing_loss import LabelSmoothingLoss
from duramark.tts.utils.common import th_accuracy,th_accuracy_topn,th_accuracy_neighbor
import numpy as np



import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            duration_token_size: int,
            duration_embedding_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            speech_sampling: Callable,
            duration_sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        self.duration_token_size = duration_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder

        self.sos_eos = 0
        self.llm_embedding = torch.nn.Embedding(1, llm_input_size)
        self.duration_task_id = 0
        self.speech_task_id = 1
        self.task_embedding_size = 256
        self.task_embedding = torch.nn.Embedding(2, self.task_embedding_size)
        self.llm = llm

        self.duration_query = nn.Linear(self.task_embedding_size, llm_input_size)


        self.duration_embedding = torch.nn.Embedding(duration_token_size, duration_embedding_size)
        self.duration_task_combine = nn.Linear(duration_embedding_size + self.task_embedding_size, llm_input_size)

        self.text_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )
        # 2. build speech token language model related modules
        
        self.speech_criterion_ce = LabelSmoothingLoss(
            size=speech_token_size,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.duration_criterion_ce = LabelSmoothingLoss(
            size=self.duration_token_size,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules

        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.speech_sampling = speech_sampling
        self.duration_sampling = duration_sampling

        # 分类器
        self.speech_classifier = nn.Linear(llm_output_size, speech_token_size)
        self.duration_classifier = nn.Linear(llm_output_size, duration_token_size)

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len
    
    def build_mixed_embedding(self, text_speech_duration_token, text_speech_duration_mask, text_emb, text_token_len):
        B, Seq = text_speech_duration_token.shape
        device = text_speech_duration_token.device
        mixed_emb = torch.zeros(B, Seq, self.llm_input_size, device=device)

        duration_pos = (text_speech_duration_mask == 1)
        
        # 收集duration位置的LM输出
        duration_token = text_speech_duration_token[duration_pos]
        duration_emb = self.duration_embedding(duration_token)

        speech_task_id_emb = self.task_embedding.weight[self.speech_task_id].reshape(1, -1).repeat(duration_emb.size(0),1)

        combined = torch.cat([duration_emb, speech_task_id_emb], dim=1)
        mixed_emb[duration_pos] = self.duration_task_combine(combined).to(torch.float32)

        speech_pos = (text_speech_duration_mask == 2)
        speech_emb = self.speech_embedding(text_speech_duration_token[speech_pos])
        mixed_emb[speech_pos] = speech_emb

        text_pos = (text_speech_duration_mask == 0)
        duration_task_id_emb = self.task_embedding.weight[self.duration_task_id].reshape(1, -1).repeat(duration_emb.size(0),1)

        # mixed_emb[text_pos] = self.text_task_combine(combined).to(torch.float32)
        mixed_emb[text_pos] = self.duration_query(duration_task_id_emb).to(torch.float32)
        
        return mixed_emb

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        text_speech_duration_token = batch['text_speech_duration_token'].to(device)
        text_speech_duration_token_len = batch['text_speech_duration_token_len'].to(device)
        text_speech_duration_mask = batch['text_speech_duration_mask'].to(device)
        embedding = batch['embedding'].to(device)

        speech_token_target = torch.where(text_speech_duration_mask==2,text_speech_duration_token,IGNORE_ID)
        
        # speech_token_target = torch.where(text_speech_duration_mask==0,self.speech_token_size,speech_token_target)
        # speech_token_target = torch.where(text_speech_duration_mask==0,IGNORE_ID,speech_token_target)
        duration_token_target = torch.where(text_speech_duration_mask==1,text_speech_duration_token,IGNORE_ID)
        # 1. prepare llm_target
        speech_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token_target[i, 1:text_speech_duration_token_len[i]].tolist() +
                                  [IGNORE_ID]) for i in range(text_token.size(0))]
        speech_target = pad_sequence(speech_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        duration_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + duration_token_target[i, 1:text_speech_duration_token_len[i]].tolist() +
                                  [IGNORE_ID]) for i in range(text_token.size(0))]
        duration_target = pad_sequence(duration_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        text_token = self.text_affine_layer(text_token)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)

        # 4. encode speech_token
        text_speech_duration_token = self.build_mixed_embedding(text_speech_duration_token,text_speech_duration_mask,text_token,text_token_len)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         text_speech_duration_token, text_speech_duration_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        
        # 分割Logits时使用新生成的duration_logits
        duration_logits = self.duration_classifier(lm_output)
        speech_logits = self.speech_classifier(lm_output)

        speech_loss = self.speech_criterion_ce(speech_logits, speech_target)
        duration_loss = self.duration_criterion_ce(duration_logits, duration_target)
        loss =  speech_loss + 0.5*duration_loss


        speech_acc = th_accuracy(speech_logits.view(-1, self.speech_token_size), speech_target, ignore_label=IGNORE_ID)
        duration_acc = th_accuracy(duration_logits.view(-1, self.duration_token_size), duration_target, ignore_label=IGNORE_ID)

        return {'loss': loss,'speech_loss': speech_loss, 'speech_acc': speech_acc, 'duration_loss': duration_loss, 'duration_acc': duration_acc}

    def speech_sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        top_ids = self.speech_sampling(weighted_scores, decoded_tokens, sampling)
        return top_ids

    def duration_sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
    ):
        top_ids = self.duration_sampling(weighted_scores, decoded_tokens, sampling)
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_text_speech_duration_token: torch.Tensor,
            prompt_text_speech_duration_mask: torch.Tensor,
            prompt_text_speech_duration_token_len: torch.Tensor,
            embedding: torch.Tensor,
            out_path,
            watermark_bits: None,  # 接收水印比特列表
            speech_sampling: int = 25,
            duration_sampling: int = 25,
            max_token_text_ratio: float = 30,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)
        # 1. encode text
        text, text_len = self.encode(text, text_len)
        text = self.text_affine_layer(text)

        prompt_text_emd = text[:,:prompt_text_len]
        syn_text_emd = text[:,prompt_text_len:]

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        if prompt_text_speech_duration_token_len != 0:
            prompt_speech_duration_token_emb = self.build_mixed_embedding(prompt_text_speech_duration_token, prompt_text_speech_duration_mask, prompt_text_emd, prompt_text_len)
            # 提取prompt的最后一个duration token
            duration_pos = (prompt_text_speech_duration_mask == 1)
            # 收集duration位置的LM输出
            current_duration_token = prompt_text_speech_duration_token[duration_pos][-1].item()
        else:
            prompt_speech_duration_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        duration_task_id_emb = self.task_embedding.weight[self.duration_task_id].reshape(1, -1)
        speech_task_id_emb = self.task_embedding.weight[self.speech_task_id].reshape(1, -1)

        lm_input = torch.concat([sos_eos_emb, embedding, text, prompt_speech_duration_token_emb, self.duration_query(duration_task_id_emb).to(torch.float32).reshape(1, 1, -1)], dim=1)

        # 5. step by step decode
        speech_out_tokens = []
        duration_out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        # 循环遍历每一个合成文本token
        for i in range(syn_text_emd.size(1)):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            offset += lm_input.size(1)
            # 获取 Duration 的 Logits
            duration_logits_logp = self.duration_classifier(y_pred[:, -1]).log_softmax(dim=-1)
            
            # ================= [修改开始] 循环水印逻辑 =================
            if watermark_bits is not None and len(watermark_bits) > 0:
                # 使用取模运算实现循环读取：0, 1, ..., 7, 0, 1, ...
                bit_index = i % len(watermark_bits)
                target_bit = int(watermark_bits[bit_index])
                
                # 规则：1 -> 奇数帧, 0 -> 偶数帧
                if target_bit == 1:
                    # 目标：奇数 (1, 3, 5...)，屏蔽偶数索引
                    duration_logits_logp[:, 0::2] = -float('inf')
                elif target_bit == 0:
                    # 目标：偶数 (0, 2, 4...)，屏蔽奇数索引
                    duration_logits_logp[:, 1::2] = -float('inf')
            # ================= [修改结束] =================
            # duration_logits_logp[:, 0:5] = -float('inf')
            duration_top_ids = self.duration_sampling_ids(duration_logits_logp.squeeze(dim=0), duration_out_tokens, duration_sampling).item()
            duration_out_tokens.append(duration_top_ids)
            duration_emb = self.duration_embedding.weight[duration_top_ids].reshape(1, -1)
            # print(duration_top_ids, duration_logits_logp[:, 0:11])
            current_duration_token = duration_top_ids
            combined = torch.cat([duration_emb, speech_task_id_emb], dim=1)
            lm_input = self.duration_task_combine(combined).to(torch.float32).reshape(1, 1, -1)

            for j in range(duration_top_ids):
                y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
                offset += lm_input.size(1)
                speech_logits_logp = self.speech_classifier(y_pred[:, -1]).log_softmax(dim=-1)
                speech_top_ids = self.speech_sampling_ids(speech_logits_logp.squeeze(dim=0), speech_out_tokens, speech_sampling).item()
                speech_out_tokens.append(speech_top_ids)
                yield speech_top_ids, current_duration_token
                lm_input = self.speech_embedding.weight[speech_top_ids].reshape(1, 1, -1)
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            offset += lm_input.size(1)
            lm_input = self.duration_query(duration_task_id_emb).to(torch.float32).reshape(1, 1, -1)

            

            


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class Qwen2LM(torch.nn.Module):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
