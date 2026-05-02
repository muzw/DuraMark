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
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from duramark.tts.utils.mask import make_pad_mask


# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# Licensed under the Apache License, Version 2.0
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from duramark.tts.utils.mask import make_pad_mask

class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 duration_token_size: int = 512,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 using_detector: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.using_detector = using_detector
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        
        # 1. Text Embedding
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        # 2. Duration Embedding & Projection
        # Embedding 保持 input_size (512) 以获得更好的表征能力，随后投影到 output_size (80)
        self.duration_embedding = nn.Embedding(duration_token_size, input_size)
        self.duration_out_proj = nn.Linear(input_size, output_size)
        self.duration_scale = nn.Parameter(torch.tensor(0.1))

        # 3. Phase Projection
        # 将 0-1 的标量映射为向量
        self.phase_proj = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, input_size)
        )
        self.phase_out_proj = nn.Linear(input_size, output_size)
        self.phase_scale = nn.Parameter(torch.tensor(0.1))

        # 4. Other Components
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.length_regulator = length_regulator
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Fetch Data
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)
        
        # Condition Data
        duration_token = batch['regulate_duration_token'].to(device) # [B, T_feat]
        duration_phase = batch['duration_phase'].to(device)          # [B, T_feat]

        if self.using_detector:
            text_token = batch['text_token'].to(device)
            text_token_len = batch['text_token_len'].to(device)
            alignment = batch['alignment'].to(device)

        # X-Vector Projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # Create Masks
        # text_mask: 用于 Encoder，长度为 token_len
        text_mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        # feat_mask: 用于 Decoder 和 Duration/Phase，长度为 feat_len
        feat_mask = (~make_pad_mask(feat_len)).float().unsqueeze(-1).to(device)

        # 1. Text Encode
        token = self.input_embedding(torch.clamp(token, min=0)) * text_mask
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)

        p_emb = self.phase_proj(duration_phase.unsqueeze(-1).float())
        p_emb = self.phase_out_proj(p_emb)
        # 直接加到 h 上 (应用 text_mask 确保 padding 处为 0)
        h = h + (self.phase_scale * p_emb) * text_mask
        
        # 2. Length Regulator (Text Length -> Feat Length)
        # h 变为 [B, T_feat, 80]
        h, h_lengths = self.length_regulator(h, feat_len)

        d_emb = self.duration_embedding(duration_token)
        d_emb = self.duration_out_proj(d_emb) # shape: [B, T_text, D]

        # 准备插值: F.interpolate 需要输入格式为 [B, C, L]
        d_emb = d_emb.transpose(1, 2) # [B, D, T_text]
        # 执行最近邻插值 (Nearest Neighbor Interpolation)
        d_emb = F.interpolate(d_emb, size=h.shape[1], mode='nearest') # [B, D, T_feat]
        # 转回 [B, T, D] 格式
        d_emb = d_emb.transpose(1, 2) # [B, T_feat, D]

        # 加到 h 上 (应用 feat_mask 确保 padding 处为 0)
        # 注意：此时 h 已经是 [B, T_feat, D]
        h = h + (self.duration_scale * d_emb) * feat_mask

        # 4. Prepare Decoder Conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        # Feat Interpolation (Standard practice)
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)
        
        # Decoder Mask (B, 1, T)
        decoder_mask = feat_mask.transpose(1, 2)

        # 5. Compute Loss
        if self.using_detector:
            detector_input = (text_token, text_token_len, token_len, feat_len, alignment) # removed token_len here as per previous context usually
            flow_loss, duration_loss, acc = self.decoder.compute_loss(
                feat.transpose(1, 2).contiguous(),
                decoder_mask,
                h.transpose(1, 2).contiguous(),
                embedding,
                detector_input,
                cond=conds
            )
            loss = flow_loss + 4*duration_loss
            return {'loss': loss, "flow_loss": flow_loss, "duration_loss": duration_loss, "acc": acc}
        else:
            flow_loss = self.decoder.compute_loss(
                feat.transpose(1, 2).contiguous(),
                decoder_mask,
                h.transpose(1, 2).contiguous(),
                embedding,
                cond=conds
            )
            loss = flow_loss
            return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  regulate_duration_token,
                  duration_phase,
                  token_len,
                  prompt_token,
                  prompt_regulate_duration_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  flow_cache,
                  prompt_duration_phase,
                  ):
        assert token.shape[0] == 1
        
        # 0. Validate Inputs
        if duration_phase is None or prompt_duration_phase is None:
             raise ValueError("duration_phase and prompt_duration_phase must be provided for inference.")

        # 1. Prepare Embeddings
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # 2. Concat Text Tokens
        token_len1 = prompt_token.shape[1]
        token_len2 = token.shape[1]
        full_token = torch.concat([prompt_token, token], dim=1)
        full_token_len = prompt_token_len + token_len 

        # 3. Concat Duration & Phase (Frame Level)
        # 这些是已经展平到帧级的 Tokens (frame-level aligned)
        full_duration_token = torch.concat([prompt_regulate_duration_token, regulate_duration_token], dim=1)
        full_phase = torch.concat([prompt_duration_phase, duration_phase], dim=1)
        
        # 计算总帧数 (Total Mel Length)
        # total_mel_len = full_duration_token.shape[1]
        # mel_len1 = prompt_feat.shape[1]
        # mel_len2 = total_mel_len - mel_len1

        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        total_mel_len = mel_len1 +mel_len2

        # 4. Text Encoding
        mask = (~make_pad_mask(full_token_len)).unsqueeze(-1).to(embedding)
        full_token_emb = self.input_embedding(torch.clamp(full_token, min=0)) * mask
        
        h, h_lengths = self.encoder(full_token_emb, full_token_len)
        h = self.encoder_proj(h)

        p_emb = self.phase_proj(full_phase.unsqueeze(-1).float())
        p_emb = self.phase_out_proj(p_emb)
        # 直接加到 h 上 (应用 text_mask 确保 padding 处为 0)
        h = h + (self.phase_scale * p_emb) * mask

        # 5. Length Regulation (Critical Step)
        # 我们使用 inference 模式，并期望它能扩展到我们需要的长度
        # 注意：通常 LR.inference 是预测长度。但在这里，我们有目标长度 total_mel_len。
        # 如果你的 LR 支持强制长度，最好；如果不支持，我们假设它能大概预测准，然后强制对齐。
        h, h_lengths = self.length_regulator.inference(
            h[:, :token_len1], 
            h[:, token_len1:], 
            mel_len1, 
            mel_len2, 
            self.input_frame_rate
        )

        # Force Alignment: 强制 h 的长度等于 full_duration_token 的长度
        # 这一步是为了防止 LengthRegulator 预测的长度和我们在外部计算的 duration 长度有微小偏差
        if h.shape[1] != total_mel_len:
            h = h.transpose(1, 2) # [B, C, T]
            h = F.interpolate(h, size=total_mel_len, mode='linear')
            h = h.transpose(1, 2) # [B, T, C]

        # Mask for Decoder (B, 1, T)
        decoder_mask = (~make_pad_mask(torch.tensor([total_mel_len]))).to(h).unsqueeze(1)
        feat_mask = (~make_pad_mask(torch.tensor([total_mel_len]))).to(h).unsqueeze(-1)

        d_emb = self.duration_embedding(full_duration_token)
        d_emb = self.duration_out_proj(d_emb) # shape: [B, T_text, D]

        # 准备插值: F.interpolate 需要输入格式为 [B, C, L]
        d_emb = d_emb.transpose(1, 2) # [B, D, T_text]
        # 执行最近邻插值 (Nearest Neighbor Interpolation)
        d_emb = F.interpolate(d_emb, size=h.shape[1], mode='nearest') # [B, D, T_feat]
        # 转回 [B, T, D] 格式
        d_emb = d_emb.transpose(1, 2) # [B, T_feat, D]

        # 加到 h 上 (应用 feat_mask 确保 padding 处为 0)
        # 注意：此时 h 已经是 [B, T_feat, D]
        h = h + (self.duration_scale * d_emb) * feat_mask

        # 7. Decoder Inference
        conds = torch.zeros([1, total_mel_len, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)
        
        
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=decoder_mask,
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            flow_cache=flow_cache
        )
        
        # 只返回新生成的部分
        feat = feat[:, :, mel_len1:]
        
        # 最终校验
        assert feat.shape[2] == mel_len2
        
        return feat, flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  finalize):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        if finalize is False:
            h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat, None
