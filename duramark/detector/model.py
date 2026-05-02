# Copyright (c) 2024 Alibaba Inc (authors: Your Name)
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

from typing import Dict, Optional, Callable, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from duramark.tts.utils.common import IGNORE_ID
from duramark.tts.transformer.label_smoothing_loss import LabelSmoothingLoss
from duramark.tts.utils.common import th_accuracy
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import numpy as np
from duramark.detector.downsampler import DownsampleRegulator

class ResidualConv1dBlock(nn.Module):
    """带残差连接的单层膨胀卷积块"""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        # 为了保证 T 长度完全不变，padding 计算公式为：dilation * (kernel_size - 1) // 2
        # 前提是 kernel_size 必须为奇数
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, 
            stride=1, padding=padding, dilation=dilation
        )
        self.norm = nn.LayerNorm(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # x shape: (B, D, T)
        residual = x
        x = self.conv(x)
        
        # LayerNorm 需要在特征维度操作
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        
        # 残差连接，帮助网络更深时保持梯度稳定
        return x + residual

# class FeatureExtractor(nn.Module):
#     """
#     抗混响版前端：多尺度膨胀残差卷积 (Dilated Residual Convolutions)
#     """
#     def __init__(self, input_dim, output_dim, hidden_dim=256, kernel_size=3, num_layers=4):
#         super().__init__()
        
#         # 1. 初始投影：将输入特征维度映射到隐藏层维度
#         self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
#         # 2. 核心层：膨胀残差卷积堆叠
#         self.conv_blocks = nn.ModuleList()
#         # 默认 4 层，dilation 分别为 1, 2, 4, 8
#         # 感受野将扩大为：1 + (3-1)*(1+2+4+8) = 31 帧
#         for i in range(num_layers):
#             dilation = 2 ** i 
#             self.conv_blocks.append(
#                 ResidualConv1dBlock(hidden_dim, kernel_size, dilation)
#             )
            
#         # 3. 输出投影：映射到 Encoder 所需的最终维度
#         self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        
#     def forward(self, x):
#         # x: (B, T, input_dim) -> 转置为 (B, input_dim, T) 供 Conv1d 处理
#         x = x.transpose(1, 2) 
        
#         # 初始投影
#         x = self.input_proj(x)
        
#         # 穿过所有膨胀残差层
#         for block in self.conv_blocks:
#             x = block(x)
            
#         # 输出投影
#         x = self.output_proj(x)
        
#         # 还原维度 (B, T, output_dim) 传给后续的 Transformer Encoder
#         x = x.transpose(1, 2)
#         return x

class FeatureExtractor(nn.Module):
    """
    加强版前端：2层 Conv1d (Stride=1, No Subsampling)
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, kernel_size=3):
        super().__init__()
        # 保持 padding = kernel_size // 2，确保由始至终由 T 长度不变
        
        # 第一层：Input -> Hidden
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, stride=1, padding=kernel_size//2)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.ReLU()
        
        # 第二层：Hidden -> Output (Projection to Encoder Dimension)
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size, stride=1, padding=kernel_size//2)
        self.norm2 = nn.LayerNorm(output_dim)
        self.act2 = nn.ReLU() # 或者最后一层不加 ReLU，直接进 Encoder，取决于 Encoder 这里的偏好，通常加了没事
        
    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2) # -> (B, D, T)
        
        # Layer 1
        x = self.conv1(x)
        x = x.transpose(1, 2) # Norm 需要在最后一维计算
        x = self.norm1(x)
        x = self.act1(x)
        x = x.transpose(1, 2) # 转回去准备下一层卷积
        
        # Layer 2
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.act2(x)
        
        # Output: (B, T, D_out)
        return x

        
class DurationDetector(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            speech_encoder_input_size: int,
            speech_feat_dim: int,
            input_size: int,
            text_token_size: int,
            max_text_length: int,
            text_encoder: torch.nn.Module,
            speech_encoder: torch.nn.Module,
            aligner: torch.nn.Module,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
    ):
        super().__init__()
        self.text_token_size = text_token_size
        self.input_size = input_size
        self.max_text_length = max_text_length

        self.aligner = aligner
        
        # 1. Text embedding module
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder

        self.speech_frontend = FeatureExtractor(
            input_dim=speech_feat_dim,          # e.g., 80
            output_dim=speech_encoder_input_size, # e.g., 256 (Conformer 的输入维度)
            hidden_dim=256,                     # 中间层的维度，通常和输出维度保持一致
            kernel_size=3                       # 3 或 5 都可以
        )
        
        self.speech_encoder = speech_encoder
        
        # 3. Text feature projection
        self.text_feat_projection = nn.Linear(self.text_encoder.output_size(), input_size)
        self.speech_feat_projection = nn.Linear(self.speech_encoder.output_size(), input_size)

        self.downsampler = DownsampleRegulator(input_size,input_size)
        
        # 7. Loss functions
        self.text_criterion = LabelSmoothingLoss(
            size=max_text_length,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def encode(
            self,
            feats: torch.Tensor,
            lengths: torch.Tensor,
            encoder: torch.nn.Module,
    ):
        encoder_out, encoder_mask = encoder(feats, lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        return encoder_out, encoder_out_lens

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            batch: dict containing:
                - text_token: (B, T_text)
                - text_token_len: (B,)
                - speech_feat: (B, T_speech, D_speech)
                - speech_feat_len: (B,)
                - alignment: (B, T_speech) - alignment from each frame to text token index
                
        Returns:
            Dictionary containing losses and accuracy
        """
        # Extract inputs
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_feat = batch['speech_feat'].to(device)
        speech_feat_len = batch['speech_feat_len'].to(device)
        alignment = batch['alignment'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)


        target = [torch.tensor([IGNORE_ID] * (text_token_len[i]) + alignment[i, :speech_token_len[i]].tolist()
                                ) for i in range(text_token.size(0))]

        target = pad_sequence(target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. Text embedding
        text_emb = self.text_embedding(text_token)  # (B, T_text, D_text_emb)
        text_emd, text_emd_len = self.encode(text_emb, text_token_len, self.text_encoder)
        text_emd = self.text_feat_projection(text_emd)

        # 2. Speech embedding
        speech_emd = self.speech_frontend(speech_feat)
        speech_emd, speech_emd_len = self.encode(speech_emd, speech_feat_len, self.speech_encoder)
        speech_emd = self.speech_feat_projection(speech_emd)

        speech_emd, speech_emd_len = self.downsampler(speech_emd, speech_emd_len, speech_token_len)

        # 4. Pass through transformer encoder
        text_logits = self.aligner.forward_logits(text_emd, text_emd_len, speech_emd, speech_emd_len)

        # 8. Compute losses
        loss = self.text_criterion(
            text_logits, 
            target
        )
        
        # 9. Compute accuracy
        accuracy = th_accuracy(text_logits.view(-1, self.max_text_length), target, ignore_label=IGNORE_ID
        )
        return {'loss': loss, 'accuracy': accuracy}


    def forward_by_flow(
            self,
            text_token,
            text_token_len,
            speech_feat,
            speech_feat_len,
            speech_token_len,
            alignment,
            device: torch.device,
    ):
        """
        Args:
            batch: dict containing:
                - text_token: (B, T_text)
                - text_token_len: (B,)
                - speech_feat: (B, T_speech, D_speech)
                - speech_feat_len: (B,)
                - alignment: (B, T_speech) - alignment from each frame to text token index
                
        Returns:
            Dictionary containing losses and accuracy
        """

        target = [torch.tensor([IGNORE_ID] * (text_token_len[i]) + alignment[i, :speech_token_len[i]].tolist()
                                ) for i in range(text_token.size(0))]

        target = pad_sequence(target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. Text embedding
        text_emb = self.text_embedding(text_token)  # (B, T_text, D_text_emb)
        text_emd, text_emd_len = self.encode(text_emb, text_token_len, self.text_encoder)
        text_emd = self.text_feat_projection(text_emd)

        # 2. Speech embedding
        speech_emd = self.speech_frontend(speech_feat)
        speech_emd, speech_emd_len = self.encode(speech_emd, speech_feat_len, self.speech_encoder)
        speech_emd = self.speech_feat_projection(speech_emd)

        speech_emd, speech_emd_len = self.downsampler(speech_emd, speech_emd_len, speech_token_len)
        
        # 4. Pass through transformer encoder
        text_logits = self.aligner.forward_logits(text_emd, text_emd_len, speech_emd, speech_emd_len)

        # 8. Compute losses
        loss = self.text_criterion(
            text_logits, 
            target
        )
        
        # 9. Compute accuracy
        accuracy = th_accuracy(text_logits.view(-1, self.max_text_length), target, ignore_label=IGNORE_ID
        )
        return loss, accuracy


    def infer(
            self,
            text_token,
            text_token_len,
            speech_feat,
            speech_feat_len,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            batch: dict containing:
                - text_token: (B, T_text)
                - text_token_len: (B,)
                - speech_feat: (B, T_speech, D_speech)
                - speech_feat_len: (B,)
                - alignment: (B, T_speech) - alignment from each frame to text token index
                
        Returns:
            Dictionary containing losses and accuracy
        """
        # Extract inputs
        text_token = text_token.to(device)
        text_token_len = text_token_len.to(device)
        speech_feat = speech_feat.to(device)
        speech_feat_len = speech_feat_len.to(device)
        speech_token_len = torch.round(speech_feat_len*256*50/22050).to(device).int()

        # 1. Text embedding
        text_emb = self.text_embedding(text_token)  # (B, T_text, D_text_emb)
        text_emd, text_emd_len = self.encode(text_emb, text_token_len, self.text_encoder)
        text_emd = self.text_feat_projection(text_emd)

        # 2. Speech embedding
        speech_emd = self.speech_frontend(speech_feat)
        speech_emd, speech_emd_len = self.encode(speech_emd, speech_feat_len, self.speech_encoder)
        speech_emd = self.speech_feat_projection(speech_emd)

        speech_emd, speech_emd_len = self.downsampler(speech_emd, speech_emd_len, speech_token_len)
        
        # 4. Pass through transformer encoder
        text_logits = self.aligner.forward_logits(text_emd, text_emd_len, speech_emd, speech_emd_len)

        text_logits = text_logits[0,text_token_len.item():,:text_token_len.item()]
        text_logits = F.softmax(text_logits, dim=-1)
        # 形状为 (speech_length, text_length) 每行是当前帧在text上的概率分布，每行和为1

        alignment_path, path_score, durations = self.aligner.forward_alignment(text_logits)
        # 假设 aligment_map 已经是概率 (Batch=1 的情况)
        soft_durations = self.aligner.extract_soft_duration(text_logits)
        return durations, soft_durations


