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

from typing import Dict, Optional, Tuple, List
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from duramark.tts.utils.common import IGNORE_ID
from numba import jit # 引入 numba

# 定义独立的 Numba 加速函数 (放在类外面)
@jit(nopython=True)
def viterbi_numba(log_probs):
    speech_len, text_len = log_probs.shape
    dp = np.full((speech_len, text_len), -np.inf)
    # 记录路径: 0=stay, 1=move
    backpointers = np.zeros((speech_len, text_len), dtype=np.int8)
    
    dp[0, 0] = log_probs[0, 0]
    
    for i in range(1, speech_len):
        for j in range(text_len):
            # Stay (来自 i-1, j)
            score_stay = dp[i-1, j]
            
            # Move (来自 i-1, j-1)
            score_move = -np.inf
            if j > 0:
                score_move = dp[i-1, j-1]
            
            # 比较
            if score_stay >= score_move:
                dp[i, j] = score_stay + log_probs[i, j]
                backpointers[i, j] = 0
            else:
                dp[i, j] = score_move + log_probs[i, j]
                backpointers[i, j] = 1
                
    return dp, backpointers

class Aligner(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            max_text_length: int,
            dropout_rate: float,
            encoder: torch.nn.Module,
    ):
        super().__init__()
        self.input_size = input_size

        self.text_classifier = nn.Linear(encoder.output_size(), max_text_length)

        # [新增] 模态编码 (Segment Embeddings)
        # 0: Text, 1: Speech
        # 这是一个可学习的向量，告诉模型哪部分是文本，哪部分是语音
        self.segment_embedding = nn.Embedding(2, input_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(input_size, dropout=dropout_rate)
        
        self.encoder = encoder

    def pad_unpad_sequence(self, text_emd, text_emd_len, speech_emd, speech_emd_len):
        text_emd = unpad_sequence(text_emd, text_emd_len.cpu(), batch_first=True)
        speech_emd = unpad_sequence(speech_emd, speech_emd_len.cpu(), batch_first=True)
        inputs = [torch.concat([text_emd[i], speech_emd[i]], dim=0)
                    for i in range(len(text_emd))]
        lengths = torch.tensor([i.size(0) for i in inputs], dtype=torch.int32, device=text_emd[0].device)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=IGNORE_ID)
        return inputs, lengths

    def forward_logits(
            self,
            text_emd: torch.Tensor,
            text_emd_len: torch.Tensor,
            speech_emd: torch.Tensor,
            speech_emd_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            inputs: (B, T_total, D) combined text + speech features
            input_lengths: (B,) lengths of combined sequences
            
        Returns:
            outputs: (B, T_total, hidden_size) transformer outputs
            output_lengths: (B,) output lengths (same as input_lengths)
        """

        # 1. 分别添加 Positional Encoding (在 unpad 状态下或者 pad 状态下都可以，这里假设输入已经是 pad 好的 tensor)
        # 注意：这里的 text_emd 和 speech_emd 应该已经是 (B, T, D) 形状
        # 如果前面没有 unpad/pad 的操作干扰，直接加比较方便
        
        # 给 Text 加位置编码 (0, 1, ..., T_text)
        text_emd = self.pos_encoding(text_emd)
        
        # 给 Speech 加位置编码 (0, 1, ..., T_speech)
        # 关键点：Speech 也是从 0 开始，代表它自己的时间轴
        speech_emd = self.pos_encoding(speech_emd)
        
        # 2. 添加 Segment Embedding (模态区分)
        # Text 加上 type=0 的向量
        text_seg = self.segment_embedding(torch.zeros_like(text_emd[:, :, 0], dtype=torch.long))
        text_emd = text_emd + text_seg
        
        # Speech 加上 type=1 的向量
        speech_seg = self.segment_embedding(torch.ones_like(speech_emd[:, :, 0], dtype=torch.long))
        speech_emd = speech_emd + speech_seg

        # 3. 拼接 (Concatenation)
        # 原有的逻辑：unpad -> concat -> pad
        # 由于我们已经加了 PE 和 Segment Emb，现在可以放心地拼接了

        # 3. unpad and pad
        inputs, inputs_len = self.pad_unpad_sequence(text_emd, text_emd_len, speech_emd, speech_emd_len)

        encoder_out, encoder_mask = self.encoder(inputs, inputs_len, decoding_chunk_size=1, num_decoding_left_chunks=-1)

        text_logits = self.text_classifier(encoder_out)
        
        return text_logits


    def extract_duration_from_alignment(self, alignment_path, text_length):
        """
        从对齐路径中提取每个文本位置的持续时间
        
        参数:
            alignment_path: 对齐路径，每个元素为 (frame_index, text_index)
            text_length: 文本长度
        
        返回:
            durations: list, 每个文本位置对应的语音帧数
        """
        durations = [0] * text_length
        
        for _, text_idx in alignment_path:
            durations[text_idx] += 1
        
        return durations

    def extract_soft_duration(self, aligment_map: torch.Tensor):
        """
        计算软时长 (Expected Duration)，用于软投票水印检测。
        
        参数:
            aligment_map: (T_speech, N_text) 的概率矩阵 (0~1之间)
                          如果是 Logits，请先做 Softmax。
        返回:
            soft_durations: List[float], 浮点数时长列表
        """
        # 确保转为 numpy 或 tensor 处理
        if isinstance(aligment_map, np.ndarray):
            aligment_map = torch.from_numpy(aligment_map)
            
        # 核心逻辑：直接对每一列求和
        # 含义：第 j 个 token 在所有时间步上出现的概率总和 = 它的期望帧数
        # 形状变化: (T, N) -> (N,)
        soft_durations = aligment_map.sum(dim=0)
        
        return soft_durations.tolist()

    def forward_alignment(self, aligment_map: torch.Tensor):
        if isinstance(aligment_map, torch.Tensor):
            aligment_map = aligment_map.cpu().detach().numpy()
        
        speech_length, text_length = aligment_map.shape
        log_probs = np.log(aligment_map + 1e-12)
        
        # === 调用 Numba 加速函数 ===
        dp, backpointers = viterbi_numba(log_probs)
        # =========================
        
        # 后续的回溯逻辑和方案一类似，只是要把 backpointers 的逻辑稍作调整
        # (Numba 很难返回复杂的 tuple list，所以在外面做回溯比较好)
        
        if dp[speech_length-1, text_length-1] == -np.inf:
            end_j = np.argmax(dp[speech_length-1, :])
        else:
            end_j = text_length - 1

        alignment_path = []
        curr_j = end_j
        
        for i in range(speech_length - 1, -1, -1):
            alignment_path.append((i, curr_j))
            if backpointers[i, curr_j] == 1:
                curr_j -= 1
        
        alignment_path.reverse()
        
        alignment_path.reverse()
        
        # 防止数值下溢
        try:
            path_score = np.exp(dp[speech_length-1, end_j])
        except:
            path_score = 0.0

        # --- 2. 提取高精度 Duration ---
        durations = self.extract_duration_from_alignment(alignment_path, text_length)
        
        return alignment_path, path_score, durations


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)