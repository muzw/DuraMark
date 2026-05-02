# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
import json
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import inflect
try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    try:
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        from tn.english.normalizer import Normalizer as EnNormalizer
    except ImportError:
        ZhNormalizer = None
        EnNormalizer = None
    use_ttsfrd = False
from duramark.tts.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph

import torch.nn as nn

from joblib import load

# Lazy-loaded tokenizer (avoids network call at import time)
_frontend_tokenizer = None


def _get_frontend_tokenizer():
    global _frontend_tokenizer
    if _frontend_tokenizer is None:
        from transformers import BertTokenizer
        _frontend_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    return _frontend_tokenizer

class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 instruct: bool = False,
                 allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option,
                                                                     providers=["CUDAExecutionProvider" if torch.cuda.is_available() else
                                                                                "CPUExecutionProvider"])
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        else:
            self.spk2info = {}
        self.instruct = instruct
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
        else:
            if ZhNormalizer is not None:
                self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False)
                self.en_tn_model = EnNormalizer()
            else:
                self.zh_tn_model = None
                self.en_tn_model = None

        self.frame_rate = 50

    def _extract_text_token(self, text):
        # text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = _get_frontend_tokenizer().encode(text, add_special_tokens=False)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len

    def _extract_speech_token(self, speech):
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None,
                                                         {self.speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          self.speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        if text_frontend is False:
            return [text] if split is True else text
        text = text.strip()
        if contains_chinese(text):
            if self.use_ttsfrd:
                texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
                text = ''.join(texts)
            elif self.zh_tn_model is not None:
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
            else:
                # No text normalization available, return raw text as single segment
                texts = [text]
        else:
            if self.use_ttsfrd:
                texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
                text = ''.join(texts)
            elif self.en_tn_model is not None:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
            else:
                # No text normalization available, return raw text as single segment
                texts = [text]
        if split is False:
            return text
        return texts
    
    
    def gen_text_speech_duration_token(self, speech_tokens, speech_feat, text_token, intervals):
        speech_length = speech_tokens.size(1)
        # 生成duration和speech token的混合序列
        mixed_tokens = []
        mixed_mask = []

        durations = []
        timestamp = []

        # 添加一个标志来检查是否有duration超过512
        has_long_duration = False
        
        # 计算每个文本对应的duration（当前字开始时间到下一个字开始时间之差）
        for i in range(len(intervals)):
            current_start = intervals[i][0]

            if i == 0:
                current_start = 0
            
            # 计算下一个字的开始时间
            if i < len(intervals) - 1:
                next_start = intervals[i+1][0]
            else:
                # 最后一个字，使用语音总长度作为结束
                next_start = speech_length / self.frame_rate
            
            # 转换为帧数
            start_frame = int(current_start * self.frame_rate)
            end_frame = int(next_start * self.frame_rate)
            
            # 确保帧索引在有效范围内
            start_frame = np.clip(start_frame, 0, speech_length-1)
            end_frame = np.clip(end_frame, 0, speech_length)
            
            # 计算duration（帧数）
            duration = end_frame - start_frame
            if duration <= 0:
                # 如果duration为0或负值，设为最小1帧
                duration = 1
                end_frame = start_frame + 1

            # 检查duration是否超过512帧
            if duration > 256:
                has_long_duration = True
                break  # 如果有一个超过，就跳出循环

            durations.append(duration)
            timestamp.append(start_frame)
            
            # 添加duration和对应的speech tokens
            mixed_tokens.extend([duration,duration])  # duration作为数值
            mixed_mask.extend([0,1])  # duration标记为1
            
            # 添加该duration对应的speech tokens
            speech_segment = speech_tokens[0][start_frame:end_frame]
            mixed_tokens.extend(speech_segment)
            mixed_mask.extend([2] * len(speech_segment))  # speech token标记为2


        
        # 如果有duration超过512帧，跳过这个sample
        if has_long_duration:
            print( "error reference")

        # 生成flow的86hz的duration
        feat_frame_rate = 50#22050/256
        regulate_duration = []
        durations = []
        timestamp = []
        duration_phase = []
        feat_length = speech_tokens.size(1)
        # 计算每个文本对应的duration（当前字开始时间到下一个字开始时间之差）
        for i in range(len(intervals)):
            current_start = intervals[i][0]
            if i == 0:
                current_start = 0
            
            # 计算下一个字的开始时间
            if i < len(intervals) - 1:
                next_start = intervals[i+1][0]
            else:
                # 最后一个字，使用语音总长度作为结束
                next_start = feat_length / feat_frame_rate
            
            # 转换为帧数
            start_frame = int(current_start * feat_frame_rate)
            end_frame = int(next_start * feat_frame_rate)
            
            # 确保帧索引在有效范围内
            start_frame = np.clip(start_frame, 0, feat_length-1)
            end_frame = np.clip(end_frame, 0, feat_length)
            
            # 计算duration（帧数）
            duration = end_frame - start_frame
            if duration <= 0:
                # 如果duration为0或负值，设为最小1帧
                duration = 1
                end_frame = start_frame + 1

            durations.append(duration)
            timestamp.append(start_frame)

        duration_index = -1
        current_step = 0     # 新增：用于记录当前字念到了第几帧

        for frame_idx in range(feat_length):
            if frame_idx in timestamp:
                duration_index += 1
                current_step = 0 # 换新字了，计数器归零
            # 计数器 +1 (从1开始计数，方便计算 1.0)
            current_step += 1
            cur_dur = durations[duration_index]

            # 计算 Phase: 当前步数 / 总时长
            # 例如时长为2: 第一帧 1/2=0.5, 第二帧 2/2=1.0
            phase = current_step / cur_dur

            regulate_duration.append(durations[duration_index])
            duration_phase.append(phase) # 新增
                    
        text_speech_duration_token = mixed_tokens
        text_speech_duration_mask = mixed_mask

        return text_speech_duration_token, text_speech_duration_mask, regulate_duration, duration_phase


    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input


    def frontend_zero_shot(self, tts_text, prompt_text, ref_word_interval, prompt_speech_16k, resample_rate, watermark_bits=None, out_path=None):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        
        # 这里的 resample_rate 通常是 22050
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        
        # 提取 feat 和 len (这是 High Frame Rate 的基准)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        
        # 提取 50Hz codec token (用于 LLM)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)

        
        prompt_text_speech_duration_token, prompt_text_speech_duration_mask, prompt_regulate_duration, prompt_duration_phase = \
            self.gen_text_speech_duration_token(speech_token, speech_feat, prompt_text_token, ref_word_interval)

        # 转换为 Tensor 并增加 batch 维度
        prompt_text_speech_duration_token = torch.tensor(prompt_text_speech_duration_token).unsqueeze(0)
        prompt_text_speech_duration_mask = torch.tensor(prompt_text_speech_duration_mask).unsqueeze(0)
        prompt_regulate_duration = torch.tensor(prompt_regulate_duration).unsqueeze(0)
        prompt_duration_phase = torch.tensor(prompt_duration_phase).unsqueeze(0) # 新增

        embedding = self._extract_spk_embedding(prompt_speech_16k)
        
        model_input = {
            'text': tts_text_token, 
            'text_len': tts_text_token_len,
            'prompt_text': prompt_text_token, 
            'prompt_text_len': prompt_text_token_len,
            'llm_prompt_text_speech_duration_token': prompt_text_speech_duration_token,
            'llm_prompt_text_speech_duration_mask': prompt_text_speech_duration_mask,
            
            # Flow 所需的 Prompt Info
            'flow_prompt_duration_token': prompt_regulate_duration, # [1, T_feat]
            'prompt_duration_phase': prompt_duration_phase,            # [1, T_feat] <--- 新增 Key
            
            'flow_prompt_speech_token': speech_token, 
            'flow_prompt_speech_token_len': speech_token_len,
            'prompt_speech_feat': speech_feat, 
            'prompt_speech_feat_len': speech_feat_len,
            'llm_embedding': embedding, 
            'flow_embedding': embedding,
            "out_path": out_path,
            "watermark_bits": watermark_bits
        }
        return model_input

    def frontend_no_ref(self, tts_text, watermark_bits=None, out_path=None, model_dir=None):
        """Build model input without a reference audio for watermark-only synthesis.

        If a spk_embeddings.pt file exists in model_dir, randomly sample from it.
        Otherwise fall back to a fixed-seed random embedding.
        """
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)

        # Load embedding pool if available
        if model_dir:
            pool_path = os.path.join(model_dir, 'spk_embeddings.pt')
            if os.path.exists(pool_path):
                pool = torch.load(pool_path, map_location=self.device)
                idx = torch.randint(0, pool.shape[0], (1,)).item()
                embedding = pool[idx].unsqueeze(0)
            else:
                rng = torch.Generator(device=self.device).manual_seed(42)
                embedding = torch.randn(1, 192, generator=rng, device=self.device)
        else:
            rng = torch.Generator(device=self.device).manual_seed(42)
            embedding = torch.randn(1, 192, generator=rng, device=self.device)
        model_input = {
            'text': tts_text_token,
            'text_len': tts_text_token_len,
            'llm_embedding': embedding,
            'flow_embedding': embedding,
            'out_path': out_path,
            'watermark_bits': watermark_bits,
        }
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k, resample_rate)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

    def frontend_instruct2(self, tts_text, instruct_text, prompt_speech_16k, resample_rate):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(instruct_text + '<|endofprompt|>')
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        if resample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                       'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        model_input = {'source_speech_token': source_speech_token, 'source_speech_token_len': source_speech_token_len,
                       'flow_prompt_speech_token': prompt_speech_token, 'flow_prompt_speech_token_len': prompt_speech_token_len,
                       'prompt_speech_feat': prompt_speech_feat, 'prompt_speech_feat_len': prompt_speech_feat_len,
                       'flow_embedding': embedding}
        return model_input
