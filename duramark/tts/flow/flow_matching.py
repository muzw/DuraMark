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
import onnxruntime
import torch
import torch.nn.functional as F
from matcha.models.components.flow_matching import BASECFM


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, detector: torch.nn.Module = None, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator
        self.detector = detector

        # ============================================================
        # [修改点 1]: 彻底冻结 Detector，防止共谋 (Co-adaptation)
        # ============================================================
        if self.detector is not None:
            # 1. 关闭梯度需求：确保 optimizer 不会更新它
            for param in self.detector.parameters():
                param.requires_grad = False
            # 2. 强制设为 eval 模式：冻结 BatchNorm 统计量和 Dropout
            self.detector.eval()

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, flow_cache=torch.zeros(1, 80, 0, 2)):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu) * temperature
        cache_size = flow_cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = flow_cache[:, :, :, 0]
            mu[:, :, :cache_size] = flow_cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        flow_cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), flow_cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        if self.inference_cfg_rate > 0:
            # Do not use concat, it may cause memory format changed and trt infer with wrong results!
            x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
            mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
            mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
            t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
            spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
            cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        else:
            x_in, mask_in, mu_in, t_in, spks_in, cond_in = x, mask, mu, t, spks, cond
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            if self.inference_cfg_rate > 0:
                x_in[:] = x
                mask_in[:] = mask
                mu_in[0] = mu
                t_in[:] = t.unsqueeze(0)
                spks_in[0] = spks
                cond_in[0] = cond
            else:
                x_in, mask_in, mu_in, t_in, spks_in, cond_in = x, mask, mu, t, spks, cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in
            )
            if self.inference_cfg_rate > 0:
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator.forward(x, mask, mu, t, spks, cond)
        elif isinstance(self.estimator, onnxruntime.InferenceSession):
            ort_inputs = {
                'x': x.cpu().numpy(),
                'mask': mask.cpu().numpy(),
                'mu': mu.cpu().numpy(),
                't': t.cpu().numpy(),
                'spks': spks.cpu().numpy(),
                'cond': cond.cpu().numpy()
            }
            output = self.estimator.run(None, ort_inputs)[0]
            return torch.tensor(output, dtype=x.dtype, device=x.device)
        else:
            self.estimator.set_input_shape('x', (2, 80, x.size(2)))
            self.estimator.set_input_shape('mask', (2, 1, x.size(2)))
            self.estimator.set_input_shape('mu', (2, 80, x.size(2)))
            self.estimator.set_input_shape('t', (2,))
            self.estimator.set_input_shape('spks', (2, 80))
            self.estimator.set_input_shape('cond', (2, 80, x.size(2)))
            # run trt engine
            self.estimator.execute_v2([x.contiguous().data_ptr(),
                                       mask.contiguous().data_ptr(),
                                       mu.contiguous().data_ptr(),
                                       t.contiguous().data_ptr(),
                                       spks.contiguous().data_ptr(),
                                       cond.contiguous().data_ptr(),
                                       x.data_ptr()])
            return x

    def compute_loss(self, x1, mask, mu, spks=None, detector_input=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # t = torch.zeros([b, 1, 1], device=mu.device, dtype=mu.dtype)

        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)
        flow_loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        
        if detector_input:
            # text_token, text_token_len, speech_token_len, speech_feat_len, alignment = detector_input
            # # ==========================================================
            # # [改进点]: 使用 ODE 轨迹公式推导 x1 (Clean Mel)
            # # 公式: x1 = x_t + (1 - t) * v_t
            # # ==========================================================
            
            # # y 是当前的 x_t
            # # pred 是预测的 v_t
            # # t 的维度是 [B, 1, 1]，可以直接广播到 [B, 80, T]
            # estimated_clean_mel = y + (1.0 - t) * pred

            # # # 2. 确保 Detector 冻结并 eval (保持之前的修正)
            # self.detector.eval()
            # duration_loss, acc = self.detector.forward_by_flow(text_token, text_token_len, estimated_clean_mel.transpose(1, 2).contiguous(), speech_feat_len, speech_token_len, alignment, device=pred.device)
            # return flow_loss, duration_loss, acc
            text_token, text_token_len, speech_token_len, speech_feat_len, alignment = detector_input
            
            # 1. 还原 x1
            estimated_clean_mel = y + (1.0 - t) * pred
            
            # 2. 筛选 t > 0.4 的样本
            t_vals = t.view(-1)
            valid_mask = t_vals > -1
            
            if valid_mask.any():
                # --- [A] 筛选 Lengths 和 Token ---
                sub_text_token = text_token[valid_mask]
                sub_text_token_len = text_token_len[valid_mask]
                sub_speech_token_len = speech_token_len[valid_mask]
                sub_speech_feat_len = speech_feat_len[valid_mask]
                
                # --- [B] 筛选 Mel 和 Alignment ---
                sub_estimated_clean_mel = estimated_clean_mel[valid_mask]
                sub_alignment = alignment[valid_mask]

                # ==========================================================
                # [关键修正]: 重新计算当前子集的最大长度并裁剪
                # ==========================================================
                # 获取筛选后这批数据的最大真实长度
                curr_max_speech_len = sub_speech_feat_len.max()
                
                # 1. 裁剪 Mel 谱图
                # Mel 形状: [Batch, n_feats, Time] -> 检查第 2 维
                if sub_estimated_clean_mel.size(2) > curr_max_speech_len:
                    sub_estimated_clean_mel = sub_estimated_clean_mel[:, :, :curr_max_speech_len]
                
                # 2. 裁剪 Alignment
                # Alignment 形状: [Batch, Time] -> 检查第 1 维
                if sub_alignment.size(1) > curr_max_speech_len:
                    sub_alignment = sub_alignment[:, :curr_max_speech_len] # 注意这里只用两个冒号
                # ==========================================================

                self.detector.eval()
                
                # 传入裁剪后的数据
                duration_loss, acc = self.detector.forward_by_flow(
                    sub_text_token, 
                    sub_text_token_len, 
                    sub_estimated_clean_mel.transpose(1, 2).contiguous(), 
                    sub_speech_feat_len, 
                    sub_speech_token_len, 
                    sub_alignment, 
                    device=pred.device
                )
            else:
                duration_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
                acc = torch.tensor(0.0, device=pred.device)
            return flow_loss, duration_loss, acc
        else:
            return flow_loss


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        self.rand_noise = torch.randn([1, 80, 50 * 300])

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = self.rand_noise[:, :, :mu.size(2)].to(mu.device) * temperature
        if self.fp16 is True:
            z = z.half()
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None
