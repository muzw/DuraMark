import torch
import torch.nn as nn
import torch.nn.functional as F
from duramark.tts.utils.mask import make_pad_mask

class ConvBlock(nn.Module):
    """
    封装了：Conv1d -> Transpose -> LayerNorm -> ReLU -> Transpose
    输入输出保持 (B, C, T) 格式，内部处理 LayerNorm 的维度问题
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=kernel_size//2
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # x input: (B, C, T)
        x = self.conv(x)       # -> (B, C_out, T_new)
        
        # --- 你的核心逻辑开始 ---
        x = x.transpose(1, 2)  # -> (B, T_new, C_out) 适配 LayerNorm
        x = self.norm(x)
        x = self.act(x)
        x = x.transpose(1, 2)  # -> (B, C_out, T_new) 转回卷积格式
        # --- 你的核心逻辑结束 ---
        
        return x

class DownsampleRegulator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, kernel_size=3):
        super().__init__()
        
        # 1. 初始特征提取 (86Hz)
        # 使用封装好的 Block，代码更整洁
        self.pre_conv = ConvBlock(input_dim, hidden_dim, kernel_size, stride=1)
        
        # 2. 过渡层 (100Hz)
        self.mid_conv = ConvBlock(hidden_dim, hidden_dim, kernel_size, stride=1)
        
        # 3. 降采样层 (100Hz -> 50Hz)
        # Stride=2 物理减半
        self.down_conv = ConvBlock(hidden_dim, output_dim, kernel_size, stride=2)

    def forward(self, x, xlens, target_len_50hz):
        """
        Args:
            x: (B, T_86hz, D)
            xlens: (B,)
            target_len_50hz: (B,)
        """
        # (B, T, D) -> (B, D, T) 初始转置，进入卷积模式
        x = x.transpose(1, 2)
        
        # 1. Mask 原始输入
        if xlens is not None:
            mask = (~make_pad_mask(xlens)).to(x).unsqueeze(1)
            x = x * mask
            
        # 2. 提取特征 (Block 内部会自动处理 transpose-norm-transpose)
        x = self.pre_conv(x)
        
        # 3. 插值到 100Hz (在 Channel 维度进行，所以需要 B, C, T)
        max_target_len = target_len_50hz.max().item()
        intermediate_len = max_target_len * 2
        
        x_100hz = F.interpolate(
            x, 
            size=intermediate_len, 
            mode='linear', 
            align_corners=False
        )
        
        # 4. 中间层处理
        x_100hz = self.mid_conv(x_100hz)
        
        # 5. 物理下采样 (100Hz -> 50Hz)
        out = self.down_conv(x_100hz)
        
        # (B, D, T') -> (B, T', D) 最后转回文本格式
        out = out.transpose(1, 2)
        
        # 6. Mask 输出
        if target_len_50hz is not None:
            # 严格对齐长度
            if out.size(1) != max_target_len:
                print(out.size(1),max_target_len)
                out = out.transpose(1, 2)
                out = F.interpolate(out, size=max_target_len, mode='nearest')
                out = out.transpose(1, 2)

            out_mask = (~make_pad_mask(target_len_50hz)).to(out).unsqueeze(-1)
            out = out * out_mask
            
        return out, target_len_50hz