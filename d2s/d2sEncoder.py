##
# This code is modified based on https://github.com/SJTUwxz/LoCoNet_ASD.git
##

import torch
import torch.nn as nn
from torchvggish import vggish
from d2s.visualEncoder import visualFrontend, visualConv1D, visualTCN
from d2s.sat import SAT
from d2s.cat import CAT

pretrained_weights = {
    'vggish': "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"
}

class d2sEncoder(nn.Module):

    def __init__(self):
        super(d2sEncoder, self).__init__()

        self.visualFrontend = visualFrontend()
        self.visualTCN = visualTCN()
        self.visualConv1D = visualConv1D()

        self.audioEncoder = vggish.VGGish(pretrained_weights, preprocess=False, postprocess=False)
        self.audio_pool = nn.AdaptiveAvgPool1d(1)

        self.crossA2V = CAT(d_model=128, nhead=8)
        self.crossV2A = CAT(d_model=128, nhead=8)

        self.speaker_embed = nn.Embedding(3, 256)
        self.conv_attn_sub1 = SAT(num_blocks=1, embed_dim=256, num_sub=3, attn_type="S")
        self.conv_attn_time1 = SAT(num_blocks=1, embed_dim=256, num_sub=3, attn_type="T")
        self.conv_attn_sub2 = SAT(num_blocks=1, embed_dim=256, num_sub=3, attn_type="S")
        self.conv_attn_time2 = SAT(num_blocks=1, embed_dim=256, num_sub=3, attn_type="T")
        self.crossT2S1 = CAT(d_model=256, nhead=8)
        self.crossS2T1 = CAT(d_model=256, nhead=8)
        self.crossT2S2 = CAT(d_model=256, nhead=8)
        self.crossS2T2 = CAT(d_model=256, nhead=8)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape
        x = x.view(B * T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)
        x = x.transpose(1, 2)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1, 2)
        return x

    def forward_audio_frontend(self, x):
        t = x.shape[-2]
        numFrames = t // 4
        pad = 8 - (t % 8)
        x = torch.nn.functional.pad(x, (0, 0, 0, pad), "constant")
        x = self.audioEncoder(x)
        b, c, t2, freq = x.shape
        x = x.view(b * c, t2, freq)
        x = self.audio_pool(x)
        x = x.view(b, c, t2)[:, :, :numFrames]
        x = x.permute(0, 2, 1)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src=x1, tar=x2, adjust=0)
        x2_c = self.crossV2A(src=x2, tar=x1, adjust=0)
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2, b=1, s=1):
        x_ori = torch.cat((x1, x2), 2)
        x1_sub = x_ori.permute(1, 0, 2)
        speaker_ids = torch.arange(3, device=x1_sub.device)
        speaker_embed = self.speaker_embed(speaker_ids)
        x1_sub += speaker_embed
        x1_sub = x1_sub.permute(1, 0, 2)
        x1_sub = self.conv_attn_sub1(x1_sub)
        x1_time = x_ori
        x1_time = self.conv_attn_time1(x1_time)
        x2_time = self.crossT2S1(src=x1_time, tar=x1_sub, adjust=0)
        x2_sub = self.crossS2T1(src=x1_sub, tar=x1_time, adjust=0)
        x2_sub = self.conv_attn_sub2(x2_sub)
        x2_time = self.conv_attn_time2(x2_time)
        result_time = self.crossT2S2(src=x2_time, tar=x2_sub, adjust=0)
        result_sub = self.crossS2T2(src=x2_sub, tar=x2_time, adjust=0)
        result = result_time + result_sub
        result = torch.reshape(result, (-1, 256))
        return result

    def forward_audio_backend(self, x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self, x):
        x = torch.reshape(x, (-1, 128))
        return x
