import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers.helpers import to_2tuple

from einops.einops import rearrange

from FAMDnet.utils.registries import MODEL_REGISTRY

from .base import BaseNetwork
from .xception import Xception
from .efficientnet import EfficientNet
from .modules.head import Classifier2D, Localizer
from .modules.transformer_block import FeedForward2D



class StarReLU(nn.Module):

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=False,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        # print(f"DynamicFilter input shape: {x.shape}")

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)

        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)

        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        query.size(-1)
    )
    p_attn = F.softmax(scores, dim=-1)
    p_val = torch.matmul(p_attn, value)
    return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        attentions = []
        for (width, height), query, key, value in zip(
            self.patchsize,
            torch.chunk(_query, len(self.patchsize), dim=1),
            torch.chunk(_key, len(self.patchsize), dim=1),
            torch.chunk(_value, len(self.patchsize), dim=1),
        ):
            out_w, out_h = w // width, h // height

            # 1) embedding and reshape
            query = query.view(b, d_k, out_h, height, out_w, width)
            query = (
                query.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )
            key = key.view(b, d_k, out_h, height, out_w, width)
            key = (
                key.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )
            value = value.view(b, d_k, out_h, height, out_w, width)
            value = (
                value.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )

            y, _ = attention(query, key, value)

            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, out_h, out_w, d_k, height, width)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)
            attentions.append(y)
            output.append(y)

        output = torch.cat(output, 1)
        self_attention = self.output_linear(output)

        return self_attention


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, in_channel=256):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=in_channel)
        self.feed_forward = FeedForward2D(
            in_channel=in_channel, out_channel=in_channel
        )

    def forward(self, rgb):
        self_attention = self.attention(rgb)
        output = rgb + self_attention
        output = output + self.feed_forward(output)
        return output


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class MatMul(nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class LinAngularAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kq_matmul = MatMul()
        self.kqv_matmul = MatMul()
        if self.sparse_reg:
            self.qk_matmul = MatMul()
            self.sv_matmul = MatMul()

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, x):
        N, L, C = x.shape   #x=(B,H*W,C)
        """N, C, H, W = x.shape # x=(B, C, H, W)
        L = H * W"""
        qkv = (
            self.qkv(x)
            .reshape(N, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.sparse_reg:
            attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v)

        attn = self.kq_matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                self.sv_matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * self.kqv_matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.LinAngularAttention = LinAngularAttention(in_channels=128)

    def forward(self, x):
        # x1 = self.LinAngularAttention(x)
        B, N, C = x.shape
        """B, C, H, W = x.shape  # x=(B, C, H, W)
        N = H * W"""
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)    # torch.Size([3, 32, 8, 16, 784])
        q, k, v = qkv.unbind(0)                                # q, k, v = torch.Size([32, 8, 16, 784])
        q = torch.nn.functional.normalize(q, dim=-1)           # torch.Size([32, 8, 16, 784])
        k = torch.nn.functional.normalize(k, dim=-1)           # torch.Size([32, 8, 16, 784])
        attn = (q @ k.transpose(-2, -1)) * self.temperature    # torch.Size([32, 8, 16, 16])
        attn = attn.softmax(dim=-1)                            # torch.Size([32, 8, 16, 16])
        attn = self.attn_drop(attn)                            # torch.Size([32, 8, 16, 16])
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)    # torch.Size([32, 784, 128])
        x = self.proj(x)                                       # torch.Size([32, 784, 128])
        x = self.proj_drop(x)                                  # torch.Size([32, 784, 128])
        x = x # + x1
        return x


class CAFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(self):
        super(CAFM, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(32, 16, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(32, 16, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(32, 16, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(32, 16, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(16, 32, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(16, 32, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(16, 32, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(16, 32, 1, stride=1, padding=0)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.reshape([b, 1, -1])
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])
        a2 = F.softmax(a2, dim=-1)

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2

        f1 = f1.squeeze(0)
        f2 = f2.squeeze(0)

        f1 = f1.transpose(0, 1)
        f2 = f2.transpose(0, 1)

        f = f1 + f2

        return f


class LinAngularXCA_CA(nn.Module):
    def __init__(self):
        super(LinAngularXCA_CA, self).__init__()
        self.la = LinAngularAttention(in_channels=32)
        self.xa = XCA(dim=32)
        self.cafm = CAFM()

    def forward(self, rgb, freq):
        rgb = to_3d(rgb)
        la1 = self.la(rgb)
        la1 = to_4d(la1, 80, 80)
        freq = to_3d(freq)
        xa1 = self.xa(freq)
        xa1 = to_4d(xa1, 80, 80)

        result = self.cafm(la1, xa1)
        # print(result.shape)
        result = result.permute(1, 2, 0)
        result = to_4d(result, 80, 80)

        return result


class PatchTrans(BaseNetwork):
    def __init__(self, in_channel, in_size):
        super(PatchTrans, self).__init__()
        self.in_size = in_size

        patchsize = [
            (in_size, in_size),
            (in_size // 2, in_size // 2),
            (in_size // 4, in_size // 4),
            (in_size // 8, in_size // 8),
        ]

        self.t = TransformerBlock(patchsize, in_channel=in_channel)

    def forward(self, enc_feat):
        output = self.t(enc_feat)
        return output


@MODEL_REGISTRY.register()
class FAMDnet(BaseNetwork):
    def __init__(self, model_cfg):
        super(FAMDnet, self).__init__()
        img_size = model_cfg["IMG_SIZE"]
        backbone = model_cfg["BACKBONE"]
        texture_layer = model_cfg["TEXTURE_LAYER"]
        feature_layer = model_cfg["FEATURE_LAYER"]
        depth = model_cfg["DEPTH"]
        num_classes = model_cfg["NUM_CLASSES"]
        drop_ratio = model_cfg["DROP_RATIO"]
        has_decoder = model_cfg["HAS_DECODER"]

        freq_h = img_size // 4
        freq_w = freq_h // 2 + 1

        if "xception" in backbone:
            self.model = Xception(num_classes)
        elif backbone.split("-")[0] == "efficientnet":
            self.model = EfficientNet({'NAME': backbone, 'PRETRAINED': True})

        self.texture_layer = texture_layer
        self.feature_layer = feature_layer

        with torch.no_grad():
            input = {"img": torch.zeros(1, 3, img_size, img_size)}
            layers = self.model(input)
        texture_dim = layers[self.texture_layer].shape[1]
        feature_dim = layers[self.feature_layer].shape[1]

        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PatchTrans(in_channel=texture_dim, in_size=freq_h),
                        DynamicFilter(32, size=80),
                        LinAngularXCA_CA(),
                    ]
                )
            )

        self.classifier = Classifier2D(
            feature_dim, num_classes, drop_ratio, "sigmoid"
        )

        self.has_decoder = has_decoder
        if self.has_decoder:
            self.decoder = Localizer(texture_dim, 1)

    def forward(self, x):
        rgb = x["img"]
        B = rgb.size(0)

        layers = {}
        rgb = self.model.extract_textures(rgb, layers)

        for attn, filter, cma in self.layers:
            rgb = attn(rgb)
            freq = filter(rgb)
            rgb = cma(rgb, freq)

        features = self.model.extract_features(rgb, layers)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(B, features.size(1))

        logits = self.classifier(features)

        if self.has_decoder:
            mask = self.decoder(rgb)
            mask = mask.squeeze(-1)

        else:
            mask = None

        output = {"logits": logits, "mask": mask, "features:": features}
        return output


if __name__ == "__main__":
    from torchsummary import summary

    model = FAMDnet(num_classes=2, has_decoder=False)
    model.cuda()
    summary(model, input_size=(3, 320, 320), batch_size=12, device="cuda")
