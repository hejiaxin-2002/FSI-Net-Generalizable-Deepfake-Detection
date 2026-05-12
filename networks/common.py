import torch
import torch.nn as nn
import torch.nn.functional as F


class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # Simple Pixel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


class ElementScale(nn.Module):
    
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__() 


        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad  
        )


    def forward(self, x):
        return x * self.scale  


class MultiOrderDWConv(nn.Module):

    
    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3],
                 channel_split=[1, 3, 4],
                ):
        super(MultiOrderDWConv, self).__init__()

      
        self.split_ratio = [i / sum(channel_split) for i in channel_split]

        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)  
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims) 
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2  

        self.embed_dims = embed_dims

    
        assert len(dw_dilation) == len(channel_split) == 3  
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3  
        assert embed_dims % sum(channel_split) == 0  

  
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2, 
            groups=self.embed_dims,            
            stride=1,
            dilation=dw_dilation[0],            
        )

        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1,
            dilation=dw_dilation[1], 
        )

        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1,
            dilation=dw_dilation[2],  
        )
     
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1
        )

    def forward(self, x):
        x_0 = self.DW_conv0(x)  

       
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])

     
        x_2 = self.DW_conv2(x_0[:, self.embed_dims - self.embed_dims_2:, ...])


        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)

        x = self.PW_conv(x)  
        return x


class MultiOrderGatedAggregation(nn.Module):

    
    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_force_fp32=False,
                 ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
      
        self.proj_1 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        
        self.gate = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
    
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )

  
        self.proj_2 = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        
        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)


    def feat_decompose(self, x):
       
        x = self.proj_1(x)


        x_d = F.adaptive_avg_pool2d(x, output_size=1)

  
        x = x + self.sigma(x - x_d)
        x = self.act_value(x) 
        return x

    def forward(self, x):
        shortcut = x.clone()  

       
        x = self.feat_decompose(x)

      
        F_branch = self.gate(x)
        G_branch = self.value(x)

    
        x = self.proj_2(self.act_gate(F_branch) * self.act_gate(G_branch))
        x = x + shortcut  

        return x


class Dense(nn.Module):
    def __init__(self, in_channels):
        super(Dense, self).__init__()
      
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.gelu(x1 + x)
        x2 = self.conv2(x1)
        x2 = self.gelu(x2 + x1 + x)
        x3 = self.conv3(x2)
        x3 = self.gelu(x3 + x2 + x1 + x)
        x4 = self.conv4(x3)
        x4 = self.gelu(x4 + x3 + x2 + x1 + x)
        x5 = self.conv5(x4)
        x5 = self.gelu(x5 + x4 + x3 + x2 + x1 + x)
        x6 = self.conv6(x5)
        x6 = self.gelu(x6 + x5 + x4 + x3 + x2 + x1 + x)
        return x6


# Channel Attention (CA) Layer
class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class UNet(nn.Module):
    def __init__(self, in_channels, wave):
        super(UNet, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
   
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class HLFD(nn.Module):
    def __init__(self, dim, wave='haar'):
        super(HLFD, self).__init__()
        n_feats = dim
        self.down = nn.AvgPool2d(kernel_size=2)
        self.dense = Dense(n_feats)
        self.unet = UNet(n_feats, wave)
        self.alise1 = nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0)
        self.alise2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.att = CALayer(n_feats)

    def forward(self, x):
        # x: shape [B, 32, 64, 64]
        low = self.down(x)  
        up = F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)
        high = x - up  
        lowf = self.unet(low) 
        highfeat = self.dense(high) 
        lowfeat = F.interpolate(lowf, size=x.size()[-2:], mode='bilinear', align_corners=True)
        temp = self.alise1(torch.cat([highfeat, lowfeat], dim=1))
        temp = self.att(temp)
        out = self.alise2(temp) + x
        return out
