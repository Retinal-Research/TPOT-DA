import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=('relu1_2', 'relu2_2', 'relu3_3')):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features

        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15
        }
        max_idx = max([self.layer_map[l] for l in layers])
        self.vgg = nn.Sequential(*list(vgg.children())[:max_idx + 1])
        self.selected_layers = [self.layer_map[l] for l in layers]

        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.selected_layers:
                features.append(x)
        return features

class Adapter(nn.Module):
    def __init__(self, channels: int = 3, scale=1.0, dropout=0.3):
        super().__init__()
        self.scale = scale
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.scale * self.adapter(x)


class BottleneckAdapter(nn.Module):
    def __init__(self, in_channels, reduction=4, scale=0.5, use_deep=True, dropout=0.3):
        super().__init__()
        hidden = max(4, in_channels // reduction)
        self.scale = scale

        if use_deep:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, in_channels, kernel_size=1)
            )
        else:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(hidden, in_channels, kernel_size=1)
            )

    def forward(self, x):
        return x + self.scale * self.adapter(x)

    
class GeneratorWithFullAdapter(nn.Module):
    def __init__(self, base_generator, n_feat: int = 80, scale_unetfeats: int = 48):
        super().__init__()
        self.base = base_generator

        encoder_ch3 = n_feat + 2 * scale_unetfeats  # 176
        decoder_ch1 = n_feat  # 80

        for p in self.base.parameters():
            p.requires_grad = False

        for p in self.base.stage1_encoder.encoder_level3.parameters():
            p.requires_grad = True
        for p in self.base.stage1_decoder.decoder_level3.parameters():
            p.requires_grad = True
        for p in self.base.sam12.parameters():
            p.requires_grad = True

        # ✅ Only keep adapter on encoder_level3 and decoder output
        self.adapter_enc3    = BottleneckAdapter(encoder_ch3, scale=0.5, use_deep=True)
        self.adapter_dec3    = BottleneckAdapter(encoder_ch3, scale=0.5, use_deep=True)
        self.adapter_fea3_0  = Adapter(decoder_ch1, scale=1.0)
        self.adapter_image   = Adapter(3, scale=1.0)

    def forward(self, x):
        inputs = x
       
        fea = self.base.shallow_feat1(x)
        enc1 = self.base.stage1_encoder.encoder_level1(fea)
        x = self.base.stage1_encoder.down12(enc1)

        enc2 = self.base.stage1_encoder.encoder_level2(x)
        x = self.base.stage1_encoder.down23(enc2)

        # Only adapter on encoder_level3
        enc3 = self.base.stage1_encoder.encoder_level3(x)
        enc3 = self.adapter_enc3(enc3)

        dec3 = self.base.stage1_decoder.decoder_level3(enc3)
        dec3 = self.adapter_dec3(dec3)

        
        dec2 = self.base.stage1_decoder.up32(dec3, self.base.stage1_decoder.skip_attn2(enc2))
        dec2 = self.base.stage1_decoder.decoder_level2(dec2)

        dec1 = self.base.stage1_decoder.up21(dec2, self.base.stage1_decoder.skip_attn1(enc1))
        dec1 = self.base.stage1_decoder.decoder_level1(dec1)

        dec1 = self.adapter_fea3_0(dec1)

        
        _, out = self.base.sam12(dec1, inputs)

        out = self.adapter_image(out)
        out = torch.clamp(out, 0, 1)
        return out




