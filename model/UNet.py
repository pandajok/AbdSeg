import torch
import torch.nn as nn
import torch.nn.functional as F
import math
__all__ = ['conv3x3x3', 'conv1x1x1', '_ConvINReLU3D', '_ConvIN3D']
from functools import reduce
import pdb
class UNet(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()

        # UNet parameter.
        num_class = cfg['NUM_CLASSES']
        num_channel = cfg['NUM_CHANNELS']
        self.num_depth = cfg['NUM_DEPTH']
        self.is_preprocess = cfg['IS_PREPROCESS']
        self.is_postprocess = cfg['IS_POSTPROCESS']

        encoder_conv_block = ResFourLayerConvBlock
        decoder_conv_block = ResTwoLayerConvBlock


        self.input = InputLayer(input_size=cfg['INPUT_SIZE'], clip_window=cfg['WINDOW_LEVEL'])
        self.output = OutputLayer()

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = encoder_conv_block(1, num_channel[0], num_channel[0])
        self.conv1_0 = encoder_conv_block(num_channel[0], num_channel[1], num_channel[1])
        self.conv2_0 = encoder_conv_block(num_channel[1], num_channel[2], num_channel[2])
        self.conv3_0 = encoder_conv_block(num_channel[2], num_channel[3], num_channel[3])
        self.conv4_0 = encoder_conv_block(num_channel[3], num_channel[4], num_channel[4])

        self.conv3_1 = decoder_conv_block(num_channel[3] + num_channel[4], num_channel[3], num_channel[3])
        self.conv2_2 = decoder_conv_block(num_channel[2] + num_channel[3], num_channel[2], num_channel[2])
        self.conv1_3 = decoder_conv_block(num_channel[1] + num_channel[2], num_channel[1], num_channel[1])
        self.conv0_4 = decoder_conv_block(num_channel[0] + num_channel[1], num_channel[0], num_channel[0])

        self.final = nn.Conv3d(num_channel[0], num_class, kernel_size=1, bias=False)
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
            out_size = x.shape[2:]
            if self.is_preprocess:
                x = self.input(x)

            x = self.conv0_0(x)
            x1_0 = self.conv1_0(self.pool(x))
            x2_0 = self.conv2_0(self.pool(x1_0))
            x3_0 = self.conv3_0(self.pool(x2_0))
            x4_0 = self.conv4_0(self.pool(x3_0))


            x3_0 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_0 = self.conv2_2(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_0 = self.conv1_3(torch.cat([x1_0, self.up(x2_0)], 1))
            x = self.conv0_4(torch.cat([x, self.up(x1_0)], 1))
            x = self.final(x)
            if self.is_postprocess:
                x = self.output(x, out_size)

            return x
class InputLayer(nn.Module):
    """Input layer, including re-sample, clip and normalization image."""

    def __init__(self, input_size, clip_window):
        super(InputLayer, self).__init__()
        self.input_size = input_size
        self.clip_window = clip_window

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True)
        x = torch.clamp(x, min=self.clip_window[0], max=self.clip_window[1])
        mean = torch.mean(x)
        std = torch.std(x)
        x = (x - mean) / (1e-5 + std)
        return x


class OutputLayer(nn.Module):
    """Output layer, re-sample image to original size."""

    def __init__(self):
        super(OutputLayer, self).__init__()

    def forward(self, x, output_size):
        x = F.interpolate(x, size=output_size, mode='trilinear', align_corners=True)
        return x
class ResTwoLayerConvBlock(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1):
        super(ResTwoLayerConvBlock, self).__init__()
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit = _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        output = self.relu(output)

        return output


class ResFourLayerConvBlock(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1):
        super(ResFourLayerConvBlock, self).__init__()
        self.residual_unit_1 = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1))
        self.residual_unit_2 = nn.Sequential(
            _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit_1 = _ConvIN3D(in_channel, inter_channel, 1, stride=stride, padding=0)
        self.shortcut_unit_2 = nn.Sequential()
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        output_1 = self.residual_unit_1(x)
        output_1 += self.shortcut_unit_1(x)

        output_1 = self.relu_1(output_1)
        output_2 = self.residual_unit_2(output_1)
        output_2 += self.shortcut_unit_2(output_1)
        output_2 = self.relu_2(output_2)

        return output_2

class ResTwoLayerConvBlock1(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1):
        super(ResTwoLayerConvBlock1, self).__init__()
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit = _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output


class ResFourLayerConvBlock1(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1, is_dynamic_empty_cache=False):
        super(ResFourLayerConvBlock1, self).__init__()
        self.big_kernel_conv = BigBlock(in_channel)
        self.residual_unit_1 = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1))
        self.residual_unit_2 = nn.Sequential(
            _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1),
            SENet(out_channel))
        self.shortcut_unit_1 = _ConvIN3D(in_channel, inter_channel, 1, stride=stride, padding=0)
        self.shortcut_unit_2 = nn.Sequential()
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.big_kernel_conv(x)
        output_1 = self.residual_unit_1(x)
        output_1 += self.shortcut_unit_1(x)

        output_1 = self.relu_1(output_1)
        output_2 = self.residual_unit_2(output_1)
        output_2 += self.shortcut_unit_2(output_1)

        output_2 = self.relu_2(output_2)

        return output_2
class SENet(nn.Module):
    def __init__(self,channel,ratio=4):
        super(SENet,self).__init__()
        self.max_pool = MaxPool3d()
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//ratio,False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//ratio,channel,False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,h,w,d= x.size()
        avg = self.max_pool(x).view([b,c]) # 
        fc = self.fc(avg).view([b,c,1,1,1]) #
        return x*fc 


class MaxPool3d(torch.nn.Module):
    def __init__(self):
        super(MaxPool3d, self).__init__()

    def forward(self, x):
        b, c, h, w, d = x.size()

        pool_size = 3
        stride = pool_size

        pad_d = int(math.ceil(float(d)/float(stride))*stride - d)
        pad_w = int(math.ceil(float(w)/float(stride))*stride - w)
        pad_h = int(math.ceil(float(h)/float(stride))*stride - h)

        x = torch.nn.functional.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
        x = x.unfold(2, pool_size, stride).unfold(3, pool_size, stride).unfold(4, pool_size, stride)

        x = x.contiguous().view(b, c, -1, pool_size*pool_size*pool_size)
        x, _ = torch.topk(x, 3, dim=-1)
        x = x.mean(dim=-1, keepdim=True)

        x = x.view(b, c, -1)
        x, _ = torch.topk(x, 3, dim=-1)
        x = x.mean(dim=-1, keepdim=True)


        return x

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, p=0.2):
        super(_ConvINReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.drop = nn.Dropout3d(p=p, inplace=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.relu(x)

        return x


class _ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(_ConvIN3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
