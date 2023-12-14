import torch.nn as nn
import torch
from model.Blocks import DownSamplerBlock, UpSamplerBlock, SkipBlock

class EncoderDecoderSkipConcat(nn.Module):
  """
  Encoder Decoder model starting with Downsampling blocks and finishing with Upsampling blocks
  """
  def __init__(self, num_input_channels,
               num_output_channels,
               n_upsampler=[128, 128, 128, 64, 32, 16],
               n_downsampler=[16, 32, 64, 128, 128, 128],
               n_skip=[4,4,4,4,4,4],
               k_upsampler=[5, 5, 5, 5, 5, 5],
               k_downsampler=[3, 3, 3, 3, 3, 3],
               k_skip=[1,1,1,1,1,1],
               stride=2,
               pad='zero',
               bias=True,
               scale_factor=2,
               mode_upscale='nearest'):
    super(EncoderDecoderSkipConcat, self).__init__()

    if (len(n_upsampler) != len(k_upsampler)) or (len(n_downsampler) != len(k_downsampler)):
      raise ValueError("Unable to instantiate convolutional layers due to a difference of length between kernel sizes list and input sizes list")

    self.n_skip0 = n_skip[0]
    self.n_skip1 = n_skip[1]
    self.n_skip2 = n_skip[2]
    self.n_skip3 = n_skip[3]
    self.n_skip4 = n_skip[4]
    self.n_skip5 = n_skip[5]

    self.downsample_block1 = DownSamplerBlock(num_input_channels, n_downsampler[0], (k_downsampler[0], k_downsampler[0]), stride=stride, pad=pad, bias=bias)
    self.downsample_block2 = DownSamplerBlock(n_downsampler[0], n_downsampler[1], (k_downsampler[1], k_downsampler[1]), stride=stride, pad=pad, bias=bias)
    self.downsample_block3 = DownSamplerBlock(n_downsampler[1], n_downsampler[2], (k_downsampler[2], k_downsampler[2]), stride=stride, pad=pad, bias=bias)
    self.downsample_block4 = DownSamplerBlock(n_downsampler[2], n_downsampler[3], (k_downsampler[3], k_downsampler[3]), stride=stride, pad=pad, bias=bias)
    self.downsample_block5 = DownSamplerBlock(n_downsampler[3], n_downsampler[4], (k_downsampler[4], k_downsampler[4]), stride=stride, pad=pad, bias=bias)
    self.downsample_block6 = DownSamplerBlock(n_downsampler[4], n_downsampler[5], (k_downsampler[5], k_downsampler[5]), stride=stride, pad=pad, bias=bias)

    self.skip_block1 = SkipBlock(num_input_channels, n_skip[0], (k_skip[0], k_skip[0]), pad=pad, bias=bias)
    self.skip_block2 = SkipBlock(n_downsampler[0], n_skip[1], (k_skip[1], k_skip[1]), pad=pad, bias=bias)
    self.skip_block3 = SkipBlock(n_downsampler[1], n_skip[2], (k_skip[2], k_skip[2]), pad=pad, bias=bias)
    self.skip_block4 = SkipBlock(n_downsampler[2], n_skip[3], (k_skip[3], k_skip[3]), pad=pad, bias=bias)
    self.skip_block5 = SkipBlock(n_downsampler[3], n_skip[4], (k_skip[4], k_skip[4]), pad=pad, bias=bias)
    self.skip_block6 = SkipBlock(n_downsampler[4], n_skip[5], (k_skip[5], k_skip[5]), pad=pad, bias=bias)

    self.upsample_block1 = UpSamplerBlock(n_downsampler[-1], n_upsampler[0], (k_upsampler[0], k_upsampler[0]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)
    self.upsample_block2 = UpSamplerBlock(n_upsampler[0]+n_skip[5], n_upsampler[1], (k_upsampler[1], k_upsampler[1]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)
    self.upsample_block3 = UpSamplerBlock(n_upsampler[1]+n_skip[4], n_upsampler[2], (k_upsampler[2], k_upsampler[2]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)
    self.upsample_block4 = UpSamplerBlock(n_upsampler[2]+n_skip[3], n_upsampler[3], (k_upsampler[3], k_upsampler[3]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)
    self.upsample_block5 = UpSamplerBlock(n_upsampler[3]+n_skip[2], n_upsampler[4], (k_upsampler[4], k_upsampler[4]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)
    self.upsample_block6 = UpSamplerBlock(n_upsampler[4]+n_skip[1], n_upsampler[5], (k_upsampler[5], k_upsampler[5]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)

    self.conv = nn.Conv2d(in_channels=n_upsampler[-1]+n_skip[0], out_channels=num_output_channels, kernel_size=(1, 1), stride=1, padding=0, bias=bias)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    
    if self.n_skip0 > 0:
      s1 = self.skip_block1(x)
    x = self.downsample_block1(x)
    if self.n_skip1 > 0:
      s2 = self.skip_block2(x)
    x = self.downsample_block2(x)
    if self.n_skip2 > 0:
      s3 = self.skip_block3(x)
    x = self.downsample_block3(x)
    if self.n_skip3 > 0:
      s4 = self.skip_block4(x)
    x = self.downsample_block4(x)
    if self.n_skip4 > 0:
      s5 = self.skip_block5(x)
    x = self.downsample_block5(x)
    if self.n_skip5 > 0:
      s6 = self.skip_block6(x)
    x = self.downsample_block6(x)
  
    x = self.upsample_block1(x)
    if self.n_skip5 > 0:
      x = torch.concat((x, s6), dim=1)
    x = self.upsample_block2(x)
    if self.n_skip4 > 0:
      x = torch.concat((x, s5), dim=1)
    x = self.upsample_block3(x)
    if self.n_skip3 > 0:
      x = torch.concat((x, s4), dim=1)
    x = self.upsample_block4(x)
    if self.n_skip2 > 0:
      x = torch.concat((x, s3), dim=1)
    x = self.upsample_block5(x)
    if self.n_skip1 > 0:
      x = torch.concat((x, s2), dim=1)
    x = self.upsample_block6(x)
    if self.n_skip0 > 0:
      x = torch.concat((x, s1), dim=1)

    x = self.conv(x)

    return self.sigmoid(x)