import torch.nn as nn

class DownSamplerBlock(nn.Module):
  """
  Downsampling block composed of 2 convolutional layers with associated batch normalization and activation function
  The down sampling effect is done on the first convolutional if stride > 1
  """
  def __init__(self, n_in, n_out, kernel_size, stride=2, pad='zero', bias=True):
    super(DownSamplerBlock, self).__init__()

    if pad == "zero":
      to_pad = int((kernel_size[0] - 1) / 2)

    self.conv1 = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=kernel_size, stride=stride, padding=to_pad, bias=bias)
    self.bn1 = nn.BatchNorm2d(n_out)
    self.act1 = nn.LeakyReLU()

    self.conv2 = nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=kernel_size, stride=1, padding=to_pad, bias=bias)
    self.bn2 = nn.BatchNorm2d(n_out)
    self.act2 = nn.LeakyReLU()

  def forward(self, x):
    x = self.act1(self.bn1(self.conv1(x)))
    x = self.act2(self.bn2(self.conv2(x)))
    return x

class UpSamplerBlock(nn.Module):
  """
  Upsampling block composed of 2 convolutional layers with associated batch normalization and activation function
  The effective upsampling is done after the 2 convolutional layers
  """
  def __init__(self, n_in, n_out, kernel_size, pad='zero', bias=True, scale_factor=2, mode='nearest'):
    super(UpSamplerBlock, self).__init__()
    if pad == "zero":
      to_pad = int((kernel_size[0] - 1) / 2)

    self.bn0 = nn.BatchNorm2d(n_in)

    self.conv1 = nn.Conv2d(in_channels=n_in, out_channels=n_in, kernel_size=kernel_size, stride=1, padding=to_pad, bias=bias)
    self.bn1 = nn.BatchNorm2d(n_in)
    self.act1 = nn.LeakyReLU()

    self.conv2 = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1,1), stride=1, padding=0, bias=bias)
    self.bn2 = nn.BatchNorm2d(n_out)
    self.act2 = nn.LeakyReLU()

    self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

  def forward(self, x):
    x = self.bn0(x)
    x = self.act1(self.bn1(self.conv1(x)))
    x = self.act2(self.bn2(self.conv2(x)))
    x = self.upsample(x)
    return x

class EncoderDecoder(nn.Module):
  """
  Encoder Decoder model starting with Downsampling blocks and finishing with Upsampling blocks
  """
  def __init__(self, num_input_channels,
               num_output_channels,
               n_upsampler=[128, 128, 128, 64, 32, 16],
               n_downsampler=[16, 32, 64, 128, 128, 128],
               k_upsampler=[5, 5, 5, 5, 5, 5],
               k_downsampler=[3, 3, 3, 3, 3, 3],
               stride=2,
               pad='zero',
               bias=True,
               scale_factor=2,
               mode_upscale='nearest'):
    super(EncoderDecoder, self).__init__()

    if (len(n_upsampler) != len(k_upsampler)) or (len(n_downsampler) != len(k_downsampler)):
      raise ValueError("Unable to instantiate convolutional layers due to a difference of length between kernel sizes list and input sizes list")

    self.downsample_block1 = DownSamplerBlock(num_input_channels, n_downsampler[0], (k_downsampler[0], k_downsampler[0]), stride=stride, pad=pad, bias=bias)
    self.downsample_block2 = DownSamplerBlock(n_downsampler[0], n_downsampler[1], (k_downsampler[1], k_downsampler[1]), stride=stride, pad=pad, bias=bias)
    self.downsample_block3 = DownSamplerBlock(n_downsampler[1], n_downsampler[2], (k_downsampler[2], k_downsampler[2]), stride=stride, pad=pad, bias=bias)
    self.downsample_block4 = DownSamplerBlock(n_downsampler[2], n_downsampler[3], (k_downsampler[3], k_downsampler[3]), stride=stride, pad=pad, bias=bias)
    self.downsample_block5 = DownSamplerBlock(n_downsampler[3], n_downsampler[4], (k_downsampler[4], k_downsampler[4]), stride=stride, pad=pad, bias=bias)
    self.downsample_block6 = DownSamplerBlock(n_downsampler[4], n_downsampler[5], (k_downsampler[5], k_downsampler[5]), stride=stride, pad=pad, bias=bias)

    self.upsample_block1 = UpSamplerBlock(n_downsampler[-1], n_upsampler[0], (k_upsampler[0], k_upsampler[0]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)
    self.upsample_block2 = UpSamplerBlock(n_upsampler[0], n_upsampler[1], (k_upsampler[1], k_upsampler[1]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)
    self.upsample_block3 = UpSamplerBlock(n_upsampler[1], n_upsampler[2], (k_upsampler[2], k_upsampler[2]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)
    self.upsample_block4 = UpSamplerBlock(n_upsampler[2], n_upsampler[3], (k_upsampler[3], k_upsampler[3]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)
    self.upsample_block5 = UpSamplerBlock(n_upsampler[3], n_upsampler[4], (k_upsampler[4], k_upsampler[4]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)
    self.upsample_block6 = UpSamplerBlock(n_upsampler[4], n_upsampler[5], (k_upsampler[5], k_upsampler[5]), pad=pad, bias=bias, scale_factor=scale_factor, mode=mode_upscale)

    self.conv = nn.Conv2d(in_channels=n_upsampler[-1], out_channels=num_output_channels, kernel_size=(1, 1), stride=1, padding=0, bias=bias)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.downsample_block1(x)
    x = self.downsample_block2(x)
    x = self.downsample_block3(x)
    x = self.downsample_block4(x)
    x = self.downsample_block5(x)
    x = self.downsample_block6(x)

    x = self.upsample_block1(x)
    x = self.upsample_block2(x)
    x = self.upsample_block3(x)
    x = self.upsample_block4(x)
    x = self.upsample_block5(x)
    x = self.upsample_block6(x)

    x = self.conv(x)

    return self.sigmoid(x)