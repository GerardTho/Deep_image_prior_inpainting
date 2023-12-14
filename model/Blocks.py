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
  def __init__(self, n_in, n_out, kernel_size, pad='zero', bias=True, scale_factor=2, mode='nearest', upsample=True):
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
    self.upsamplebool = upsample

  def forward(self, x):
    x = self.bn0(x)
    x = self.act1(self.bn1(self.conv1(x)))
    x = self.act2(self.bn2(self.conv2(x)))
    if self.upsamplebool :
      x = self.upsample(x)
    return x

class SkipBlock(nn.Module):
  """
  Skip block to create skip connections
  """
  def __init__(self, n_in, n_out, kernel_size, pad='zero', bias=True):
    super(SkipBlock, self).__init__()
    if pad == "zero":
      to_pad = int((kernel_size[0] - 1) / 2)

    self.conv1 = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=kernel_size, stride=1, padding=to_pad, bias=bias)
    self.bn1 = nn.BatchNorm2d(n_out)
    self.act1 = nn.LeakyReLU()

  def forward(self, x):
    x = self.act1(self.bn1(self.conv1(x)))
    return x
