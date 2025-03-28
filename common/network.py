import torch
import torch.nn as nn
import torch.nn.functional as F


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f K' % (num_params / 1e3))


############### Basic Convolutional Layers ###############
class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ActConv(nn.Module):
    """ Conv. with activation. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.act = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out


############### MuLUT Blocks ###############
class MuLUTUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    def __init__(self, mode, nf, upscale=1, out_c=1, dense=True):
        super(MuLUTUnit, self).__init__()
        self.act = nn.ReLU()
        self.upscale = upscale

        if mode == '2x2':
            self.conv1 = Conv(1, nf, 2)
        elif mode == '2x2d':
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == '2x2d3':
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '1x4':
            self.conv1 = Conv(1, nf, (1, 4))
        elif mode == '1x3':
            self.conv1 = Conv(1, nf, (1, 3))
        elif mode == '1x1':
            self.conv1 = Conv(3, nf, (1, 1))
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            if mode == '1x1':
                self.conv6 = Conv(nf * 5, 3, 1)
            else:
                self.conv6 = Conv(nf * 5, 1 * upscale * upscale, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            if mode == '1x1':
                self.conv6 = Conv(nf, 3 * upscale * upscale, 3)
            else:
                self.conv6 = Conv(nf, upscale * upscale, 1)
        if self.upscale > 1:
            self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        if self.upscale > 1:
            x = self.pixel_shuffle(x)
        return x


class MuLUTcUnit(nn.Module):
    """ Channel-wise MuLUT block [RGB(3D) to RGB(3D)]. """

    def __init__(self, mode, nf):
        super(MuLUTcUnit, self).__init__()
        self.act = nn.ReLU()

        if mode == '1x1' or mode == '1x1q':
            self.conv1 = Conv(3, nf, 1)
        else:
            raise AttributeError

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        if mode == '1x1q':
            self.conv6 = Conv(nf * 5, 1, 1)
        else:
            self.conv6 = Conv(nf * 5, 3, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        return x


class MuLUTmixUnit(nn.Module):
    """ Channel-wise MuLUT block [RGB(3D) to RGB(3D)]. """

    def __init__(self, mode, nf):
        super(MuLUTmixUnit, self).__init__()
        self.act = nn.ReLU()

        if mode == '2x2':
            self.conv1 = Conv(2, nf, [1,2])
        else:
            raise AttributeError

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        self.conv6 = Conv(nf * 5, 1, 1)

    def forward(self, x):
        
        x = self.act(self.conv1(x))
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        return x

class PoolNear(nn.Module):
    """ Channel-wise MuLUT block [RGB(3D) to RGB(3D)]. """

    def __init__(self):
        super(PoolNear, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
    def forward(self, x):
        x = self.pool(x)
        x = nn.functional.interpolate(x, scale_factor=4)
        return x

# class PoolNear(nn.Module):
#     """ Channel-wise MuLUT block [RGB(3D) to RGB(3D)]. """

#     def __init__(self, kernel_size=16, stride=16):
#         super(PoolNear, self).__init__()
#         self.pool = nn.AvgPool2d(kernel_size=16, stride=16)
#         self.kernel_size = kernel_size
#         self.stride = stride

#     def forward(self, x):
#         _, _, h, w = x.shape
#         # Calculate padding to make the input size a multiple of 16
#         pad_h = (self.kernel_size - x.size(2) % self.kernel_size) % self.kernel_size
#         pad_w = (self.kernel_size - x.size(3) % self.kernel_size) % self.kernel_size
        
#         # Pad the input
#         x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        
#         # Perform pooling
#         x = self.pool(x)
        
#         # Upsample back to the original size
#         x = F.interpolate(x, size=(h, w))
        
#         return x

############### Image Super-Resolution ###############
class SRNet(nn.Module):
    """ Wrapper of a generalized (spatial-wise) MuLUT block. 
        By specifying the unfolding patch size and pixel indices,
        arbitrary sampling pattern can be implemented.
    """

    def __init__(self, mode, nf=64, upscale=None, dense=True):
        super(SRNet, self).__init__()
        self.mode = mode
        print(mode)
        if 'x1' in mode:
            assert upscale is None
        if mode == 'Sx1':
            self.model = MuLUTUnit('2x2', nf, upscale=1, dense=dense)
            self.K = 2
            self.S = 1
        elif mode == 'SxN':
            self.model = MuLUTUnit('2x2', nf, upscale=1, dense=dense)
            self.K = 2
            self.S = upscale
        elif mode == 'Cx1':
            self.model = MuLUTcUnit('1x1', nf)
            self.K = 2
            self.S = 1
        elif mode == 'Px1':
            self.model = PoolNear()
            self.K = 2
            self.S = 1
        elif mode == 'Qx1':
            self.model = MuLUTcUnit('1x1q', nf)
            self.K = 2
            self.S = 1
        elif mode == 'Mx1':
            self.model_rg = MuLUTmixUnit('2x2', nf)
            self.model_gb = MuLUTmixUnit('2x2', nf)
            self.model_rb = MuLUTmixUnit('2x2', nf)
            self.K = 2
            self.S = 1
        elif mode == 'MxN':
            self.model_rg = MuLUTmixUnit('2x2', nf)
            self.model_gb = MuLUTmixUnit('2x2', nf)
            self.model_rb = MuLUTmixUnit('2x2', nf)
            self.K = 2
            self.S = 1
        elif mode == 'Vx1':
            self.model = MuLUTUnit('1x3', nf, upscale=1, dense=dense)
            self.K = 2
            self.S = 1
        elif mode == 'VxN':
            self.model = MuLUTUnit('1x3', nf, upscale=1, dense=dense)
            self.K = 2
            self.S = 1
        elif mode == 'TMx1':
            self.model = MuLUTcUnit('1x1', nf)
            self.K = 1
            self.S = 1
        elif mode == 'Dx1':
            self.model = MuLUTUnit('2x2d', nf, upscale=1, dense=dense)
            self.K = 3
            self.S = 1
        elif mode == 'DxN':
            self.model = MuLUTUnit('2x2d', nf, upscale=upscale, dense=dense)
            self.K = 3
            self.S = upscale
        elif mode == 'Yx1':
            self.model = MuLUTUnit('1x4', nf, upscale=1, dense=dense)
            self.K = 3
            self.S = 1
        elif mode == 'YxN':
            self.model = MuLUTUnit('1x4', nf, upscale=upscale, dense=dense)
            self.K = 3
            self.S = upscale
        elif mode == 'Ex1':
            self.model = MuLUTUnit('2x2d3', nf, upscale=1, dense=dense)
            self.K = 4
            self.S = 1
        elif mode == 'ExN':
            self.model = MuLUTUnit('2x2d3', nf, upscale=upscale, dense=dense)
            self.K = 4
            self.S = upscale
        elif mode in ['Ox1', 'Hx1']:
            self.model = MuLUTUnit('1x4', nf, upscale=1, dense=dense)
            self.K = 4
            self.S = 1
        elif mode == ['OxN', 'HxN']:
            self.model = MuLUTUnit('1x4', nf, upscale=upscale, dense=dense)
            self.K = 4
            self.S = upscale
        else:
            raise AttributeError
        self.P = self.K - 1

    def forward(self, x):
        if 'TM' in self.mode:
            B, C, H, W = x.shape
            x = self.model(x)
            # print('down')
            return x
        elif 'C' in self.mode:
            B, C, H, W = x.shape
            
            x = self.model(x)
            # print('down')
            return x
        elif 'Q' in self.mode:
            B, C, H, W = x.shape
            
            x = self.model(x)
            # print('down')
            return x
        elif 'P' in self.mode:
            B, C, H, W = x.shape
            
            x = self.model(x)
            # print('down')
            return x
        elif 'M' in self.mode:
            B, C, H, W = x.shape
            x_rg = x[:, :2, :, :]
            x_gb = x[:, 1:, :, :]
            # x_rb = torch.stack((x[:, 0:1, :, :], x[:, 2:, :, :]),dim=1).squeeze(2)
            x_rb = torch.stack((x[:, 2:, :, :], x[:, 0:1, :, :]),dim=1).squeeze(2)
            processed_tensors = []

            for x, im in zip([x_rg, x_gb, x_rb], ['rg', 'gb', 'rb']):

                if 'rg' in im:
                    x = self.model_rg(x)   # B*C*L,K,K
                    x_rg_ = x
                elif 'gb' in im:
                    x = self.model_gb(x)   # B*C*L,K,K
                    x_gb_ = x
                else:
                    x = self.model_rb(x)   # B*C*L,K,K
                    x_rb_ = x
                processed_tensors.append(x)
            if x.is_cuda:
                device = x.device
            else:
                device = torch.device('cpu')
            combined_x = torch.cat(processed_tensors, dim=1).to(device)
            # print('down')
            return combined_x#, x_rg_, x_gb_, x_rb_
        else:
            B, C, H, W = x.shape
            x = F.unfold(x, self.K)  # B,C*K*K,L
            x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
            x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
            x = x.reshape(B * C * (H - self.P) * (W - self.P),
                        self.K, self.K)  # B*C*L,K,K
            x = x.unsqueeze(1)  # B*C*L,l,K,K

            if 'Y' in self.mode:
                x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                            x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

                x = x.unsqueeze(1).unsqueeze(1)
            elif 'V' in self.mode:
                # print(x.shape)
                x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1],
                            x[:, :, 1, 1]], dim=1)
                # print(x.shape)
                x = x.unsqueeze(1).unsqueeze(1)
            elif 'H' in self.mode:
                x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                            x[:, :, 2, 3], x[:, :, 3, 2]], dim=1)

                x = x.unsqueeze(1).unsqueeze(1)
            elif 'O' in self.mode:
                x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                            x[:, :, 1, 3], x[:, :, 3, 1]], dim=1)

                x = x.unsqueeze(1).unsqueeze(1)

            x = self.model(x)   # B*C*L,K,K
            x = x.squeeze(1)
            x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
            x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
            x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
            x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                    self.S, stride=self.S)
            return x


############### Grayscale Denoising, Deblocking, Color Image Denosing ###############
class DNNet(nn.Module):
    """ Wrapper of basic MuLUT block without upsampling. """

    def __init__(self, mode, nf=64, dense=True):
        super(DNNet, self).__init__()
        self.mode = mode

        self.S = 1
        if mode == 'Sx1':
            self.model = MuLUTUnit('2x2', nf, dense=dense)
            self.K = 2
        elif mode == 'Dx1':
            self.model = MuLUTUnit('2x2d', nf, dense=dense)
            self.K = 3
        elif mode == 'Yx1':
            self.model = MuLUTUnit('1x4', nf, dense=dense)
            self.K = 3
        else:
            raise AttributeError
        self.P = self.K - 1

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)   # B*C*L,K,K
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))     # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)
        return x


############### Image Demosaicking ###############
class DMNet(nn.Module):
    """ Wrapper of the first stage of MuLUT network for demosaicking. 4D(RGGB) bayer patter to (4*3)RGB"""

    def __init__(self, mode, nf=64, dense=False):
        super(DMNet, self).__init__()
        self.mode = mode

        if mode == 'SxN':
            self.model = MuLUTUnit('2x2', nf, upscale=2, out_c=3, dense=dense)
            self.K = 2
            self.C = 3
        else:
            raise AttributeError
        self.P = 0  # no need to add padding self.K - 1
        self.S = 2  # upscale=2, stride=2

    def forward(self, x):
        B, C, H, W = x.shape
        # bayer pattern, stride = 2
        x = F.unfold(x, self.K, stride=2)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H // 2) * (W // 2))  # stride = 2
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H // 2) * (W // 2),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        # print("in", torch.round(x[0, 0]*255))

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)  # B*C*L,out_C,S,S
        # self.C along with feat scale
        x = x.reshape(B, C, (H // 2) * (W // 2), -1)  # B,C,L,out_C*S*S
        x = x.permute((0, 1, 3, 2))  # B,C,outC*S*S,L
        x = x.reshape(B, -1, (H // 2) * (W // 2))  # B,C*out_C*S*S,L
        x = F.fold(x, ((H // 2) * self.S, (W // 2) * self.S),
                   self.S, stride=self.S)
        return x


if __name__ == '__main__':


    inp = torch.randn(1, 3, 500, 500)
    m = PoolNear()
    out = m(inp)
    print(out.shape)