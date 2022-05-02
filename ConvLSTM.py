import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, h_channels, kernel_size, device):
        '''
        Parameters
        -----------
        in_channels: int
            Number of channels of input tensor
        h_channels: int
            Number of channels of hidden state
        kernel size: (int, int)
            Size of convolutional kernel  
        '''
        super(ConvLSTMCell, self).__init__()

        self.h_channels = h_channels
        padding = kernel_size // 2, kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels + h_channels,
                              out_channels=4 * h_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=True)
        self.device = device
    
    def forward(self, input_data, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input_data, h_prev), dim=1)  # concatenate along channel axis

        combined_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_output, self.h_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, image_size):
        """ initialize the first hidden state as zeros """
        height, width = image_size
        return (torch.zeros(batch_size, self.h_channels, height, width, device=self.device),
                torch.zeros(batch_size, self.h_channels, height, width, device=self.device))

    
class ConvLSTM(nn.Module):
    
    def __init__(self, in_channels, h_channels, num_layers, kernel_size, device):
        '''
        Parameters
        -----------
        in_channels: int
            Number of channels of input tensor
        h_channels: int
            Number of channels of hidden state
        num_layers: int
            Number of LSTM layers stacked on each other
        kernel_size: (int, int)
            Size of kernel in convolutions
        '''
        
        super(ConvLSTM, self).__init__()
        
        self.in_channels = in_channels
        self.num_layer = num_layers
        
        layer_list = []
        for i in range(num_layers):
            cur_in_channels = in_channels if i == 0 else h_channels[i - 1]
            layer_list.append(ConvLSTMCell(in_channels=cur_in_channels,
                                           h_channels=h_channels[i],
                                           kernel_size=kernel_size,
                                           device=device))
            
        self.layer_list = nn.ModuleList(layer_list)
            
    def forward(self, x, states=None):
        # (t, b, c, h, w) -> (b, t, c, h, w)
        x = x.permute(1, 0, 2, 3, 4)
        b,_,_,h,w = x.size()
                    
        if states is None:
            hidden_states = self.init_hidden(batch_size=b, image_size=(h,w))
        else:
            hidden_states, cell_states = states
        
        layer_output_list = []
        last_state_list = []
        cur_layer_input = x
        for i, layer in enumerate(self.layer_list):
            # [TODO: layer forward]
            h = hidden_states[i]
            c = cell_states[i]

            output_inner = []
            for t in range(x.size(1)):
                h,c = layer(input_data=cur_layer_input[:,t,:,:,:], prev_state = [h,c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h,c])

        return layer_output_list[-1], last_state_list[-1]   
        # return hidden_states, (hidden_states, cell_states)
    
    def init_hidden(self, batch_size, image_size):
        # [TODO: initialze hidden and cell states]
        hidden_states = []
        for i in range(self.num_layer):
            hidden_states.append(self.layer_list[i].init_hidden(batch_size, image_size))
        return hidden_states
        
    
def activation_factory(name):
    """
    Returns the activation layer corresponding to the input activation name.
    Parameters
    ----------
    name : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function after the
        convolution.
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    if name is None or name == "identity":
        return nn.Identity()

    raise ValueError(f'Activation function `{name}` not yet implemented')

    
def make_conv_block(conv):

    out_channels = conv.out_channels
    modules = [conv]
    modules.append(nn.GroupNorm(16, out_channels))
    modules.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*modules)


class DCGAN64Encoder(nn.Module):
    
    def __init__(self, in_c, out_c):

        super(DCGAN64Encoder, self).__init__()
        
        h_c = out_c // 2 
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(in_c, h_c, 3, 2, 1)),
            make_conv_block(nn.Conv2d(h_c, h_c, 3, 1, 1)),
            make_conv_block(nn.Conv2d(h_c, out_c, 3, 2, 1))])
        
    def forward(self, x):
        out = x
        for layer in self.conv:
            out = layer(out)
        return out
        

class DCGAN64Decoder(nn.Module):
    
    def __init__(self, in_c, out_c, last_activation=None):
        
        super(DCGAN64Decoder, self).__init__()
        
        h_c = in_c // 2
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(in_c, h_c, 3, 2, 1, output_padding=1)),
            make_conv_block(nn.ConvTranspose2d(h_c, h_c, 3, 1, 1)),
            make_conv_block(nn.ConvTranspose2d(h_c, out_c, 3, 2, 1, output_padding=1))])

        self.last_activation = activation_factory(last_activation)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.conv):
            out = layer(out)
        return self.last_activation(out)    
    
    
class Seq2Seq(nn.Module):
    
    def __init__(self, args):
        
        super(Seq2Seq, self).__init__()
                
        self.h_channels = args.h_channels
        self.num_layers = args.num_layers
        self.kernel_size = args.kernel_size
                
        self.seq_len = args.seq_len
        self.horizon = args.horizon
        
        self.frame_encoder = DCGAN64Encoder(self.in_channels, self.h_channels).to(self.gpu) 
        self.frame_decoder = DCGAN64Decoder(self.h_channels, self.out_channels).to(self.gpu)

        self.model = ConvLSTM(in_channels=self.h_channels, 
                              h_channels=[self.h_channels] * self.num_layers, 
                              num_layers=self.num_layers, 
                              kernel_size=self.kernel_size,
                              device=self.gpu)
                    
    def forward(self, in_seq, out_seq, teacher_forcing_rate=None):
         
        next_frames = []
        hidden_states, states = None, None
        
        # encoder
        for t in range(self.seq_len - 1):
            # pass
            # [TODO: call ConvLSTM]
            h_t, c_t = self.frame_encoder()
        
        # decoder
        for t in range(self.horizon):
            
            if teacher_forcing_rate is None:
                pass
                # [TODO: use predicted frames as the input]
            
            
            else:
                pass
                # [TODO: choose from predicted frames and out_seq as the input
                # [      based on teacher forcing rate]
                
            # [TODO: call ConvLSTM]
            pass
            
            out = self.frame_decoder(hidden_states[-1])
            next_frames.append(out)
        
        next_frames = torch.stack(next_frames, dim=1) 
        return next_frames