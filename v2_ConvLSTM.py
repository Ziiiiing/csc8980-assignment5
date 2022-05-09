import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, h_channels, kernel_size, device):
        super(ConvLSTMCell, self).__init__()

        self.h_channels = h_channels
        padding = kernel_size // 2, kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels + h_channels,
                              out_channels=4 * h_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=True,
                              device = device)
        self.device = device
    
    def forward(self, input_data, prev_state):
        h_prev, c_prev = prev_state 
        
        combined = torch.cat((input_data, h_prev), dim=1)  # concatenate along channel axis  
        # combined is cuda

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
    
    def __init__(
        self, 
        in_channels, 
        h_channels, 
        num_layers,
        kernel_size, 
        device):
        
        super(ConvLSTM, self).__init__()
        
        self.in_channels = in_channels
        self.num_layer = num_layers
        
        layer_list = []
        for i in range(num_layers):
            cur_in_channels = in_channels if i == 0 else h_channels[i - 1]
            layer_list.append(ConvLSTMCell(in_channels=cur_in_channels,    # ???
                                           h_channels=h_channels[i],       # ???
                                           kernel_size=kernel_size,
                                           device=device))
            
        self.layer_list = nn.ModuleList(layer_list)
            
    def forward(self, x, states=None):

        # input tensor(single frame) is [b,c,h,w] 
        batchsize,_,height,width = x.size() 
                    
        if states is None:
            hidden_states, cell_states = self.init_hidden(batch_size=batchsize, image_size=(height, width))  
        else:
            hidden_states, cell_states = states
        
        current_input = x
        next_hidden_states = []   # for all convlstm layers
        next_cell_states = []     # for all convlstm layers
        
        for i, layer in enumerate(self.layer_list):
            # [TODO: layer forward] 

            # update hidden state & cell state for this layer
            hidden_state_this_layer = hidden_states[i]
            cell_state_this_layer = cell_states[i]
            state_this_layer = (hidden_state_this_layer, cell_state_this_layer)
            hidden_state_this_layer, cell_state_this_layer = layer(input_data=current_input, prev_state=state_this_layer)   


            # store updated hidden state and cell state for all layers
            next_hidden_states.append(hidden_state_this_layer)
            next_cell_states.append(cell_state_this_layer)

            # update the input for next forward
            # rest layers take the hidden state of the previous layer as input
            current_input = hidden_state_this_layer   # torch.Size([16, 64, 16, 16]
            
            

        # hidden_states -> list of hidden state for each conv layer, 
        # (hidden_states, cell_states) -> tuple of list of state(h,c) for each layer       
        return next_hidden_states, (next_hidden_states, next_cell_states)  

        # return last convlstm layer only
        # return next_hidden_states[-1], (next_hidden_states[-1], next_cell_states[-1])
        
    
    def init_hidden(self, batch_size, image_size):
        # [TODO: initialze hidden and cell states]   
        # initialize all layers' hidden states and cell states as zeros

        hidden_states = []
        cell_states = []
        for layer in self.layer_list:
            hidden_state_cur_layer, cell_state_cur_layer = layer.init_hidden(batch_size, image_size)
            hidden_states.append(hidden_state_cur_layer)
            cell_states.append(cell_state_cur_layer)

        return hidden_states, cell_states




        
    
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
            nn.ConvTranspose2d(h_c, out_c, 3, 2, 1, output_padding=1)])

        self.last_activation = activation_factory(last_activation)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.conv):
            out = layer(out)
        return self.last_activation(out)    
    
    
class Seq2Seq(nn.Module):
    
    def __init__(self, h_channels, num_layers, kernel_size, seq_len, horizon):
        
        super(Seq2Seq, self).__init__()
                
        self.h_channels = h_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
                
        self.seq_len = seq_len
        self.horizon = horizon
        
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

        # in_seq is a 5d tensor like [b,seq_len,c,h,w]
        # change this into [seq_len, b,c,h,w]
        in_seq = in_seq.permute(1,0,2,3,4)

        # prepare the ground truth for teacher forcing
        out_seq = out_seq.permute(1,0,2,3,4)  # [t,b,c,h,w]
        
        # encoder
        for t in range(self.seq_len): 
            # [TODO: call ConvLSTM]

            in_frame = in_seq[t]
            # perform frame encoder to increase channel but decrease frame size (for faster training)
            in_frame = self.frame_encoder(in_frame)   # 4d tensor [b,c,h,w]
      
            # call ConvLSTM
            hidden_states,states = self.model(in_frame, states)

        # decoder
        for t in range(self.horizon):
            
            if teacher_forcing_rate is None:
                # [TODO: use predicted frames as the input]
                input = hidden_states[-1]
            else:
                # [TODO: choose from predicted frames and out_seq as the input
                # [ based on teacher forcing rate]   
                target_frame = out_seq[t-1] if t >0 else hidden_states[-1]
                target_frame = self.frame_encoder(target_frame)
                input = target_frame

            # [TODO: call ConvLSTM]
            hidden_states, states = self.model(input, states) 

            out = self.frame_decoder(hidden_states[-1])
            next_frames.append(out)      
        
        next_frames = torch.stack(next_frames, dim=1) 
        return next_frames