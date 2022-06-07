import torch.nn as nn
import torch

class PatchAntenna1ConvAutoEncoder(nn.Module):
    def __init__(self, config = {'latent_size': 20,
                                'coder_channel_1': 8,
                                'coder_channel_2': 16}):
        super(PatchAntenna1ConvAutoEncoder, self).__init__()
        self.config = config
        Latent_size = self.config['latent_size']
        coder_channel_1 = self.config['coder_channel_1']
        coder_channel_2 = self.config['coder_channel_2']


        self.conv_encoder1 = nn.Conv2d(in_channels=4,
                                    out_channels=coder_channel_1,
                                    kernel_size=3,
                                    padding = 2,
                                    stride=2)
        self.conv_encoder2 = nn.Conv2d(in_channels=coder_channel_1,
                                    out_channels=coder_channel_2,
                                    kernel_size=3,
                                    stride=2)


        self.linear_to_latent = nn.Linear(in_features=coder_channel_2*90,
                                        out_features= Latent_size)
        
        self.latent_to_linear = nn.Linear(in_features=Latent_size,
                                        out_features= coder_channel_2*90)

        self.conv_decoder1 =  nn.ConvTranspose2d(in_channels=coder_channel_2,
                                        out_channels=coder_channel_1,
                                        kernel_size=3,
                                        stride=2,
                                        padding = (0,0),
                                        output_padding=(0,1))


        self.conv_decoder2 =  nn.ConvTranspose2d(in_channels=coder_channel_1,
                                        out_channels=4,
                                        kernel_size=3,
                                        padding=2,
                                        stride=2)

        self.activation = nn.LeakyReLU()


    def encode(self,x):
        x = x.reshape(-1,4,3,361) # [Batch_size, Channels, Height, Width]
        x = self.conv_encoder1(x)
        x = self.activation(x)
        x = self.conv_encoder2(x)
        x = self.activation(x)
        x = x.flatten()
        latent_space = self.linear_to_latent(x)

        return latent_space

    def decode(self,y):
        y = self.latent_to_linear(y)
        y = self.activation(y)
        y = y.reshape(1,self.config['coder_channel_2'],1,90)
        y = self.conv_decoder1(y)
        y = self.activation(y)
        y = self.conv_decoder2(y).reshape(-1,361,3,4)
        return y


    def forward(self, x):

        self.latent_space = self.encode(x)
        output = self.decode(self.latent_space)

        return output



class ConvAutoEncoderAndLatentRegressor(nn.Module):
    def __init__(self, config = {'latent_size': 20,
                                'coder_channel_1': 16,
                                'coder_channel_2': 32,
                                'Parameter Number': 3}):
        super(ConvAutoEncoderAndLatentRegressor, self).__init__()
        self.config = config
        Latent_size = self.config['latent_size']
        coder_channel_1 = self.config['coder_channel_1']
        coder_channel_2 = self.config['coder_channel_2']
        num_params = self.config['Parameter Number']

        KERNEL_SIZE_1 = (3,11)
        KERNEL_SIZE_2 = (3,11)

        STRIDE_1 = (1,2)
        STRIDE_2 = (2)

        PADDING_1 = (1,10)
        PADDING_2 = (0,0)

        self.batch_size = 4


        self.batch_norm_1 = nn.BatchNorm2d(4)
        self.batch_norm_2 = nn.BatchNorm2d(coder_channel_1)
        self.batch_norm_3 = nn.BatchNorm2d(coder_channel_2)
        

        self.conv_encoder1 = nn.Conv2d(in_channels=4,
                                    out_channels=coder_channel_1,
                                    kernel_size=KERNEL_SIZE_1,
                                    padding = PADDING_1,
                                    stride=STRIDE_1)
        self.conv_encoder2 = nn.Conv2d(in_channels=coder_channel_1,
                                    out_channels=coder_channel_2,
                                    kernel_size=KERNEL_SIZE_2,
                                    padding = PADDING_2,
                                    stride=STRIDE_2)


        self.linear_to_latent = nn.Linear(in_features=coder_channel_2*88,
                                        out_features= Latent_size)


        
        self.latent_to_linear = nn.Linear(in_features=Latent_size,
                                        out_features= coder_channel_2*88)

        self.conv_decoder1 =  nn.ConvTranspose2d(in_channels=coder_channel_2,
                                        out_channels=coder_channel_1,
                                        kernel_size=KERNEL_SIZE_2,
                                        stride=STRIDE_2,
                                        padding = PADDING_2,
                                        output_padding=(0,1))


        self.conv_decoder2 =  nn.ConvTranspose2d(in_channels=coder_channel_1,
                                        out_channels=4,
                                        kernel_size=KERNEL_SIZE_1,
                                        padding=PADDING_1,
                                        stride=STRIDE_1)

        self.activation = nn.LeakyReLU()

        self.param_to_latent_regressor = nn.Sequential(
                                                        nn.Linear(num_params,50),
                                                        nn.ReLU(),
                                                        nn.Linear(50,200),
                                                        nn.ReLU(),
                                                        nn.Linear(200,200),
                                                        nn.ReLU(),
                                                        nn.Linear(200,50),
                                                        nn.ReLU(),
                                                        nn.Linear(50,Latent_size))


    def encode(self,x):
        self.batch_size = len(x)
        x = x.reshape(self.batch_size,4,3,361) # [Batch_size, Channels, Height, Width]
        x = self.conv_encoder1(x)
        x = self.activation(x)
        x = self.conv_encoder2(x)
        x = self.activation(x)
        x = x.reshape(len(x),-1)
        latent_space = self.linear_to_latent(x)
        
        return latent_space

    def decode(self,y):
        y = self.latent_to_linear(y)
        y = self.activation(y)
        y = y.reshape(self.batch_size,self.config['coder_channel_2'],1,88)
        y = self.conv_decoder1(y)
        y = self.activation(y)
        y = self.conv_decoder2(y).reshape(-1,361,3,4)
        return y

    def forward(self,params):
        self.batch_size = len(params)
        latent_guess = self.param_to_latent_regressor(params)

        return self.decode(latent_guess)

    def autoencode_train(self, x):

        self.latent_space = self.encode(x)
        output = self.decode(self.latent_space)

        return output
            


class AutoencoderFullyConnected(nn.Module):
    def __init__(self,latent_size = 3):
        super(AutoencoderFullyConnected, self).__init__()


        self.encode_lin1 = nn.Linear(in_features = 4*3*361,
                                    out_features = 1000)

        self.encode_lin2 = nn.Linear(in_features=1000,
                                    out_features= 200)

        self.encode_lin3 = nn.Linear(in_features=200,
                                    out_features= latent_size)

        self.decode_lin1 =nn.Linear(in_features=latent_size,
                                    out_features= 200)

        self.decode_lin2 =nn.Linear(in_features=200,
                                    out_features= 1000)

        self.decode_lin3 =nn.Linear(in_features=1000,
                                    out_features= 4*3*361)


        self.activation = nn.LeakyReLU()


    def encode(self,x):
        batch_size = len(x)
        x = x.reshape(batch_size,4*3*361) # 
        x = self.encode_lin1(x)
        x = self.activation(x)
        x = self.encode_lin2(x)
        x = self.activation(x)
        latent_space = self.encode_lin3(x)
        
        return latent_space

    def decode(self,y):

        y = self.decode_lin1(y)
        y = self.activation(y)
        y = self.decode_lin2(y)
        y = self.activation(y)
        y = self.decode_lin3(y).reshape(-1,361,3,4)
        return y


    def forward(self, x):

        self.latent_space = self.encode(x)
        reconstruction = self.decode(self.latent_space)

        return reconstruction


class AutoencoderFCResNet(nn.Module):
    def __init__(self,latent_size = 3):
        super(AutoencoderFCResNet, self).__init__()


        self.encode_lin1 = nn.Linear(in_features = 4*3*361,
                                    out_features = 1000)

        self.encode_lin2 = nn.Linear(in_features=1000,
                                    out_features= 200)

        self.encode_lin3 = nn.Linear(in_features=200,
                                    out_features= latent_size)

        self.decode_lin1 =nn.Linear(in_features=latent_size,
                                    out_features= 200)

        self.decode_lin2 =nn.Linear(in_features=200,
                                    out_features= 1000)

        self.decode_lin3 =nn.Linear(in_features=1000,
                                    out_features= 4*3*361)

        self.res_encode = nn.Linear(in_features=4*3*361,
                                    out_features=200)

        self.res_decode = nn.Linear(in_features= latent_size,
                                    out_features=1000)


        self.activation = nn.LeakyReLU()


    def encode(self,x):
        batch_size = len(x)
        x = x.reshape(batch_size,4*3*361) # 
        res = x
        x = self.encode_lin1(x)
        x = self.activation(x)
        x = self.encode_lin2(x)
        x = x+self.res_encode(res)
        x = self.activation(x)
        latent_space = self.encode_lin3(x)
        
        return latent_space

    def decode(self,y):

        res = y
        y = self.decode_lin1(y)
        y = self.activation(y)
        y = self.decode_lin2(y)

        y = y+self.res_decode(res)
        y = self.activation(y)
        y = self.decode_lin3(y).reshape(-1,361,3,4)
        return y


    def forward(self, x):

        self.latent_space = self.encode(x)
        reconstruction = self.decode(self.latent_space)

        return reconstruction