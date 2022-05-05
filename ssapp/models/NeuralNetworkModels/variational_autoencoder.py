import torch.nn as nn
import torch



class VAE(nn.Module):
    """Based slightly on https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b"""
    def __init__(self, config = {'latent_size': 1,
                                'coder_channel_1': 16,
                                'coder_channel_2': 32,
                                'batch_size' : 4}):
        super(VAE, self).__init__()
        self.config = config
        Latent_size = self.config['latent_size']
        coder_channel_1 = self.config['coder_channel_1']
        coder_channel_2 = self.config['coder_channel_2']
        self.batch_size = self.config['batch_size']

        KERNEL_SIZE_1 = (3,11)
        KERNEL_SIZE_2 = (3,11)

        STRIDE_1 = (1,2)
        STRIDE_2 = (2)

        PADDING_1 = (1,10)
        PADDING_2 = (0,0)
        

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


        self.fc_mu = nn.Linear(in_features=coder_channel_2*88,
                                        out_features= Latent_size)

        self.fc_std = nn.Linear(in_features=coder_channel_2*88,
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

        self.activation = nn.ReLU()

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc # hack to get sampling on the GPU
        self.N.scale = self.N.scale
        self.kl = 0


    def var_encode(self,x):
        self.batch_size = len(x) # To do correct reshapes for smaller batches
        x = x.reshape(-1,4,3,361) # [Batch_size, Channels, Height, Width]
        x = self.conv_encoder1(x)
        x = self.activation(x)
        x = self.conv_encoder2(x)
        x = self.activation(x)
        x = x.reshape(self.batch_size,-1)
        #Predict Mean and Std of encoding
        mu =  self.fc_mu(x)
        sigma = torch.exp(self.fc_std(x))

        return mu,sigma

    def var_encode_sample(self,x):
        """Same as var_encode, but return a sample from the distribution instead of mu and sigma"""
        mu,sigma = self.var_encode(x)
        return mu + sigma*self.N.sample(mu.shape)

    def decode(self,z):
        z = self.latent_to_linear(z)
        z = self.activation(z)
        z = z.reshape(-1,self.config['coder_channel_2'],1,88)
        z = self.conv_decoder1(z)
        z = self.activation(z)
        z = self.conv_decoder2(z).reshape(-1,361,3,4)
        return z


    def forward(self, x):
        #Encode
        mu,sigma = self.var_encode(x)
        #Sample latent space
        z = mu + sigma*self.N.sample(mu.shape)
        #Reconstruct from latent space sample
        reconstruction = self.decode(z)
        # Calculate Kullbach Liebler Divergence
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() 

        return reconstruction


