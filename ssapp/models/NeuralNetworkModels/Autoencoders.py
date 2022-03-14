import torch.nn as nn


class PatchAntenna1ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(PatchAntenna1ConvAutoEncoder, self).__init__()


        self.conv_encoder1 = nn.Conv2d(in_channels=4,
                                    out_channels=8,
                                    kernel_size=3,
                                    padding = (2,2),
                                    stride=2)
        self.conv_encoder2 = nn.Conv2d(in_channels=8,
                                    out_channels=16,
                                    kernel_size=(3,1),
                                    stride=2,
                                    padding = (2,0))

        self.linear_to_latent = nn.Linear(in_features=16*91*3,
                                        out_features= 10)
        
        self.latent_to_linear = nn.Linear(in_features=10,
                                        out_features= 16*91*3)

        self.conv_decoder1 =  nn.ConvTranspose2d(in_channels=16,
                                        out_channels=8,
                                        kernel_size=(3,1),
                                        stride=2,

                                        padding = (0,0))






        self.conv_decoder2 =  nn.ConvTranspose2d(in_channels=8,
                                        out_channels=4,
                                        kernel_size=3,
                                        padding=2,
                                        stride=2)


    def encode(self,x):
        x = x.reshape(-1,4,3,361) # [Batch_size, Channels, Height, Width]
        x = self.conv_encoder1(x)
        print('Encoder conv1Shape',x.shape)
        x = self.conv_encoder2(x)
        print('Encoder Conv2Shape',x.shape)
        x = x.flatten()
        latent_space = self.linear_to_latent(x)

        return latent_space

    def decode(self,y):
        y = self.latent_to_linear(y)
        y = y.reshape(1,16,3,91)
        y = self.conv_decoder1(y)
        return y


    def forward(self, x):
        self.latent_space = self.encode(x)
        output = self.latent_space
        return output

