from matplotlib import pyplot as plt
import torch
import numpy as np


def plotEncodingVerificaiton(model, train_field):

    EXTENT = [0,90,-180,180]
    ASPECT = 0.3
    LINEWIDTH = 3
    X_TEXT_DISPLACEMENT = -45
    thetas = np.linspace(-180,180,361)
    def setImgAxis(ax):
        ax.set_xlabel(r'$ \phi $')
        ax.set_ylabel(r'$ \theta $')
        ax.set_aspect(ASPECT) 

    fig,axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axs[0,0].imshow(torch.abs(train_field[0,:,:,1:2]),aspect = 1/100, interpolation = 'none',extent=EXTENT)
    axs[0,0].set_title('Truth')
    axs[0,0].text(X_TEXT_DISPLACEMENT,0,'Co-polar',rotation = 'vertical',fontsize = 13, va = 'center')
    setImgAxis(axs[0,0])

    axs[0,1].imshow(torch.abs(model(train_field.float())[0,:,:,1:2]).detach(), interpolation = 'none',extent=EXTENT)
    axs[0,1].set_title('Decoded')
    axs[0,1].axvline(x = 45, color = 'r', linestyle = '--')
    setImgAxis(axs[0,1])

    axs[0,2].plot(thetas,10*torch.log10(torch.abs(train_field[0,:,1,1:2])),label = 'Data',linewidth = LINEWIDTH)
    axs[0,2].plot(thetas,10*torch.log10(torch.abs(model(train_field.float())[0,:,1,1:2])).detach(),label = 'Model',linewidth = LINEWIDTH) 
    axs[0,2].grid()
    axs[0,2].set_title('Middle-Cut at $\phi = 45$')
    axs[0,2].set_xlabel(r'$\theta$')
    axs[0,2].legend()
    axs[0,2].set_ylabel('Copolar Amplitude [dB]')

    axs[1,0].imshow(torch.abs(train_field[0,:,:,3:4]),aspect = 1/100, interpolation = 'none',extent=EXTENT)
    axs[1,0].text(X_TEXT_DISPLACEMENT,0,'Cross-polar',rotation = 'vertical',fontsize = 13, va = 'center')
    setImgAxis(axs[1,0]) 
    

    axs[1,1].imshow(torch.abs(model(train_field.float())[0,:,:,3:4]).detach(),aspect = 1/100, interpolation = 'none',extent=EXTENT) 
    setImgAxis(axs[1,1])
    axs[1,1].axvline(x = 45, color = 'r', linestyle = '--')

    axs[1,2].plot(thetas,10*torch.log10(torch.abs(train_field[0,:,1,3:4])),linewidth = LINEWIDTH)
    axs[1,2].plot(thetas,10*torch.log10(torch.abs(model(train_field.float())[0,:,1,3:4])).detach(),linewidth = LINEWIDTH) 
    axs[1,2].set_xlabel(r'$\theta$')
    axs[1,2].grid()
    axs[1,2].set_ylabel('Crosspolar Amplitude [dB]')