from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ssapp.data.AntennaDatasetLoaders import load_serialized_dataset, get_single_dataset_example
from torch.utils.data.dataloader import DataLoader



def plot_dataset_field_example(dataset,idx):
    params, fields = next(iter(DataLoader(dataset, batch_size=len(dataset))))

    phi_name = ['$\phi$ = 0\degree','\phi$ = 45\degree','\phi$ = 90\degree']

    mag_co = lambda a,i: 20*np.log10(a[i,:,:,0],a[i,:,:,1]) # Convert fields to dB power plots, copolar
    mag_cross = lambda a,i: 20*np.log10(a[i,:,:,2],a[i,:,:,3]) # -||-, crosspolar

    theta = np.linspace(-180,180,361) # Generate theta values for x-axis

    print(mag_co(fields,idx))
    # Plots
    plt.figure()
    plt.plot(theta,mag_co(fields,idx))
    plt.xlabel(r'\theta')
    plt.figure()
    plt.plot(theta,mag_cross(fields,idx))





def plot_dataset_timeseries_histogram(dataset):

    params, fields = next(iter(DataLoader(dataset,batch_size = len(dataset))))

    x = np.linspace(-180,180, 361)

    Y = fields[:,:,:,0].numpy().flatten()

    x = np.array([x for y in range(len(Y)*3)]).flatten()

    fig,axs = plt.subplots(ncols=2,nrows=1,figsize = (12,4))

    mag = lambda a,b: 20*np.log10(np.sqrt(a**2+b**2))

    titles = []

    for idx,ax in enumerate(axs.flatten()):
        Y = mag(fields[:,:,0,0+2*idx],fields[:,:,0,1+2*idx])

        cmap = copy(plt.cm.plasma)
        cmap.set_bad(cmap(0))
        h, xedges, yedges = np.histogram2d(x, Y,bins=[361,100])
        pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap,
                                norm=LogNorm(), rasterized=True)
        fig.colorbar(pcm, ax=ax, label="# points", pad=0)
        ax.set_title("2d histogram and log color scale")