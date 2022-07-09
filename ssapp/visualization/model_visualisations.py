from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plotModelPrediction(pred,field, idx = 0, phi_cuts = [0,1,2], title = None,include_phase = False,ylim = None):



    assert idx < len(pred),"Index must be less than batch_size"

    if pred.shape[1:4] != (361,3,4):
        pred = pred.reshape(-1,361,3,4)
    if field.shape[1:4] != (361,3,4):
        field = field.reshape(-1,361,3,4)

    theta = np.linspace(-180,180,361)
    fig,axs = plt.subplots(ncols=2,nrows=1,figsize = (10,3.2))

    # Convert and detach from autograd if pred or field is tensor
    if type(pred) == type(torch.tensor([])): 
        pred = pred.detach().numpy()
    if type(field) == type(torch.tensor([])):
        field = field.detach().numpy()


    mag_co = lambda a,i,pc: 20*np.log10(np.sqrt(a[i,:,pc,0]**2+a[i,:,pc,1]**2)) # Convert fields to dB power plots, copolar
    mag_cross = lambda a,i,pc: 20*np.log10(np.sqrt(a[i,:,pc,2]**2+a[i,:,pc,3]**2)) # -||-, crosspolar
    phi_labels = ['$\phi = 0$','$\phi = 45\degree$','$\phi = 90\degree$']
    sns.set_theme()
    plt.style.use('default')

    for phi_cut in phi_cuts:

        

        axs[0].plot(theta,mag_co(field,idx,phi_cut).T,
                    color='C'+str(phi_cut*2),
                    label = 'Truth at '+phi_labels[phi_cut]
                    ,linewidth = 2)

        axs[0].plot(theta,mag_co(pred,idx,phi_cut).T,
                    color='C'+str(phi_cut*2+1),
                    linestyle = '--',
                    linewidth = 2)
        axs[0].set_ylabel('$|E_{co}|$ dB')
        axs[0].set_xlabel(r'$\theta\degree$')
        #axs[0].set_title('$E_{co}$')
        axs[0].grid(True)
        axs[0].set_xlim([-180,180])
        axs[0].set_xticks([-180,-90,0,90,180])
        if ylim is not None:
            axs[0].set_ylim(ylim)
        

    for phi_cut in phi_cuts:
        axs[1].plot(theta,mag_cross(field,idx,phi_cut).T,
                    color='C'+str(phi_cut*2),
                    linewidth = 2)
        
        axs[1].plot(theta,mag_cross(pred,idx,phi_cut).T,
                    label = 'Prediction at '+phi_labels[phi_cut],
                    color='C'+str(phi_cut*2+1),
                    linestyle = '--',
                    linewidth = 2)



                    
        axs[1].set_ylabel('$|E_{cross}|$ dB')
        axs[1].set_xlabel(r'$\theta\degree$')
        axs[1].grid()
        #axs[1].set_title('$E_{cross}$')
        axs[1].grid(True)
        axs[1].set_xlim([-180,180])
        axs[1].set_xticks([-180,-90,0,90,180])
        if ylim is not None:
            axs[1].set_ylim(ylim)
            

    
    

   
    fig.legend(ncol=2,loc = 'lower center',bbox_to_anchor = (0.51,-0.21))
    # Set figure title
    if title is not None:
        fig.suptitle(title)



    pass