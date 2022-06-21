from matplotlib import pyplot as plt
import numpy as np


def plotModelPrediction(pred,field, idx = 0, phi_cuts = [0,1,2], title = None,include_phase = False):

    assert pred.shape == field.shape
    assert pred.shape[1:3] == (361,3,4),"Prediction shape must be (batch_size,361,3,4)"
    assert idx < len(pred),"Index must be less than batch_size"

    fig,axs = plt.subplots(ncols=2,nrows=1,figsize = (12,4))

    mag_co = lambda a,i: 20*np.log10(np.sqrt(a[i,:,:,0]**2+a[i,:,:,1]**2)) # Convert fields to dB power plots, copolar
    mag_cross = lambda a,i: 20*np.log10(np.sqrt(a[i,:,:,2]**2+a[i,:,:,3]**2)) # -||-, crosspolar

    axs[0].plot(pred[idx,:,phi_cuts,0],pred[idx,:,phi_cuts,1],label = 'Prediction')
    axs[0].plot(field[idx,:,phi_cuts,0],field[idx,:,phi_cuts,1],label = 'Truth')
    
    axs[1].plot(pred[idx,:,phi_cuts,0],pred[idx,:,phi_cuts,1],label = 'Prediction')
    axs[1].plot(field[idx,:,phi_cuts,0],field[idx,:,phi_cuts,1],label = 'Truth')



    pass