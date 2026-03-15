#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 07:43:53 2020

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
from .projection_operators import backprojectionDDb_cuda

def FDK(proj, geo, filterType, cutoff, libFiles):
    
    if filterType == 'FBP':
        print("----------------\nStarting FBP... \n\n")
        proj = filterProj(proj, geo, cutoff)
    elif filterType == 'BP':
        print("----------------\nStarting BP... \n\n")
    else:
        raise ValueError('Unknown filter type.')
        
    vol = backprojectionDDb_cuda(proj, geo, -1, libFiles)
    return vol
    # return proj
    
    

def filterProj(proj, geo, cutoff):


    filteredProj = np.empty(proj.shape)
    
    us = np.linspace(geo.nu-1, 0, geo.nu) * geo.du
    vs = np.linspace(-(geo.nv-1)/2, (geo.nv-1)/2, geo.nv) * geo.dv
        
    # Detector Coordinate sytem in (mm)
    uCoord, vCoord = np.meshgrid(us, vs)
    
    # Compute weighted projections (Fessler Book Eq. (3.10.6))
    weightFunction = geo.DSO / np.sqrt(geo.DSO**2 + uCoord**2 + vCoord**2)
                       
    # Apply weighting function on each proj
    for i in range(geo.nProj):
        filteredProj[:,:,i] = proj[:,:,i] * weightFunction
    
    
    # Increase filter length to two times nv to decrease freq step
    h_Length = int(2**np.ceil(np.log2(np.abs(2*geo.nv)))) 
    
    # Builds ramp filter in space domain
    ramp_kernel = ramp_builder(h_Length)
    
    # Window filter in freq domain
    H_filter = filter_window(ramp_kernel, h_Length, cutoff)
    
    # Replicate to all colluns to build a 2D filter kernel
    H_filter = np.transpose(np.tile(H_filter, (geo.nu,1)))
    
    # Proj in freq domain
    H_proj = np.zeros([h_Length,geo.nu])
    
    # Filter each projection
    for i in range(geo.nProj):
         
       H_proj[0:geo.nv,:] = filteredProj[:,:,i]
    
       # Fourier transfor in projections
       H_proj = np.fft.fftshift(np.fft.fft(H_proj, axis=0, norm='ortho'))  
        
       # Multiplication in frequency = convolution in space
       H_proj = H_proj * H_filter
        
       # Inverse Fourier transfor
       H_proj = np.real(np.fft.ifft(np.fft.ifftshift(H_proj), axis=0, norm='ortho'))
    
       filteredProj[:,:,i] = H_proj[0:geo.nv,:]
       #filteredProj[:,:,i] = denoise_tv_chambolle(filteredProj[:, :, i], weight=0.1) # denoising TV

    return filteredProj  

## Function Ramp Filter
"""
The function builds Ramp Filter in space domain
Reference: Jiang Hsieh's book (second edition,page 73) Eq. 3.29
Reference: Fessler Book Eq.(3.4.14)
"""
def ramp_builder(h_Length):

    n = np.linspace(-h_Length/2, (h_Length/2)-1, h_Length)
    h = np.zeros(n.shape)
    h[int(h_Length/2)] = 1/4            # Eq. 3.29
    odd = np.mod(n,2) == 1              # Eq. 3.29
    h[odd] = -1 / (np.pi * n[odd])**2   # Eq. 3.29
    
    return h


## Function Window Ramp Filter
"""
The function builds Ramp Filter apodizided in frequency domain
Reference: Fessler Book and MIRT
"""
def filter_window(ramp_kernel, h_Length, cutoff):
    # Bring filter to frequency domain
    H_ramp = np.abs(np.fft.fftshift(np.fft.fft(ramp_kernel)))

    # Generate Hanning window with smooth cutoff
    w = np.round(h_Length * cutoff)  # Cutoff frequency
    n = np.linspace(-h_Length/2, (h_Length/2)-1, h_Length)
    H_window = np.zeros_like(n)
    smooth_region = np.abs(n) < w / 2
    H_window[smooth_region] = 0.5 * (1 + np.cos(2 * np.pi * n[smooth_region] / w))

    # Apply the window and taper the ramp filter
    H_filter = H_ramp * H_window
    return H_filter
