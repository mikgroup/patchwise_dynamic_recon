import numpy as np
import sigpy as sp
import sigpy.plot as pl
import cupy as cp
import scipy.io
import math
from scipy.sparse import csc_matrix
import numpy.matlib
import matplotlib.pyplot as plt

def M_forward(M,c,gpu = False):
    if gpu:
        xp = cp
    else:
        xp = np
    return xp.matmul(M,c)

def C_forward(C,m,gpu = False):
    if gpu:
        xp = cp
    else:
        xp = np
    return xp.matmul(m,C)

def M_adjoint(M,x,gpu = False):
    if gpu:
        xp = cp
    else:
        xp = np
    return xp.matmul(M.T.conj(),x)

def C_adjoint(C,x,gpu = False):
    if gpu:
        xp = cp
    else:
        xp = np
    return xp.matmul(x,C.T.conj())



def soft_thresh_complex(x, l,gpu = False):
    if gpu:
        xp = cp
    else:
        xp = np
    return xp.sign(abs(x)) * xp.maximum(xp.abs(x) - l, 0.)*xp.exp(1j*xp.angle(x))

def R_forward(im,patch_no,patch_size,stride_length):
    [frames,n,m]=im.shape
    n_patch_per_side=math.floor((n-patch_size)/stride_length)+1
    row_patch_no=math.floor((patch_no-1)/n_patch_per_side)
    column_patch_no=(patch_no-1)%n_patch_per_side
    crop=im[:,int(row_patch_no*stride_length):int(row_patch_no*stride_length+patch_size),int(column_patch_no*stride_length):int(column_patch_no*stride_length+patch_size)]
    crop=crop.reshape((frames*patch_size*patch_size,1))
    return crop
def R_adjoint(crop,patch_no,im_size,im_frames,stride_length):
    [a,b]=crop.shape
    patch_size=int((a/im_frames)**(1/2))
#     print(patch_size)
    crop=crop.reshape((im_frames,patch_size,patch_size))
    n_patch_per_side=math.floor((im_size-patch_size)/stride_length)+1
    row_patch_no=math.floor((patch_no-1)/n_patch_per_side)
    column_patch_no=(patch_no-1)%n_patch_per_side
    padded=cp.zeros((im_frames,im_size,im_size),dtype=cp.complex)
    padded[:,int(row_patch_no*stride_length):int(row_patch_no*stride_length+patch_size),int(column_patch_no*stride_length):int(column_patch_no*stride_length+patch_size)]=crop
    return padded


def powermethods(matrix_A,iterations = 10):
    max_eig=0
    for i in range(iterations):
        c = cp.random.rand(32*32,1)
        frwrd=M_forward(matrix_A,c,gpu = True)
        mag_c=cp.dot(cp.transpose(c),c)
        eig=cp.dot(cp.transpose(frwrd),frwrd)/mag_c
        eig = eig[0][0]
#         print(eig)
        max_eig=max(eig,max_eig)
#         print(max_eig)
        
    return float(abs(max_eig))

def modelCforward(M,C):
    # C shape: 1,256,256
    # M shape: 225,24,1032,1032
    patchnum = 25
    result=cp.zeros((24,96,96),dtype=cp.complex)
    for i in range(patchnum):
        M_current=cp.array(M[i,:,:,:].reshape(24*1024,1024))
        patch=R_forward(C,patch_no=i+1,stride_length=16,patch_size=32)
        res = M_forward(M=M_current,c=patch,gpu=True)
#         print(R_adjoint(crop=res,im_frames=24,im_size=256,patch_no=i+1,stride_length=16) )
        result+=R_adjoint(crop=res,im_frames=24,im_size=96,patch_no=i+1,stride_length=16) 
    return result
def modelCadjoint(Im,M):
    # Im shape: (24,256,256)
    # M shape: 225,24,1032,1032
    patch_c = cp.zeros((225,96,96),dtype=cp.complex)
    for i in range(25):
        patch = R_forward(Im,patch_no=i+1,patch_size=32,stride_length=16)
        M_patch = cp.array(M[i,:,:,:].reshape(24*1024,1024))
        c_est = M_adjoint(M_patch,patch,gpu=True)
        radj = R_adjoint(c_est,im_frames=1,patch_no=i+1,stride_length=16,im_size=96)  
        patch_c[i,:,:] = radj.squeeze()
    return patch_c.sum(axis=0)[None,:,:]
def modelMadjoint(Im,C):
    # C shape: (1,256,256)
    # Im shape: (24,256,256)
    patch_m = np.zeros((25,24,1024,1024),dtype=cp.complex)
    for i in range(25):
        patch = R_forward(Im,patch_no=i+1,patch_size=32,stride_length=16)
        C_patch = R_forward(C,patch_no=i+1,patch_size=32,stride_length=16)
        cadj = C_adjoint(C_patch,patch,gpu=True).reshape(24,1024,1024)
        patch_m[i,:,:,:] = cp.asnumpy(cadj)
    return patch_m


def sensforward(LPS_image,sensmaps):
    # LPS (256*24,256)
    # sensmaps (256,256,12)
    LPS = LPS_image.reshape(24,96,96).transpose(1,2,0)
    return LPS[:,:,:,None]*sensmaps[:,:,None,:]
def fftforward(sensmaps):
    # sensmaps (256,256,24,12)
    return sp.fft(sensmaps,axes=(0,1))
def maskforward(fftout,mask):
    # fftout (256,256,24,12)
    # mask (256,256,24,12)
    return fftout*mask
def forwardmodel(LPS_image,sensmaps,mask):
    return maskforward(fftforward(sensforward(LPS_image,sensmaps)),mask)
def maskadjoint(kspace,mask):
    return kspace*mask
def fftadjoint(maskout):
    return sp.ifft(maskout,axes=(0,1))
def sensadjoint(fftout,sensmaps):
    ssp = sensmaps[:,:,None,:].conj()
    return cp.sum(fftout*ssp,axis=3)
def adjointmodel(kspace,sensmaps,mask):
    return sensadjoint(fftadjoint(maskadjoint(kspace,mask)),sensmaps)


def FourierCforward(M,C,sensmaps,mask):
    re_image = modelCforward(M,C)
    return forwardmodel(re_image,sensmaps,mask)
def FourierCadjoint(kspace,sensmaps,mask,M):
    ims = adjointmodel(kspace,sensmaps,mask).transpose(2,0,1)
#     print(ims.shape)
    return modelCadjoint(ims,M)
def FourierMadjoint(kspace,sensmaps,mask,C):
    ims = adjointmodel(kspace,sensmaps,mask).transpose(2,0,1)
    return modelMadjoint(ims,C)