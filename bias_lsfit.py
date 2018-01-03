import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
import sys,os
import math
from numpy import genfromtxt
from astropy.io import ascii
#from readfilters import read_filters
#from scipy.interpolate import splev, splrep
from bias_theor import growthfactor, continuum_func, physical_func
iarg = 0
bands = sys.argv[iarg+1]
db = sys.argv[iarg+2]
cso = sys.argv[iarg+3]
#snapnum = sys.argv[iarg+4]

if bands[:1] == 'V':
    #snapnum = [int(snapnum)]
    #print snapnum
    #snapnum = [131]
    snapnum = [199,156,131,113]
else:
    #snapnum = [int(snapnum)]
    snapnum = [199,156,131,113]
lbands = [bands]
dx = db
dim = 5
hl = 10
sl = 8

#cso = ['centra','sat_cen','orp_sat_cen']
#cso = cso[1]
print lbands, dx
indirs = '/Users/hengxingpan/Work/danail/ProCorr/Bias/output'
#snapnum = [156]

#lbands = ['mstars_tot','mcold','mstardot','L_tot_Halpha','mag_V_o_tot','mag_GALEX-FUV_r_tot','mag_GALEX-NUV_r_tot','mag_SDSS-u_r_tot','mag_SDSS-g_r_tot','mag_SDSS-r_r_tot','mag_SDSS-i_r_tot','mag_UKIRT-H_r_tot','mag_UKIRT-J_r_tot','mag_UKIRT-K_r_tot']
#xlow  = [8,1,32,-24,-24,-24,-24,-24,-24,-24,-24,-24]
#xupp  = [12,12,44,-14,-14,-14,-14,-14,-14,-14,-14,-14]

#dx = 0.5
#bins = 3
#lbands = lbands[bins:]
#lbands = lbands[bins:bins+1]

#p0 = 1.13398121, -22.39229749 ,  0.63451676 , -0.43
#p0 = 0.006, 1.019, -22.332 ,  0.624 , -0.409
#p0 = 1.01845884, -22.20278805 ,  0.64862178 , -0.35700368

#def growthfactor(z):
#    gf = np.full(len(z),1.0)
#    gf[np.where(z==0)] = 1.0
#    gf[np.where(z==1)] = 0.607372387536
#    gf[np.where(z==2)] = 0.417471529031
#
#    return gf
##print growthfactor(np.array([0,0,1,1,2,4,5]))
#
#def continuum_func(X, e, a, b, c, d):
#    x, y = X
#    return e + (a +  10**((b-x)*c)/growthfactor(y)*(1+y)**d
#
#def physical_func(X, e, a, b, c, d):
#    #return a +  (10**x/b)**c
#    x, y = X
#    return e + (a +  10**((x-b)*c))/growthfactor(y)*(1+y)**d


for index, iband in enumerate(lbands):

    x = []
    y = []
    bias = []

    for i in range(len(snapnum)):

        if snapnum[i] == 199 :
            z = 0
        elif snapnum[i] == 156 :
            z = 1
        elif snapnum[i] == 131 :
            z = 2
        elif snapnum[i] == 113 :
            z = 3

        #input data
        if iband[:1] == 'V':
            pb = np.genfromtxt(indirs+'/B%s/H_%s/%s_pb.txt'%(snapnum[i],dx,iband))
            tb = np.genfromtxt(indirs+'/B%s/T/Tinker10.txt'%(snapnum[i]))
            #xp = tb[:,0]
            #yp = tb[:,3]
            xp = pb[:,0]
            yp = pb[:,1]
            s = pb[:,2]
            xp = xp[np.nonzero(~np.isnan(s))]
            yp = yp[np.nonzero(~np.isnan(s))]
            s = s[np.nonzero(~np.isnan(s))]
            spbs = np.vstack((xp,yp,s)).T
            #lmin, lmax = 11, 14
            #spbs = spbs[np.where((lmin < spbs[:,0]) & (spbs[:,0] < lmax))[0],:]
            np.savetxt(indirs+'/B%s/H_%s/%s_pb.txt'%(snapnum[i],dx,iband), spbs)

        else:
            spbs = np.loadtxt(indirs+'/B%s/G_%s/%s/p_c/%s_p_b%s%s.txt'%(snapnum[i],dx,cso,iband,hl,sl))

        #spbs = spbs[np.where((lmin < spbs[:,0]) & (spbs[:,0] < lmax))[0],:]
        x = np.append(x,spbs[:,0])
        y = np.append(y,np.full(len(spbs[:,0]),z))
        bias = np.append(bias,spbs[:,1])

    #fit function
    #print x
    #print y
    #print bias
    #print growthfactor(y)
    #print growthfactor(y)*(1+y)**p0[3]
    #print func((x,y),p0[0],p0[1],p0[2],p0[3])

    #popt, pcov = curve_fit(func, (x,y), bias)
    fitfile = indirs+'/%s_%s.txt'%(iband,dx)
    if iband[:3] == 'mag':
        if os.path.isfile(fitfile):
            fit = genfromtxt(fitfile)
            p0 = fit[:dim]
            popt, pcov = curve_fit(continuum_func, (x,y), bias)
            #popt, pcov = curve_fit(continuum_func, (x,y), bias, p0)
        else:
            popt, pcov = curve_fit(continuum_func, (x,y), bias)
    else:
        if os.path.isfile(fitfile):
            fit = genfromtxt(fitfile)
            p0 = fit[:dim]
            #p0 = fit[:4]
            #print 'last term', growthfactor(y)*(1+y)**p0[3]
            #popt, pcov = curve_fit(physical_func, x, bias)

            popt, pcov = curve_fit(physical_func, (x,y), bias)
            #popt, pcov = curve_fit(physical_func, (x,y), bias, p0)
        else:
            #popt, pcov = curve_fit(physical_func, x, bias)
            popt, pcov = curve_fit(physical_func, (x,y), bias)

    perr = np.sqrt(np.diag(pcov))
    popt = np.append(popt,perr)
    print popt
    #np.savetxt(indirs+'/%s_%s.txt'%(iband,dx),popt.reshape((1,len(popt))))
    np.savetxt(indirs+'/%s_%s.txt'%(iband,dx),popt.reshape((1,len(popt))))
