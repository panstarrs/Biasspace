#! /usr/bin/env python
import math
import h5py
import numpy as np
import os.path, sys
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
#############################
#
#  INPUT ARGUMENTS
#
iarg = 1
while ( iarg <= len(sys.argv)-1):
    if( sys.argv[iarg] == '-outdir' ):
        plotdir = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-indir' ):
        indir = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-zlist' ):
        zlist = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-vlist' ):
        vlist = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-lbox' ):
        lbox = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-h0' ):
        h0 = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-obs' ):
        obs = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-bands' ):
        bands = sys.argv[iarg+1]
        minb = sys.argv[iarg+2]
        maxb = sys.argv[iarg+3]
        db = sys.argv[iarg+4]
        cso = sys.argv[iarg+5]
    iarg = iarg + 1

#############################

# setting parameters
suffix = 'png'
volh = float(lbox)**3 # In Mpc^3 h^3
vol  = volh/pow(float(h0),3.) # In Mpc^3

#iscentra = True
# LF limits and binning
#mbands = ['_GALEX-FUV_','_GALEX-NUV_','_SDSS-u_','_SDSS-g_','_SDSS-r_','_SDSS-i_','_SDSS-z_','_UKIDSS-Y_','_UKIRT-J_','_UKIRT-H_','_UKIRT-K_']
#bands = ['mhhalo','mstars_tot','L_tot_Halpha','mag_SDSS-g_r_tot', 'mag_SDSS-i_r_tot']
#mins = [9,2,32-40, -20, -20]
#maxs = [16,14,44-40,-10, -10]
#bins = [  int(i) for i in bins.split()]
#minb = [float(i) for i in minb.split()]
#maxb = [float(i) for i in maxb.split()]

lbands= [ i for i in bands.split()]
lmin = [float(i) for i in minb.split()]
lmax = [float(i) for i in maxb.split()]
if lbands[0][:1] == 'L':
  lmin = np.array(lmin)-40 # for Halpha's sake
  lmax = np.array(lmax)-40 # for Halpha's sake

# Host Halo Masses
hmin = 9 #5.
hmax = 16 #15.
dh   = 0.5
#dh   = float(db)
hbins = np.arange(hmin,hmax,dh)

#print lmin, lmax, lbands
#print bins, minb, maxb, bands
#for i in range(len(bins)) : mins[bins[i]]=minb[i]
#for i in range(len(bins)) : maxs[bins[i]]=maxb[i]

# Setting paramaters for Luminosities
#lbands = list( bands[i] for i in bins )
#lmin   = list( mins[i] for i in bins )
#lmax   = list( maxs[i] for i in bins )
lhist  = [[]]*len(lbands)
plf    = [[]]*len(lbands)
dl     = float(db)

#############################
# plot log-log histogram2d
rows = 2
cols = int(math.ceil(2*len(lbands) / float(rows)))
gs   = gridspec.GridSpec(rows, cols)
#ytit = ['log(M/Msun)','log(M/Msun)','log(Halpha L)','Mag']
fig  = plt.figure(figsize=(6*len(lbands),8))
fs   = 10
nlevel = 10

# plot mass/luminosity function
#labels = ['host halo mass','stellar mass','Dust not attenuated LF','r magnitude']
#ytit = "${log(\Phi/ \\rm{h^{3} Mpc^{-3} mag^{-1}}})$"
#xtit = ytit
colorrs = ['b']
xmin = lmin
xmax = lmax
ymin = -5.5
ymax = 0.

# stick with one
#xtit   = list(   xtit[i] for i in bins )
#ytit   = list(   ytit[i] for i in bins )
#labels = list( labels[i] for i in bins )
labels = lbands

#reading dhalo id and mass from file.txt
#dhid = np.loadtxt('/Users/hengxingpan/Work/danail/ProCorr/Bias/test/dgmapmass.txt')

for index, iband in enumerate(lbands):

  #if index == bins : continue
  #if index ==2 :
  #  lmin[index] = np.array(lmin[index])-40 # for Halpha's sake
  #  lmax[index] = np.array(lmax[index])-40 # for Halpha's sake
  #xmin[index] = lmin[index]
  #xmax[index] = lmax[index]
  print lmin[index], lmax[index], iband

  lbins = np.arange(lmin[index],lmax[index],dl)
  xlf   = lbins + dl*0.5
  plf[index]    = np.zeros(len(lbins))
  lhist[index]  = np.zeros(shape = (len(hbins),len(lbins)))

  for ivol in vlist.split():
    infg = indir+'/iz'+zlist.split()[0]+'/ivol'+ivol+'/galaxies.hdf5'
    #inft = indir+'/iz'+zlist.split()[0]+'/ivol'+ivol+'/galaxies.hdf5'
    inft = indir+'/iz'+zlist.split()[0]+'/ivol'+ivol+'/tosedfit.hdf5'
    fg = h5py.File(infg,'r')
    ft = h5py.File(inft,'r')

    smass = ft['Output001/mstars_tot'].value
    hmass = ft['Output001/mhhalo'].value

    if iband[:3] == 'mco':
      group = fg['Output001']
    else:
      group = ft['Output001']

    #gtype = fg['Output001/is_central'].value
    gtype = fg['Output001/type'].value

    if cso == 'centra':
       indc = np.where(( gtype == 0) & ( hmass >= 10**10 ))
    elif cso == 'sat_cen':
       indc = np.where((gtype != 2) & ( smass >= 10**7 ) & ( hmass >= 10**10))
    elif cso == 'orp_sat_cen':
       indc = np.where((smass >= 10**7) & ( hmass >= 10**10))

    #if iscentra : indc = np.where( gtype == 0 )

    # host halo ID
    #ghid = fg['/Output001/SubhaloID'].value[indc].astype(np.int64)
    #ind = np.nonzero(np.in1d(ghid, dhid[:,0]))

    # host halo mass
    #mapping = dict(zip(dhid[:,0], dhid[:,1]))
    #hmass = [ mapping[x] for x in ghid[ind]]
    #hmass = np.log10(hmass)

    hmass = np.log10(hmass[indc])
    #print ivol, hmass.shape
    #hmass = np.log10(hmass[ind]) - np.log10(float(h0))
    #print 'log hmass ',hmass.min(),hmass.max()
  
    # emitter luminosities vs. host halo mass
    #lu = group[iband].value
    lu = group[iband].value[indc]
    #print ivol, lu.shape
    if iband[:3] != 'mag':
      indl = np.where(lu > 0.)
      lu[indl] = np.log10(lu[indl])

    #print 'log lumi',lu.min(),lu.max()
    H,xedges,yedges =  np.histogram2d(hmass,lu,bins=[np.append(hbins,hmax),np.append(lbins,lmax[index])])
    lhist[index] = lhist[index] + H

    H, bins_edges = np.histogram(lu,bins=np.append(lbins,lmax[index]))
    plf[index] = plf[index] + H


  # emitter lumi vs Halo mass
  #fig.suptitle(label1+' '+label2)
  ax = fig.add_subplot(gs[0,index])
  ax.set_xlim(hbins[0],hbins[-1]+dh)
  ax.set_ylim(lbins[0], lbins[-1]+dl)
  #ax.set_xlabel(xtit[0],fontsize = fs)
  #ax.set_ylabel(ytit[index],fontsize = fs)
  ax.tick_params(labelsize=13)
  
  X, Y = np.meshgrid(hbins,lbins)
  Z = lhist[index].T
  np.savetxt(plotdir+lbands[index]+'_h.txt',Z)

  ind = np.where(Z > 0.)
  Z[ind] = np.log10(Z[ind])

  im = ax.imshow(Z, interpolation='nearest',origin='low', extent=[hbins[0], hbins[-1]+dh, lbins[0], lbins[-1]+dl],aspect='auto')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar(im,cax=cax)
  
  Xsm = X
  Ysm = Y
  Zsm = Z
  dr = (Zsm.max() - Zsm.min())
  lvls = Zsm.min() + dr*np.arange(nlevel)/(nlevel-1)
  al = lvls[int(nlevel/2)-1:]
  #print al
  #ax.contour(Xsm, Ysm, Zsm, levels=al, colors='b')

  # emitter lumi fucntion

  ax = fig.add_subplot(gs[1,index])
  ax.set_yscale('log')
  ax.set_xlim(xmin[index], xmax[index])
  #ax.set_ylim(ymin,ymax)
  #ax.set_xlabel(xtit[index],fontsize = fs)
  ax.set_ylabel("${\Phi/ \\rm{h^{3} Mpc^{-3} mag^{-1}}}$",fontsize = fs)

  py = plf[index]/volh/dl

  #py = np.cumsum(plf[index])
  np.savetxt(plotdir+lbands[index]+'_f.txt',np.vstack((xlf,py)).T)
  ax.plot(xlf,py,colorrs[0],label=labels[index])
  
  # Legend
  ax.legend(loc=3,fontsize='small')
  #ax.legend(loc=3,prop={"size":8})

# Save figure
#fig.tight_layout()
#plotfile = plotdir + "clf_z"+zlist.split()[0]+".%s"%(suffix)
#fig.savefig(plotfile)

if obs == 'T':
  # magnitudes function
  ytit = "${log(\Phi/ \\rm{h^{3} Mpc^{-3} mag^{-1}}})$"
  xmin = [-24.,-21.,-22.,-23.,-24.,-24.,-24.,-24.,-25.,-25.,-25.]
  xmax = [-14.,-11.,-12.,-13.,-14.,-14.,-14.,-14.,-15.,-15.,-15.]
  ymin = -5.5
  ymax = -1.
  
  # Observational data
  dobs = 'Obs_Data2/'  #dobs = 'Obs_Data2/'
  #fobs =['lf1500_z0_driver12.data','lf2300_z0_driver12.data','lfu_z0_driver12.data','lfg_z0_driver12.data','lfr_z0_driver12.data','lfi_z0_driver12.data','lfz_z0_driver12.data','lfy_z0_driver12.data','lfj_z0_driver12.data','lfh_z0_driver12.data','lfk_z0_driver12.data']
  fobs = ['lfr_z0_driver12.data']
  #bands = ['FUV','NUV','u','g','r','i','z','Y','J','H','K']
  bands = ['r']
  
  fig = plt.figure(figsize=(18.,9))
  #fig.suptitle(label1+' '+label2)
  for index, ib in enumerate(bands):
      ax = fig.add_subplot(gs[index])

      xtit = "${{\\rm M_{AB}("+ib+")}\, -\, 5log{\\rm h}}$"
      ax.set_xlim(xmax[index],xmin[index])
      ax.set_ylim(ymin,ymax) 
      ax.set_xlabel(xtit,fontsize=fs)
      ax.set_ylabel(ytit,fontsize=fs)
  
      ##########################
      # Plot observations
      file = dobs+fobs[index]
      mag, den, err, num = np.loadtxt(file,unpack=True)
      ind = np.where(den - err > 0.)
      x = mag[ind]
      y = np.log10(den[ind]/0.5) # Observations are per 0.5 mag
      eh = np.log10(den[ind]+err[ind]) - np.log10(den[ind])
      el = np.log10(den[ind]) - np.log10(den[ind]-err[ind])
      ind = np.where(np.isinf(el) | np.isnan(el))
      el[ind] = 999.
      ax.errorbar(x,y,yerr=[el,eh],fmt='o', ecolor='grey',color='grey', mec='grey', label='Driver et al. 2012')
  
      ##########################
      # Plot predictions
      py = plf[index]
      ind = np.where(py < 0.)
      ax.plot(xlf[ind],py[ind],'b--',label='Not attenuated LF')
  
  
      ##########################
      # Legend
      #leg = ax.legend(loc=3,fontsize='small')
      leg = ax.legend(loc=3,prop={"size":8})
      colors = ['b','b','grey']
      for color,text in zip(colors,leg.get_texts()):
          text.set_color(color)
      leg.draw_frame(False)

      ##########################
  # Save figure
  fig.tight_layout()
  plotfile = plotdir + "All_maf_z"+zlist.split()[0]+".%s"%(suffix)
  fig.savefig(plotfile)
  print 'making '+"All_maf_z"+zlist.split()[0]+".%s"%(suffix)

