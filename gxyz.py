#! /usr/bin/env python
import math
import h5py
import numpy as np
import os.path, sys
import matplotlib
matplotlib.use('Agg')
import struct
from bias_theor import interpgb
#############################
#
#  INPUT ARGUMENTS
#
iarg = 1
while ( iarg <= len(sys.argv)-1):
    if( sys.argv[iarg] == '-outdir' ):
        outdir = sys.argv[iarg+1]
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
    if( sys.argv[iarg] == '-bands' ):
        bands = sys.argv[iarg+1]
        #bins = sys.argv[iarg+1]
        minb = sys.argv[iarg+2]
        maxb = sys.argv[iarg+3]
        db = sys.argv[iarg+4]
        cso = sys.argv[iarg+5]
    iarg = iarg + 1

#############################
#
# setting parameters
snapnum = int(zlist.split()[0])
if   snapnum == 199 : z=0
elif snapnum == 156 : z=1
elif snapnum == 131 : z=2
elif snapnum == 113 : z=3

h0   = float(h0)
lbox = float(lbox)
volh = lbox**3 # In Mpc^3 h^3
#vol  = volh/pow(h0,3.) # In Mpc^3
print z, lbox, h0

#iscentra = True
# LF limits and binning
# gbands = ['_GALEX-FUV_','_GALEX-NUV_','_SDSS-u_','_SDSS-g_','_SDSS-r_','_SDSS-i_','_SDSS-z_','_UKIDSS-Y_','_UKIRT-J_','_UKIRT-H_','_UKIRT-K_']
# bands = ['mhhalo','mstars_tot','L_tot_Halpha','mag_SDSS-g_r_tot', 'mag_SDSS-i_r_tot']
# bands = bands.split()
# mins  = [9,2,32-40,-20.0, -20.0] #5.
# maxs  = [16,14,44-40,-10.0, -20.0] #15.

#bins = [  int(i) for i in bins.split()]
lbands= [ i for i in bands.split()]
lmin = [float(i) for i in minb.split()]
lmax = [float(i) for i in maxb.split()]

if lbands[0][:1] == 'L':
  lmin = np.array(lmin)-40 # for Halpha's sake
  lmax = np.array(lmax)-40 # for Halpha's sake

#
#exit()
#
#for i in range(len(bins)) : mins[bins[i]]=minb[i]
#for i in range(len(bins)) : maxs[bins[i]]=maxb[i]
#
#lbands = list( bands[i] for i in bins )
#lmin   = list(  mins[i] for i in bins )
#lmax   = list(  maxs[i] for i in bins )
dl   = float(db)


for index, iband in enumerate(lbands):
  #if index != bandind : continue
  bias = []
  #if index ==2 :
  #  lmin[index] = lmin[index]-40 # for Halpha's sake
  #  lmax[index] = lmax[index]-40 # for Halpha's sake

  print lmin[index], lmax[index], iband
  lbins = np.arange(lmin[index],lmax[index],dl)
  for il in range(len(lbins)):
    if iband[:1] == 'L':
      subffix = "%05.2f"%(lbins[il]+40+dl/2.)
    else:
      subffix = "%05.2f"%(lbins[il]+dl/2.)

    output_file = outdir+'/'+iband+'_%s'%(subffix.strip(" "))
    print output_file
    out_file = open(output_file,"wb")
    num = 0
    hms = []

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
      gtype = group['type'].value

      if cso == 'centra':
         indc = np.where(( gtype == 0) & ( hmass >= 10**10 ))
      elif cso == 'sat_cen':
         indc = np.where((gtype != 2) & ( smass >= 10**7 ) & ( hmass >= 10**10))
      elif cso == 'orp_sat_cen':
         indc = np.where((smass >= 10**7) & ( hmass >= 10**10))

      #if iscentra : indc = np.where( gtype == 0 )
      # xyz coords
      Xc = np.array(fg['/Output001/xghalo'])[indc]
      Yc = np.array(fg['/Output001/yghalo'])[indc]
      Zc = np.array(fg['/Output001/zghalo'])[indc]
      #Xc = np.array(fg['/Output001/xgal'])[indc]
      #Yc = np.array(fg['/Output001/ygal'])[indc]
      #Zc = np.array(fg['/Output001/zgal'])[indc]

      # host halo mass
      hmass = np.log10(hmass[indc])
      #hmass = np.log10(hmass[ind]) - np.log10(float(h0))
      # emitter luminosities or stellar mass
      lu = group[iband].value[indc]
      if iband[:3] != 'mag':
        indl = np.where(lu > 0.)
        lu[indl] = np.log10(lu[indl])

      indl = np.where((lu >= lbins[il] ) & (lu < lbins[il]+dl))
      # all halo mass
      hms = np.append(hms,hmass[indl])
      # output files
      #pos = np.vstack((Xc[indl],Yc[indl],Zc[indl])).T
      pos = np.vstack((Xc[indl],Yc[indl],Zc[indl],lu[indl])).T
      out_file.write(struct.pack('f'*pos.size, *pos.flatten()))
      num = num + len(Xc[indl])

    # output bias
    if len(hms) == 0:
      biasg = np.array([0])
    else:
      biasg = interpgb('./output/B%s/G_0.5/centra/p_c'%(snapnum),hms)
      #biasg = interpgb(outdir,hms)
      #biasg = interpgb(lmin[0],lmax[0],dl,z,10**hms)
    if iband[:1] == 'L':
      Tb = [lbins[il]+40+dl/2., biasg.mean(), biasg.std(),len(biasg)/(dl*volh),num/(dl*volh)]
    else:
      Tb = [lbins[il]+dl/2., biasg.mean(), biasg.std(),len(biasg)/(dl*volh),num/(dl*volh)]
    print Tb
    bias = np.append(bias,Tb)
    out_file.close()
  bias = bias.reshape((len(bias)/5,5))
  np.savetxt(outdir+'/'+iband+'_tb.txt',bias)
