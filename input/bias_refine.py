import numpy as np
from astropy.io import ascii

#data = ascii.read('/home/hpan/ProCorr/Bias/input/physical.txt')
#data = ascii.read('/home/hpan/ProCorr/Bias/input/lbands_r_tot_ext.txt')

data = ascii.read('allc.txt')
lbands  =  data['bands_r'].data

#data = ascii.read('allp.txt')
#lbands  =  data['physicals'].data

hl = 0
sl = 0

#dx = 0.5

allc = [[]]*4

snapnum = [199,156,131,113]
#snapnum = [199]
for i in range(len(snapnum)):


  allc[i] = data['z%s'%(i)].data
  #print allc[i]
  for index, bands in enumerate(lbands):

    if bands[:3]=='mag':
      dx = 0.5
    else:
      dx = 0.2

    indir = '../output/B%s/G_%s/sat_cen'%(snapnum[i],dx)
    stb = np.loadtxt(indir+'/p_c/%s_tb%s%s.txt'%(bands,hl,sl))
    #spb = np.loadtxt(indir+'/p_c/%s_pb.txt'%(bands))
    pb = np.genfromtxt(indir+'/p_c/%s_pb%s%s.txt'%(bands,hl,sl))
    x = pb[:,0]
    y = pb[:,1]
    s = pb[:,2]
    x = x[np.nonzero(~np.isnan(s))]
    y = y[np.nonzero(~np.isnan(s))]
    s = s[np.nonzero(~np.isnan(s))]
    spb = np.vstack((x,y,s)).T
    #filter the points outside the bins
    if bands[:3]=='mag':
      stb = stb[np.where(allc[i][index] >= stb[:,0])[0],:]
      spb = spb[np.where(allc[i][index] >= spb[:,0])[0],:]
    else:
      stb = stb[np.where(allc[i][index] <= stb[:,0])[0],:]
      spb = spb[np.where(allc[i][index] <= spb[:,0])[0],:]
    np.savetxt(indir+'/p_c/%s_t_b%s%s.txt'%(bands,hl,sl), stb)
    np.savetxt(indir+'/p_c/%s_p_b%s%s.txt'%(bands,hl,sl), spb)


