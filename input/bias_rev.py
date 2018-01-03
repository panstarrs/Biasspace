import numpy as np
from astropy.io import ascii


hl = 0
sl = 0

data = ascii.read('allc.txt')
lbands  =  data['bands_r'].data

#dx = 0.5

allc = [[]]*4

snapnum = [199,156,131,113]
#snapnum = [199]
for i in range(len(snapnum)):


  #print allc[i]
  for index, bands in enumerate(lbands):

    if bands[:3]=='mag':
      dx = 0.5
    else:
      dx = 0.2

    indir = '../output/B%s/G_%s/sat_cen'%(snapnum[i],dx)
    pb = np.loadtxt(indir+'/p_c/%s_tb%s%s.txt'%(bands,hl,sl))
    #spb = np.loadtxt(indir+'/p_c/%s_pb.txt'%(bands))
    x = pb[:,0]
    x = x[::-1]
    y = pb[:,1]
    s = pb[:,2]
    x = x[np.nonzero(~np.isnan(s))]
    y = y[np.nonzero(~np.isnan(s))]
    s = s[np.nonzero(~np.isnan(s))]
    stb = np.vstack((x,y,s)).T
    #filter the points outside the bins
    np.savetxt(indir+'/p_c/%s_tb%s%s.txt'%(bands,hl,sl), stb)


