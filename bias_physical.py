import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
import sys
import math
from numpy import genfromtxt
from bias_theor import sn_to_z ,growthfactor, continuum_func, physical_func
from astropy.io import ascii


#iarg = 0
#bands = sys.argv[iarg+1]
#minb = sys.argv[iarg+2]
#maxb = sys.argv[iarg+3]
#db = sys.argv[iarg+4]

#print bands, minb, maxb, db

indirs = '/Users/hengxingpan/Work/danail/ProCorr/Bias/output'
cso = ['centra','sat','orp','sat_cen','orp_sat','orp_sat_cen']
cso = cso[3]

fitplot = True
bins = 3 #index of lbands, bad nickname
dbins = 4
db = 0.2 #width of bin
hl = 0
sl = 0
dim = 5

#snapnum = [113]
snapnum = [199,156,131,113]
sncolors = ['black','red','blue','green']
fmts = ['ko','ro','bo','go']
linew = [0.5,0.5,0.5,0.5]
zlabels = ['z=0','z=1','z=2','z=3']
zx = 0.1
zy = np.array([0.6,0.5,0.4,0.3])


data = ascii.read('/Users/hengxingpan/Work/danail/ProCorr/Bias/input/physicals.txt')
lbands = data['physicals'].data
labels = data['labels'].data
xlabel = data['xlabel'].data
latexs = data['latexs'].data
xlow = data['mins'].data
xupp = data['maxs'].data

lbands = lbands[bins:bins+dbins]
labels = labels[bins:bins+dbins]
latexs = latexs[bins:bins+dbins]
xlow  = xlow[bins:bins+dbins]
xupp  = xupp[bins:bins+dbins]
dx = db

bestfit = [[]]*len(lbands)
fiterro = [[]]*len(lbands)
plusmin = [['$\pm$']]*len(lbands)*dim
#############################
# plot log-log histogram2d
rows = 1
cols = int(math.ceil(len(lbands) / float(rows)))
gs   = gridspec.GridSpec(rows, cols)

fig  = plt.figure(figsize=(4*cols,4*rows))

for index, iband in enumerate(lbands):

  xdata = np.linspace(xlow[index], xupp[index], 100)
  #if index==0 :
  fit = genfromtxt(indirs+'/%s_%s.txt'%(iband,db))
  print '%s'%(iband)
  print  np.around(fit[:5],2)
  print  np.around(fit[5:],2)
  popt = fit[:5]
  perr = fit[5:]
  bestfit[index] = np.around(fit[:5],2)
  fiterro[index] = np.around(fit[5:],2)

  ax = fig.add_subplot(gs[index])
  ax.set_xlim(xlow[index], xupp[index])
  #ax.tick_params(labelsize=13)

  ax.set_ylabel(r'$bias$', fontsize=15)
  if iband[0] == 'L':
    ax.set_xlabel('%s'%(xlabel[index+3]))
  else:
    ax.set_xlabel('%s'%(xlabel[index]))
  #ax.invert_xaxis()

  ax.annotate(labels[index], xy=(0.25, 0.7), xycoords="axes fraction",color='black',size=15)
  ax.tick_params(direction='in',
                bottom='on',
                top='on',
                left='on',
                right='on')
  for i in range(len(snapnum)):

    stbs = np.loadtxt(indirs+'/B%s/G_%s/%s/p_c/%s_t_b%s%s.txt'%(snapnum[i],db,cso,iband,hl,sl))
    spbs = np.loadtxt(indirs+'/B%s/G_%s/%s/p_c/%s_p_b%s%s.txt'%(snapnum[i],db,cso,iband,hl,sl))
    ymax = max(4,spbs[:,1].max())
    if index == 0 :
      ax.annotate(zlabels[i], xy=(zx, zy[i]), xycoords="axes fraction",color=sncolors[i],size=10)
    if snapnum[i] == 199 :
      ax.plot(stbs[:,0],stbs[:,1],color=sncolors[i], linestyle='--',label='halo model prediction',linewidth=linew[i])
      ax.errorbar(spbs[:,0], spbs[:,1], yerr=spbs[:,2], fmt=fmts[i],ecolor=sncolors[i],capsize=3, elinewidth=linew[i],linewidth=None,label='direct measurement',markersize='2')
      if fitplot == True: ax.plot(xdata, physical_func((xdata,np.full(len(xdata),sn_to_z(snapnum[i]))), *popt), color=sncolors[i], label='best fit',linewidth=linew[i])
    else :
      ax.plot(stbs[:,0],stbs[:,1],color=sncolors[i], linestyle='--',linewidth=linew[i])
      ax.errorbar(spbs[:,0], spbs[:,1], yerr=spbs[:,2], fmt=fmts[i],ecolor=sncolors[i],capsize=3, elinewidth=linew[i],linewidth=None,markersize='2')
      if fitplot == True: ax.plot(xdata, physical_func((xdata,np.full(len(xdata),sn_to_z(snapnum[i]))), *popt), color=sncolors[i],linewidth=linew[i])

  ax.set_ylim(0.5, 1.2*ymax)
  if index == 2 or index == 1: ax.set_ylim(0.5, 6)
  if index == 0: ax.legend(loc='best',frameon=False)


abcd = np.array(bestfit).reshape((len(lbands),dim))
abcd = np.transpose(abcd)

bestfit = np.array(bestfit).reshape((len(lbands),dim)).astype(str)
fiterro = np.array(fiterro).reshape((len(lbands),dim)).astype(str)
plusmin = np.array(plusmin).reshape((len(lbands),dim))
combine = np.core.defchararray.add(bestfit,plusmin)
combine = np.core.defchararray.add(combine,fiterro)

if iband[:1] == 'L' :
    ascii.write(np.column_stack((latexs,combine)), indirs+'/emission_parameters_latex.txt',format='latex', names=['Bands','a','b','c','d','e'],overwrite=True)
else:
    ascii.write(np.column_stack((latexs,combine)), indirs+'/physical_parameters_latex.txt',format='latex', names=['Bands','a','b','c','d','e'],overwrite=True)

#ascii.write(np.column_stack((labels,combine)), indirs+'/%s_latex.txt'%(iband),format='latex', names=['Bands','a','b','c','d'],overwrite=True)
fig.tight_layout()

if iband[:1] == 'L' :
    fig.savefig(indirs+'/emission_b.png')
else:
    fig.savefig(indirs+'/physical_b.png')
#fig.savefig(indirs+'/%s_b.png'%(iband))
plt.show()
exit()


# only for mass or luminosity function
fig  = plt.figure(figsize=(4*cols,3*rows))
for index, iband in enumerate(lbands):

  xdata = np.linspace(xlow[index], xupp[index], 100)
  ax = fig.add_subplot(gs[index])
  ax.set_xlim(xlow[index], xupp[index])
  #ax.tick_params(labelsize=13)

  ax.set_ylabel(r'dn/dlogM($h^{3}Mpc^{-3}$)', fontsize=10)
  if iband[0] == 'L':
    ax.set_xlabel('%s'%(xlabel[index+3]))
  else:
    ax.set_xlabel('%s'%(xlabel[index]))

  ax.tick_params(direction='in',
                bottom='on',
                top='on',
                left='on',
                right='on')
  ymax = 0
  for i in range(len(snapnum)):

    smf = np.loadtxt(indirs+'/B%s/G_%s/%s/f_h/%s_f.txt'%(snapnum[i],db,cso,iband))

    ind = np.where(smf[:,1] > 0.)
    smfx = smf[:,0][ind]
    smfy = np.log10(smf[:,1][ind])

    ymax = max(0,smf[:,1].max())

    if iband[:1] == 'L' :
      ax.plot(smfx+40, smfy,'black',label='%s'%(iband), color=sncolors[i])
    else:
      if snapnum[i] == 199:
        ax.plot(smfx, smfy,label='%s'%(iband), color=sncolors[i])
      else:
        ax.plot(smfx, smfy, color=sncolors[i])

    if index == 0 :
      ax.annotate(zlabels[i], xy=(zx, zy[i]), xycoords="axes fraction",color=sncolors[i],size=10)
  ax.set_ylim(-5, ymax)
  ax.legend(loc='upper right',frameon=False)

fig.tight_layout()
if iband[:1] == 'L' :
    fig.savefig(indirs+'/emission_f.png')
else:
    fig.savefig(indirs+'/physical_f.png')



# for 2d pdf of  Host Halo Masses against x axis
hmin = 9 #5.
hmax = 16 #15.
#dh   = float(db)
dh = 0.5
hbins = np.arange(hmin,hmax,dh)


dl = float(db)
for i in range(len(snapnum)):

  z = sn_to_z(snapnum[i])
  # panel one
  fig = plt.figure(figsize=(4*len(lbands),4))
  gs1 = gridspec.GridSpec(1, 3,
                         width_ratios=[2, 2, 2],
                         #height_ratios=[1, 3]
                         )
  gs1.update(wspace=0.25,left=0.06, right=0.98)
  #gs1.update(hspace=0.1, wspace=0.25,left=0.06, right=0.98)
  #gs1.update(hspace=0.1,left=0.05, right=0.98)
  for index, iband in enumerate(lbands):

    lbins = np.arange(xlow[index],xupp[index],dx)

    smf = np.loadtxt(indirs+'/B%s/G_%s/%s/f_h/%s_f.txt'%(snapnum[i],db,cso,iband))
    hsf = np.loadtxt(indirs+'/B%s/G_%s/%s/f_h/%s_h.txt'%(snapnum[i],db,cso,iband))
    '''
    ax1 = plt.subplot(gs1[0, index])
    ax1.set_title("%s   z = %s"%(iband,z))
    if index == 0: ax1.set_ylabel(r'dn/dlogM($h^{3}Mpc^{-3}$)', fontsize=10)
    ax1.xaxis.set_ticklabels([])
    ax1.set_xlim(lbins[0], lbins[-1]+dl)
    ax1.set_ylim(-5, 0)
    #ax1.invert_xaxis()
    ax1.tick_params(direction='in',
    		which='both',
    		axis='both',
                    bottom='on',
                    top='on',
                    left='on',
                    right='on')
    if iband[:1] == 'L' :
      ax1.plot(smf[:,0]+40, np.log10(smf[:,1]),'black',label='%s'%(iband))
    else:
      ax1.plot(smf[:,0], np.log10(smf[:,1]),'black',label='%s'%(iband))
    #ax1.legend(loc='best')
    if i == 0: ax1.legend(loc='best',frameon=False)
    '''

    ax2 = plt.subplot(gs1[0, index])
    ax2.set_title("%s   z = %s"%(iband,z))
    #ax2 = plt.subplot(gs1[1, index])
    if index == 0: ax2.set_ylabel(r'$\log M$ $[M_{\odot}]$', fontsize=15)

    ind = np.where(hsf > 0.)
    hsf[ind] = np.log10(hsf[ind])

    ax2.imshow(hsf.T, interpolation='nearest',origin='low', extent=[lbins[0], lbins[-1]+dl, hbins[0], hbins[-1]+dh],aspect='auto')

    ax2.tick_params(direction='in',
                    bottom='on',
                    top='on',
                    left='on',
                    right='on')


    ax2.xaxis.set_visible(True)
    #ax2.set_xlabel('%s'%(iband))
    #ax2.invert_xaxis()

    plt.setp(ax2.get_xticklabels(), visible=True)

  plt.show()
  if iband[:1] == 'L' :
    fig.savefig(indirs+'/emission_%s_h.png'%(z))
  else:
    fig.savefig(indirs+'/physical_%s_h.png'%(z))
