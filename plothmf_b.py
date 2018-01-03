import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
import sys
import itertools
from numpy import genfromtxt
from bias_theor import growthfactor, continuum_func, physical_func

print "welcome to plot halo bias and mass function"
#plot host halo mass function
#snapnum = [199]
snapnum = [199,156,131,113]

outdir = './output'

cso = 'centra'
#cso = 'centra'
bands = 'mhhalo'
#bands = 'Vhalo_c'
dh = 0.2
hl = 0
sl = 0


#labels
#labels = ['Sheth-Tormen','Tinker+10','velociraptor(mass_FOF)']
labels = [bands,'Tinker+10','Sheth-Tormen']
linestyles = ['-', '--', ':']
#labels = ['Galform','Tinker+10','Sheth-Tormen']
#labels = ['Sheth-Tormen','Tinker+10','velociraptor(mass_FOF)','galform(mhhalo)']
hmfdat = [[]]*len(labels)

# Create the figure object
#fig,ax = plt.subplots(2,1,sharex=True,squeeze=True,gridspec_kw={"height_ratios":(2,1)},figsize=(6,6), subplot_kw={"xscale":"log"})

fig,ax = plt.subplots()

# Set up the axis styles etc.
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$M(h^{-1}M_{\bigodot})$',fontsize=12)
ax.set_xlim((10**11.2,10**14))
ax.set_ylim((5e-7,1))
ax.set_ylabel(r'dn/dlogM ($h^{3}Mpc^{-3}$)', fontsize=12)
ax.yaxis.set_major_locator(LogLocator(numticks=9))
ax.tick_params(direction='in',axis='both', which='both', labelsize=11,left='on',  top='on', right='on', bottom='on',)
#ax[1].set_xlabel(r'$M(h^{-1}M_{\bigodot})$',fontsize=12)
#ax[1].set_ylabel(r'$dn/dn_{st}$')
#ax[1].set_ylim((0,2))
#ax[1].yaxis.set_major_locator(MaxNLocator(5))
#ax[1].tick_params(direction='in',axis='both', which='both', labelsize=11,left='on',  top='on', right='on', bottom='on',)

# Contract the space a bit
#plt.subplots_adjust(wspace=0.08,hspace=0.00)

# Plot the theoretical data
#sncolors = ["lightgray", "grey", "k"]
sncolors = ['black','red','blue','green']
fmts = ['ko','ro','bo','go']
zlabels = ['z=0','z=1','z=2','z=3']
zx = 0.1
zy = np.array([0.6,0.5,0.4,0.3])

for i in range(len(snapnum)):

  indir = './output/B%s'%(snapnum[i])
  #indir = './test/linp/B%s'%(snapnum[i])
  if bands[:1] == 'V' :
    hmfdat[0] = np.loadtxt(indir+'/H_%s/%s_f.txt'%(dh,bands))
  else:
    hmfdat[0] = np.loadtxt(indir+'/G_%s/%s/f_h/%s_f%s%s.txt'%(dh,cso,bands,hl,sl))

  #indir = './output/B%s'%(snapnum[i])
  hmfdat[1] = np.loadtxt(indir+'/T/Tinker10.txt')
  hmfdat[2] = np.loadtxt(indir+'/T/ST.txt')
  #hmfdat[3] = np.loadtxt('./B%s_mhhalo_f.txt'%(snapnum))

  #hmfdat[0] = hmfdat[0][1:-1,:]
  #hmfdat[1] = hmfdat[1][1:-1,:]
  #hmfdat[2] = hmfdat[2][1:-1,:]

  ax.annotate(zlabels[i], xy=(zx, zy[i]-0.1), xycoords="axes fraction",color=sncolors[i],size=10)

  for j in range(len(labels)):
    if i==0 :ax.plot(10**hmfdat[j][:,0],hmfdat[j][:,1],label=labels[j],lw=1,color=sncolors[i], linestyle=linestyles[j])
    ax.plot(10**hmfdat[j][:,0],hmfdat[j][:,1],lw=1,color=sncolors[i], linestyle=linestyles[j])
    #ax[1].plot(10**hmfdat[0][:,0],hmfdat[i][:,1]/hmfdat[0][:,1],lw=1,color=colors[i])

ax.legend(loc='best',frameon=False)
#plt.show()
fig.savefig(outdir+'/%s_f.png'%(bands))

#exit()

#plot halo bias
labels = ['velociraptor(mass_FOF)','Tinker+10']
#labels = ['Galform','Tinker+10']
#labels = ['velociraptor(mass_FOF)','Tinker+10','Sheth-Tormen']

fig,ax = plt.subplots()

ax.set_xscale('log')
ax.set_xlabel(r'$M(h^{-1}M_{\bigodot})$',fontsize=12)
ax.set_xlim((10**11.2,10**14))
ax.set_ylim((0.5,16))
ax.set_ylabel(r'$bias(M)$', fontsize=12)
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.tick_params(direction='in',axis='both', which='both', labelsize=11,left='on',  top='on', right='on', bottom='on',)

fit = genfromtxt('./output/%s_%s.txt'%(bands,dh))
print '%s'%(bands)
print  np.around(fit[:5],2)
print  np.around(fit[5:],2)
popt = fit[:5]
perr = fit[5:]


for i in range(len(snapnum)):

  indir = './output/B%s'%(snapnum[i])

  #hmfdat[3] = np.loadtxt('./B%s_mhhalo_f.txt'%(snapnum))
  if bands[:1] == 'V' :
    biass = np.genfromtxt(indir+'/H_%s/%s_pb.txt'%(dh,bands))
  else:
    biass = np.genfromtxt(indir+'/G_%s/%s/p_c/%s_pb%s%s.txt'%(dh,cso,bands,hl,sl))
  tink = np.loadtxt(indir+'/T/Tinker10.txt')
  st = np.loadtxt(indir+'/T/ST.txt')

  m = biass[:,0]
  y = biass[:,1]
  s = biass[:,2]

  m = m[np.nonzero(~np.isnan(y))]
  y = y[np.nonzero(~np.isnan(y))]
  s = s[np.nonzero(~np.isnan(y))]

  hmfdat[0] = np.vstack((m,y,s)).T
  hmfdat[1] = np.vstack((tink[:,0],tink[:,3])).T
  hmfdat[2] = np.vstack((st[:,0],    st[:,3])).T

  xdata = np.linspace(m.min(), m.max(), 100)

  if snapnum[i] == 199 :  z = 0
  elif snapnum[i] == 156 :z = 1
  elif snapnum[i] == 131 :z = 2
  elif snapnum[i] == 113 :z = 3

  ax.annotate(zlabels[i], xy=(zx, zy[i]), xycoords="axes fraction",color=sncolors[i],size=10)

  for j in range(len(labels)):
    if j==0 :
      if i==0 :
        ax.errorbar(10**hmfdat[j][:,0],hmfdat[j][:,1],yerr=hmfdat[j][:,2],label=labels[j],fmt=fmts[i],capsize=3, elinewidth=0.5,linewidth=None,markersize='1')
        ax.plot(10**xdata, physical_func((xdata,np.full(len(xdata),z)), *popt), color=sncolors[i], label='best fit')
      else:
        ax.errorbar(10**hmfdat[j][:,0],hmfdat[j][:,1],yerr=hmfdat[j][:,2],fmt=fmts[i],capsize=3, elinewidth=0.5,linewidth=None,markersize='1')
        ax.plot(10**xdata, physical_func((xdata,np.full(len(xdata),z)), *popt), color=sncolors[i])
    else:
      if i==0 :
        ax.plot(10**hmfdat[j][:,0],hmfdat[j][:,1],label=labels[j],lw=1,color=sncolors[i], linestyle=linestyles[j])
        #ax.plot(10**hmfdat[j][:,0],hmfdat[j][:,1]/growthfactor(1),label='tink1',lw=1,color=sncolors[i], linestyle=':')
        #ax.plot(10**hmfdat[j][:,0],hmfdat[j][:,1]/growthfactor(2),label='tink2',lw=1,color=sncolors[i], linestyle=':')
      else:
        ax.plot(10**hmfdat[j][:,0],hmfdat[j][:,1],lw=1,color=sncolors[i], linestyle=linestyles[j])
        #ax[1].plot(10**hmfdat[0][:,0],hmfdat[i][:,1]/hmfdat[0][:,1],lw=1,color=colors[i])

ax.legend(loc='best',frameon=False)
plt.show()
fig.savefig(outdir+'/%s_b.png'%(bands))








'''
fig1 = plt.figure(figsize=(8, 6), dpi=80)
ax = fig1.add_subplot(111)
#ax.plot(vhmf[:,0],np.log10(vhmf[:,1]),'b',label='Velocriaptor(Mass_FOF) z=%s'%(z))
ax.plot(mhmf[:,0],np.log10(mhmf[:,1]),'green',label='galform(mhhalo)')
ax.plot(thmf[:,0],np.log10(thmf[:,2]),'grey',label='Tinker10')
ax.plot(shmf[:,0],np.log10(shmf[:,2]),'red',label='Sheth-Tormen')

ax.set_xlabel(r'$logM(h^{-1}M_{\bigodot})$', fontsize=15)
ax.set_ylabel(r'dn/dlogM($h^{3}Mpc^{-3}$)', fontsize=15)
#ax.set_yscale('log')
ax.set_xlim(9,16)

ax.grid(b=True, which='major', color='grey', linestyle='-.')
ax.legend(loc='best')

fig1.savefig('B%s_vhmf_FOFt.png'%(snapnum))
#fig1.savefig('B%s_vhmf_tot.png'%(snapnum))

print "************************* DONE *************************"
'''

