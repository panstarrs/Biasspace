import matplotlib.pyplot as plt
import numpy as np
import os, glob

#t = np.arange(0.0, 2.0, 0.01)


#z0 = np.loadtxt(indir+'L210_N512_199_p.txt')
#z1 = np.loadtxt(indir+'L210_N512_156_p.txt')
#z2 = np.loadtxt(indir+'L210_N512_131_p.txt')
#z3 = np.loadtxt(indir+'/P_256/L210_N512_113_p.txt')


#l0 = np.loadtxt('camb_matterpower_199.dat')
#l1 = np.loadtxt('camb_matterpower_156.dat')
#l2 = np.loadtxt('camb_matterpower_130.dat')


#plt.loglog(z0[:,0], z0[:,1], color='black')
#plt.loglog(z1[:,0], z1[:,1])
#plt.loglog(z2[:,0], z2[:,1])
##
##plt.loglog(hz0[:,1], hz0[:,2])
##plt.loglog(hz1[:,1], hz1[:,2])
#
#plt.loglog(l0[:,0], l0[:,1])
#plt.loglog(l1[:,0], l1[:,1])
#plt.loglog(l2[:,0], l2[:,1])
#plt.show()
#
#exit()

snapnum = 113
dh = 0.2

indir = '/Users/hengxingpan/Work/danail/ProCorr/Bias/output/B%s'%(snapnum)

tink = np.loadtxt(indir+'/T/Tinker10.txt')
hmfdat = np.vstack((tink[:,0],tink[:,3])).T

#dhm = np.linspace(11.1,13.9,15)
#dhm = np.linspace(11.25,13.75,6)
dhm = np.linspace(11.1,13.9,15)
dhb = []
dhe = []
limx = 3

#plot power spectrum
#fig1 = plt.figure()
#ax = fig1.add_subplot(111)
#
#ax.plot(z0[:,0], z0[:,1],linestyle='-',color='black',linewidth=1)
#for i in range(15):
#  pdh = np.loadtxt('Vhalo_F_%s_N.txt'%(i))
#  ax.plot(pdh[:,0], pdh[:,1],
#         linestyle='-',
#         linewidth=0.5)
#ax.set_title('Matter power at z=0')
#ax.set_xlabel('k/h Mpc')
#ax.set_ylabel('P')
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.show()
#exit()

idir=indir+'/H_%s/'%(dh)
hmf = np.loadtxt(idir+'/Vhalo_F_f.txt')
#idir=indir+'/G_%s/centra/'%(dh)
#hmf = np.loadtxt(idir+'/f_h/mhhalo_f108.txt')
pp  = np.loadtxt(indir+'/P_256/L210_N512_%s_p.txt'%(snapnum))
ng  = hmf[:,2]

fig = plt.figure()
ax = fig.add_subplot(111)

ap = 0
bp = 28#14
Np = 512**3
Vsim = 210**3

ymax = 6
ind = 0
labels=['11.0 < $log(M_{\odot}/h)$ < 11.2',
        '11.6 < $log(M_{\odot}/h)$ < 11.8',
        '12.2 < $log(M_{\odot}/h)$ < 12.4',
        '12.8 < $log(M_{\odot}/h)$ < 13.0',
        '13.4 < $log(M_{\odot}/h)$ < 13.6',
        '13.8 < $log(M_{\odot}/h)$ < 14.0']

#for i, ifile in enumerate(glob.glob(idir+'/p_c/*p.txt')):
for i, ifile in enumerate(glob.glob(idir+'/*p.txt')):
#for i, ifile in enumerate(os.listdir(idir+'/H_0.2/')):
  print i, ifile
  if i >= len(ng) : continue
  pdh = np.loadtxt(ifile)
  #pdh = np.loadtxt('Vhalo_F_%s_N.txt'%(i))
  #pdh = np.loadtxt('mhhalo_F_%s.txt'%(i+1))
  dk = pdh[ap+1:bp+1,0]-pdh[ap:bp,0]
  k = pdh[ap:bp,0]
  pgg = pdh[ap:bp,1]
  pmm = pp[ap:bp,1]

  ppo = np.where(pgg > 0)
  pgg = pgg[ppo]
  pmm = pmm[ppo]
  k  =  k[ppo]
  dk = dk[ppo]
  #print pgg
  #print pmm

  bias = (pgg/pmm)**0.5
  #sigma = 2*np.pi**2/pdh[ap:bp,0]**2/dk/ng[i]/z0[ap:bp,1]
  #sigav = sigma.sum()/(bp-ap+1)**2
  #bav = bias.mean()

  ngb = ng[i]/Vsim
  npb = Np/Vsim

  sigma = (np.pi*bias)**2/k**2/dk/Vsim*(2/ngb/pgg+2/npb/pmm+1/(ngb*pgg)**2+1/(npb*pmm)**2)
  sigav = (1/np.sum(1/sigma))**0.5
  bav = np.sum(bias/sigma)/np.sum(1/sigma)

  dhb = np.append(dhb, bav)
  dhe = np.append(dhe, sigav)

  if i in np.array([0,3,6,9,12,14])+5:
    #ax.scatter(pdh[1:,0], bias,label=labels[ind])
    ax.errorbar(k, bias, yerr=sigma**0.5,fmt='.',linestyle='None' ,capsize=3, elinewidth=0.5,linewidth=None,markersize='5',label=labels[ind])

    ind = ind+1
    print k
    print bias
    ymax = bias.max()*1.2
    #ax.plot(k, bias,
    #     linestyle='-',
    #     linewidth=0.5)

ax.axvline(x=2*2*np.pi/210.,color='grey',linestyle='--')
ax.set_title('Halo bias at z=%s'%(snapnum))
ax.set_xlabel('k/h Mpc')
ax.set_ylabel('$b_h(k)$')
ax.set_ylim(0.5, ymax)
ax.set_xlim(0.001, 0.8)
#ax.set_xscale('log')
ax.legend(loc='best',frameon=False)
ax.tick_params(direction='in',which='both',
                bottom='on',
                top='on',
                left='on',
                right='on')
plt.legend(loc='best')
plt.savefig('halobias.png')
plt.show()
exit()

#plt.plot(dhm, dhb, label='Vsimu')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xlabel('Msun')
ax1.set_ylabel('bias')
ax1.tick_params(direction='in',which='both',
                bottom='on',
                top='on',
                left='on',
                right='on')

print dhm, dhb
ax1.errorbar(dhm, dhb, yerr=dhe, fmt='ko',label='velocicaptor',capsize=3, elinewidth=0.5,linewidth=None,markersize='1')
ax1.plot(hmfdat[:,0], hmfdat[:,1],label='tinker',color='black')
ax1.set_ylim(0.5, 50)
np.savetxt('Vhalo_pb.txt',np.vstack((dhm,dhb,dhe)).T)

plt.legend(loc='best')
plt.show()
plt.savefig('halobias2.png')


#dhb = []
#for i in range(15):
#  pdh = np.loadtxt('mhhalo_F_%s.txt'%(i+1))
#  #pdh = np.loadtxt('L210_N1536_%s.txt'%(i+1))
#  bias = pdh[:limx,2]/hz0[:limx,2]
#  dhb = np.append(dhb, bias.mean())
#  #plt.loglog(pdh[:-1,1], pdh[:-1,2], label='%s'%(i+1))
#
#plt.plot(dhm, dhb, label='simu')
#plt.plot(hmfdat[:,0], hmfdat[:,1],label='tink2')
