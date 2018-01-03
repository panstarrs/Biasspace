from bias_theor import growthfactor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
import sys


iarg = 1
while ( iarg <= len(sys.argv)-1):
    if( sys.argv[iarg] == '-outdir' ):
        plotdir = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-indir' ):
        indir = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-zlist' ):
        zlist = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-obs' ):
        obs = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-bands' ):
        bands = sys.argv[iarg+1]
        minb = sys.argv[iarg+2]
        maxb = sys.argv[iarg+3]
        db = sys.argv[iarg+4]
    iarg = iarg + 1


snapnum = int(zlist.split()[0])
if snapnum == 199 :  z = 0
elif snapnum == 156 :z = 1
elif snapnum == 131 :z = 2


# Host Halo Masses
hmin = 9 #5.
hmax = 16 #15.
#dh   = float(db)
dh = 0.5
hbins = np.arange(hmin,hmax,dh)


dl = float(db)
lmin = float(minb.split()[0])
lmax = float(maxb.split()[0])
#lmin = [float(i) for i in minb.split()]
#lmax = [float(i) for i in maxb.split()]
lbins = np.arange(lmin,lmax,dl)

#characteristic porperty for fitting: char
xdata = np.linspace(lmin, lmax, 100)
char = (lmin+lmax)/2.

print "growthfactor is ", growthfactor(z)
def func(X, a, b, c, d):
    x, y = X
    return (a +  (10**(0.4*(b-x)))**c)/growthfactor(z)*(1+y)**d
    #return a + b * (10**(char-x))**c

smf = np.loadtxt(indir+'/f_h/%s_f.txt'%(bands))
hsf = np.loadtxt(indir+'/f_h/%s_h.txt'%(bands))

stb = np.loadtxt(indir+'/p_c/%s_tb.txt'%(bands))
spb = np.loadtxt(indir+'/p_c/%s_pb.txt'%(bands))


# panel one
plt.figure(figsize=(8,6))

gs1 = gridspec.GridSpec(2, 1,
                       #width_ratios=[1, 3],
                       height_ratios=[1, 3]
                       )
gs1.update(left=0.1, right=0.45, hspace=0.05)

ax1 = plt.subplot(gs1[0])
ax1.set_ylabel(r'dn/dlogM($h^{3}Mpc^{-3}$)', fontsize=12)
ax1.xaxis.set_ticklabels([])
ax1.set_xlim(lbins[0], lbins[-1]+dl)
ax1.invert_xaxis()
ax1.tick_params(direction='in',
		which='both',
		axis='both',
                bottom='on',
                top='on',
                left='on',
                right='on')
ax1.plot(smf[:,0], np.log10(smf[:,1]),'black',label='%s'%(bands))
#ax1.legend(loc='best')
ax1.legend(loc='best',frameon=False)


ax2 = plt.subplot(gs1[1])
ax2.set_ylabel(r'$\log M$ $[M_{\odot}]$')

ind = np.where(hsf > 0.)
hsf[ind] = np.log10(hsf[ind])

ax2.imshow(hsf.T, interpolation='nearest',origin='low', extent=[lbins[0], lbins[-1]+dl, hbins[0], hbins[-1]+dh],aspect='auto')

ax2.tick_params(direction='in',
                bottom='on',
                top='on',
                left='on',
                right='on')


ax2.xaxis.set_visible(True)
ax2.set_xlabel('%s'%(bands))
ax2.invert_xaxis()

plt.setp(ax2.get_xticklabels(), visible=True)

# panel two
gs2 = gridspec.GridSpec(1, 1)
gs2.update(left=0.55, right=0.98)
ax3 = plt.subplot(gs2[0])
#ax3 = fig.add_subplot(gs2[0])
ax3.plot(stb[:,0], stb[:,1],'grey', linestyle='--',label='halo model prediction')
#ax3.plot(spb[:,0], spb[:,1],'black', linestyle='-',label='direct measurement')
ax3.errorbar(spb[:,0], spb[:,1], yerr=spb[:,2], fmt='ko',capsize=3, elinewidth=1,linewidth=None,label='direct measurement')

#fit function
p0 = 1.13398121, -22.39229749 ,  0.63451676 , -0.46

popt, pcov = curve_fit(func, (spb[:,0], np.full(len(spb[:,0]),z)), spb[:,1], p0)
perr = np.sqrt(np.diag(pcov))
ax3.plot(xdata, func((xdata,np.full(len(xdata),z)), *popt), 'k-', label='best fit')

ax3.set_xlim(lbins[0], lbins[-1]+dl)
ax3.set_ylabel(r'$bias$', fontsize=12)
ax3.tick_params(direction='in',
                bottom='on',
                top='on',
                left='on',
                right='on')

#ax3.annotate('local max', xy=(2, 1), xytext=(3, 1.5))
ax3.set_xlabel('%s'%(bands))
ax3.invert_xaxis()
#plt.figtext(0.45, bottom, r'$\log(%s)$'%(bands))
ax3.legend(loc='best',frameon=False)
#plt.figtext(0.5*(left+right), bottom, 'Log(%s)')

#popt = np.append(popt,-0.46)
popt = np.append(popt,perr)
print popt
popt = popt.reshape((1,len(popt)))
#f_handle = file(indir+'/parameters.txt', 'a')
f_handle = file(indir+'/%s_b.txt'%(bands), 'w')
np.savetxt(f_handle, popt, fmt='%4.2f')
f_handle.close()

#plt.show()
plt.savefig(indir+'/%s_b.png'%(bands))
