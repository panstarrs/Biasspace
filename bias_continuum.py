import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
import sys
import math
from numpy import genfromtxt
from astropy.io import ascii
from readfilters import read_filters
from scipy.interpolate import splev, splrep
from bias_theor import sn_to_z, growthfactor, continuum_func, physical_func
import matplotlib.cm as cm

indirs = '/Users/hengxingpan/Work/danail/ProCorr/Bias/output'
cso = ['centra','sat','orp','sat_cen','orp_sat','orp_sat_cen']
cso = cso[3]
hl = 0
sl = 0

snapnum = [199,156,131,113]
sncolors = ['black','red','blue','green']
fmts = ['ko','ro','bo','go']
linew = [0.5,0.5,0.5,0.5]
zlabels = ['z=0','z=1','z=2','z=3']
zx = 0.05
zy = np.array([0.6,0.5,0.4,0.3])+10

db = 0.5
dl = db
dim = 5 # number of parameters
latexname = ['Bands','a','b','c','d','e']
latexname = latexname[:dim+1]

data = ascii.read('/Users/hengxingpan/Work/danail/ProCorr/Bias/input/bands.txt')
rows = 3

#data = ascii.read('/Users/hengxingpan/Work/danail/ProCorr/Bias/input/bands1.txt')
#rows = 1

#data = ascii.read('/Users/hengxingpan/Work/danail/ProCorr/Bias/input/bands2.txt')
#rows = 2

lbands = data['bands_r'].data
labels = data['labels'].data
xlabel = data['xlabel'].data
latexs = data['latexs'].data
xlow = data['mins'].data
xupp = np.loadtxt('/Users/hengxingpan/Work/danail/ProCorr/Bias/input/allc_con.txt')
#xupp = data['maxs'].data

#colors = ['darkviolet','blueviolet','violet','green','red','darkred','indianred','firebrick','lightcoral']
#centralw = [1528,2271,3543,4770,6231,7625]


#xlow = [-24,-24,-24,-24,-24,-24,-24,-24,-24,-24]
#xlow = np.full(len(labels),float(-25))
#xupp = [-18,-18,-18,-18,-18,-18,-18,-18,-18,-18]

fitplot = True

bestfit = [[]]*len(lbands)
fiterro = [[]]*len(lbands)
plusmin = [['$\pm$']]*len(lbands)*dim

#############################
# plot log-log histogram2d
#lbands = lbands[:3]


cols = int(math.ceil(len(lbands) / float(rows)))
gs   = gridspec.GridSpec(rows, cols)
#ytit = ['log(M/Msun)','log(M/Msun)','log(Halpha L)','Mag']
fig  = plt.figure(figsize=(4*cols,4*rows))

#fig.suptitle(label1+' '+label2)

for index, iband in enumerate(lbands):

  xdata = np.linspace(xlow[index], -17, 100)
  #xdata = np.linspace(xlow[index], xupp[index], 100)
  fit = genfromtxt(indirs+'/%s_%s.txt'%(iband,db))
  popt = np.around(fit[:dim],2)
  perr = np.around(fit[dim:],2)
  print '%s'%(iband)
  print popt
  print perr
  bestfit[index] = popt
  fiterro[index] = perr

  ax = fig.add_subplot(gs[index])
  ax.set_xlim(xlow[index], xupp[index]+dl)
  ax.set_ylim(0.5, 5)
  #ax.tick_params(labelsize=13)

  ax.set_ylabel(r'$bias$', fontsize=15)
  ax.set_xlabel("${{\\rm M_{AB}("+labels[index]+")}\, -\, 5log{\\rm h}}$")
  ax.invert_xaxis()

  ax.tick_params(direction='in',
                bottom='on',
                top='on',
                left='on',
                right='on')
  #ax.text(xy=(, 0), "Direction", ha="center", va="center", rotation=45,
  #          size=15,
  #          bbox=bbox_props)

  ax.annotate(labels[index], xy=(0.35, 0.8), xycoords="axes fraction",color='black',size=15)

  #ax.annotate(labels[index], xy=(0.4, 0.6), xycoords="axes fraction",color='black',size=15)
  #ax.annotate(labels[index], xy=(0.6, 0.6), xycoords="axes fraction",color=colors[index],size=15)

  #ax.plot(stbo[:,0],stbo[:,1],'blue', linestyle='--')
  #ax.errorbar(spbo[:,0], spbo[:,1], yerr=spbo[:,2], fmt='o',color='blue',ecolor='blue',capsize=3, elinewidth=1,linewidth=None)
  for i in range(len(snapnum)):

    z = sn_to_z(snapnum[i])

    stbs = np.loadtxt(indirs+'/B%s/G_%s/%s/p_c/%s_t_b%s%s.txt'%(snapnum[i],dl,cso,iband,hl,sl))
    spbs = np.loadtxt(indirs+'/B%s/G_%s/%s/p_c/%s_p_b%s%s.txt'%(snapnum[i],dl,cso,iband,hl,sl))
    if index == 0 :
      ax.annotate(zlabels[i], xy=(zx, zy[i]), xycoords="axes fraction",color=sncolors[i],size=10)
    if snapnum[i] == 199 :
      ax.plot(stbs[:,0],stbs[:,1],color=sncolors[i], linestyle='--',label='halo model prediction',linewidth=linew[i])
      ax.errorbar(spbs[:,0], spbs[:,1], yerr=spbs[:,2], fmt=fmts[i],ecolor=sncolors[i],capsize=3, elinewidth=linew[i],linewidth=None,label='direct measurement',markersize='2')
      if fitplot == True: ax.plot(xdata, continuum_func((xdata,np.full(len(xdata),z)), *popt), color=sncolors[i], label='best fit',linewidth=linew[i])
    else :
      ax.plot(stbs[:,0],stbs[:,1],color=sncolors[i], linestyle='--',linewidth=linew[i])
      ax.errorbar(spbs[:,0], spbs[:,1], yerr=spbs[:,2], fmt=fmts[i],ecolor=sncolors[i],capsize=3, elinewidth=linew[i],linewidth=None,markersize='2')
      if fitplot == True: ax.plot(xdata, continuum_func((xdata,np.full(len(xdata),z)), *popt), color=sncolors[i],linewidth=linew[i])

  #if index == 0: ax.legend(loc='upper right',frameon=False)


abcd = np.array(bestfit).reshape((len(lbands),dim))
abcd = np.transpose(abcd)
abcd_e = np.array(fiterro).reshape((len(lbands),dim))
abcd_e = np.transpose(abcd_e)

bestfit = np.array(bestfit).reshape((len(lbands),dim)).astype(str)
fiterro = np.array(fiterro).reshape((len(lbands),dim)).astype(str)
plusmin = np.array(plusmin).reshape((len(lbands),dim))
combine = np.core.defchararray.add(bestfit,plusmin)
combine = np.core.defchararray.add(combine,fiterro)

ascii.write(np.column_stack((latexs,combine)), indirs+'/continuum_parameters_latex.txt',format='latex', names=latexname,overwrite=True)
fig.tight_layout()
fig.savefig(indirs+'/continuum_b.png')
plt.show()

#exit()

#need filter below for effective wavelength
filters = read_filters('filters_feb16.dat')

iband_est = 'mag_SDSS-z_r_tot_ext'
inband_est = 205 #V band 52 -1

xlow_est  = -24
xupp_est  = -14

inband = [294,293,201,202,203,204,46,47,48,inband_est]
#plot a b c d against wavelength and make existing filter responce curves
dc = 25 # just for better look plot
para = ['a','b','c+%smag'%(dc),'d','e']
symb = ['o','*','+','D','^']


f = plt.figure(figsize=(12,8), dpi=100)
ax = f.add_subplot(111)
xmax = 23000
xmin = 1528

x = np.linspace(xmin, xmax, xmax-xmin+1)
colors = cm.rainbow(np.linspace(0, 1, len(x)))
colormap = dict(zip(x, colors))

centralw = [] # all wavelength

b_fit = genfromtxt(indirs+'/%s_%s.txt'%(iband_est,db))
print 'estimation', np.around(b_fit,3)
popt = np.around(b_fit[:dim],2)
perr = np.around(b_fit[dim:],2)

for i in range(len(inband)):
    lamda = filters[inband[i]]['wavelength']
    respo = filters[inband[i]]['response']
    delta = lamda[1]-lamda[0]

    lamdatgd = lamda*respo*delta
    lamdatgl = lamdatgd.sum()
    tgrand = respo*delta
    tgral = tgrand.sum()
    centw = lamdatgl/tgral

    if i < len(inband)-1:
        centralw = np.append(centralw,centw)
        ax.plot(lamda,respo , color=colormap[int(centralw[i])],linestyle='--',linewidth=0.5, label=labels[i])
    else:
        ax.plot(lamda,respo , color=colormap[int(centw)],linestyle='--',linewidth=0.5)
        #ax.plot(lamda,respo , color=colors[i],linestyle='--',linewidth=0.5, label=labels[i])

print 'centralw ',np.around(centralw,0)
lw = 0.2
for i in range(len(abcd)):
    print 'abcd_in%s'%(i),abcd[i,:], abcd[i,:].mean(), abcd[i,:].std()

    spl = splrep(centralw, abcd[i,:])
    y = splev(x, spl)
    if i == 1:
        lfit = np.polyfit(centralw,abcd[i,:],1)
        lfit_f = np.poly1d(lfit)
        print lfit
        ax.plot(x,lfit_f(x)+dc, color='black',linestyle='-',linewidth=0.5)
        ax.scatter(centralw, abcd[i,:]+dc, label=para[i], color='black', marker=symb[i])
        ax.errorbar(centralw, abcd[i,:]+dc, yerr=abcd_e[i,:],color='black', capsize=3, elinewidth=0.5,linewidth=lw)
        ax.errorbar([centw],[popt[i]+dc], yerr=[perr[i]], capsize=3, elinewidth=0.5,linewidth=lw,color=colormap[int(centw)])
        #ax.scatter(centw,popt[i]+25, color=colormap[int(centw)], marker=symb[i]) #est
        #ax.scatter([centw],[popt[i]+dc], color=colormap[int(centw)], marker=symb[i]) #est
        #ax.scatter(centw,popt[i]+25, color=colormap[int(centw)], marker=symb[i]) #est
        #ax.scatter(centw,splev(centw, spl)+25,label=para[i], color='red', marker=symb[i])
    else:
        #ax.plot(x,np.full(len(y),abcd[i,:].mean()), color='black',linestyle='-',linewidth=0.5)
        #ax.plot(x,y, color='black',linestyle='-',linewidth=0.5)
        ax.errorbar(centralw, abcd[i,:], yerr=abcd_e[i,:],color='black', capsize=3, elinewidth=0.5,linewidth=lw)
        ax.errorbar([centw],[popt[i]], yerr=[perr[i]], capsize=3, elinewidth=0.5,linewidth=lw,color=colormap[int(centw)])
        ax.scatter(centralw, abcd[i,:], label=para[i], color='black', marker=symb[i])
        #ax.scatter(centw,popt[i], color=colormap[int(centw)], marker=symb[i])
        #ax.scatter(centw,splev(centw, spl),label=para[i], color='red', marker=symb[i])

ax.annotate(iband_est, xy=(0.2, 0.9), xycoords="axes fraction",color=colormap[int(centw)],size=15)
ax.set_xlabel("Wavelength (Angstroms)")
ax.set_ylabel("parameters")
#ax.set_xlim(min_wavelength, max_wavelength + (max_wavelength-min_wavelength) * 0.3)
ax.set_xlim(1300,25000)
ax.set_ylim(0, 5)
ax.legend(loc='upper right',frameon=False, ncol = 5)
ax.tick_params(direction='in',
                bottom='on',
                top='on',
                left='on',
                right='on')

#axgca = plt.gca().add_artist(first_legend)
#leg = axgca.get_legend()
#leg.legendHandles[0].set_color('black')
#leg.legendHandles[1].set_color('black')
#leg.legendHandles[2].set_color('black')


plt.show()
f.savefig(indirs+'/parameters.png')


# making estimation of ongoing and upcoming surveys
f = plt.figure(figsize=(5,4), dpi=100)
ax = f.add_subplot(111)

lamda_est = filters[inband_est]['wavelength']
respo_est = filters[inband_est]['response']
delta_est = lamda_est[1]-lamda_est[0]
tgrand_est = respo_est*delta_est
tgral_est = tgrand_est.sum()

p = [[]]*dim # interoporlation methods to get parameters
lp = [] # linear parameters

for k in range(len(abcd)):
  spl = splrep(centralw, abcd[k,:])
  p[k] = splev(lamda_est, spl)
  lp = np.append(lp,abcd[k,:].mean())

#def func_est(bx, bz, e, a, b, c, d):
#  return e + (a +  10**((b-bx)*c))/growthfactor(bz)*(1+bz)**d

for i in range(len(snapnum)):

  if snapnum[i]   == 199 :z = 0
  elif snapnum[i] == 156 :z = 1
  elif snapnum[i] == 131 :z = 2
  elif snapnum[i] == 113 :z = 3

  stbs = np.loadtxt(indirs+'/B%s/G_%s/%s/p_c/%s_t_b%s%s.txt'%(snapnum[i],dl,cso,iband_est,hl,sl))
  spbs = np.loadtxt(indirs+'/B%s/G_%s/%s/p_c/%s_p_b%s%s.txt'%(snapnum[i],dl,cso,iband_est,hl,sl))
  #stbs = np.loadtxt(indirs+'/B%s/G_1/%s/p_c/%s_tb.txt'%(snapnum[i],cso,iband_est))
  #spbs = np.loadtxt(indirs+'/B%s/G_1/%s/p_c/%s_pb.txt'%(snapnum[i],cso,iband_est))
  sest = []

  bx = spbs[:,0]
  by = spbs[:,1]
  bs = spbs[:,2]

  for j in range(len(bx)):
    bgrand = continuum_func((bx[j],z), *lp )*respo_est*delta_est
    #bgrand = continuum_func((bx[j],z), lp[0], lp[1],lfit_f(lamda_est),lp[3], lp[4])*respo_est*delta_est
    #bgrand = continuum_func((bx[j],z), p[0], p[1],p[2],p[3], p[4])*respo_est*delta_est
    #bgrand = func_est(bx[j],z, p[0], p[1],p[2],p[3], p[4])*respo_est*delta_est
    bgral = bgrand.sum()
    b_est = bgral/tgral
    sest = np.append(sest,b_est)

  ax.plot(stbs[:,0],stbs[:,1],color=sncolors[i], linestyle='--',label='halo model prediction at %s'%(snapnum[i]),linewidth=0.5)
  ax.plot(bx, continuum_func((bx,np.full(len(bx),z)),*popt), color=sncolors[i], label='b_fit at %s'%(snapnum[i]),linewidth=0.5)
  ax.plot(bx, sest, color=sncolors[i], label='b_est at %s'%(snapnum[i]),linewidth=1.0)
  ax.errorbar(spbs[:,0], spbs[:,1], yerr=spbs[:,2], fmt=fmts[i],ecolor=sncolors[i],capsize=3, elinewidth=0.5,linewidth=None,label='direct measurement at %s'%(snapnum[i]))

#ax.tick_params(labelsize=13)
ax.set_xlim(bx.min(),bx.max())
ax.set_ylim(0.5, 4)
ax.set_ylabel(r'$bias$', fontsize=15)
ax.set_xlabel('%s'%(iband_est))
ax.invert_xaxis()
ax.tick_params(direction='in',
              bottom='on',
              top='on',
              left='on',
              right='on')
ax.annotate(iband_est, xy=(0.25, 0.6), xycoords="axes fraction",color='black',size=15)
#ax.legend(loc='best',frameon=False)


#ax.text(xy=(, 0), "Direction", ha="center", va="center", rotation=45,
#          size=15,
#          bbox=bbox_props)

plt.show()
f.savefig(indirs+'/predicts.png')



'''
#gs1 = gridspec.GridSpec(1, 1
                       #width_ratios=[1, 3],
                       #height_ratios=[1, 3]
#                       )
#gs1.update(left=0.1, right=0.45, hspace=0.05)

# panel two
gs2 = gridspec.GridSpec(1, 1)
#gs2.update(left=0.55, right=0.98)
ax3 = plt.subplot(gs2[0])
#ax3 = fig.add_subplot(gs2[0])
ax3.plot(stb[:,0], stb[:,1],'grey', linestyle='--',label='halo model prediction')
#ax3.plot(spb[:,0], spb[:,1],'black', linestyle='-',label='direct measurement')
ax3.errorbar(spb[:,0], spb[:,1], yerr=spb[:,2], fmt='ko',capsize=3, elinewidth=1,linewidth=None,label='direct measurement')
#fit function
popt, pcov = curve_fit(func, spb[:,0], spb[:,1])
perr = np.sqrt(np.diag(pcov))
ax3.plot(xdata, func(xdata, *popt), 'k-', label='best fit')

ax3.set_xlim(lbins[0], lbins[-1]+dl)
ax3.set_ylim(0, 3)
ax3.set_ylabel(r'$bias$', fontsize=15)
#ax3.annotate('local max', xy=(2, 1), xytext=(3, 1.5))
ax3.set_xlabel('%s'%(bands))
ax3.invert_xaxis()
#plt.figtext(0.45, bottom, r'$\log(%s)$'%(bands))
ax3.legend(loc='best',frameon=False)
'''
