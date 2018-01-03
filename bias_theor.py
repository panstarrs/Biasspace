from hmf import cosmo
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hmf import MassFunction, growth_factor
from astropy.cosmology import LambdaCDM
import scipy
import os

delta_c = 1.686
h0 = 0.6751
new_model=LambdaCDM(H0=67.51, Om0=0.3121,Ode0=0.6879,Ob0=0.0491)

def sn_to_z(snapnum):
    if snapnum   == 199 :z = 0
    elif snapnum == 156 :z = 1
    elif snapnum == 131 :z = 2
    elif snapnum == 113 :z = 3
    return z


def growthfactor(z):
    gf = growth_factor.GrowthFactor(new_model)
    if isinstance(z,(int,float)):
        return gf.growth_factor(z)
    else:
        gfs = np.full(len(z),1.0)
        #gfs[np.where(z==0)] = 1.0
        #gfs[np.where(z==1)] = gf.growth_factor(1)
        #gfs[np.where(z==2)] = gf.growth_factor(2)
        gfs[np.where(z==0)] = 1.0
        gfs[np.where(z==1)] = 0.607372387536
        gfs[np.where(z==2)] = 0.417471529031
        gfs[np.where(z==3)] = 0.315501130698
        return gfs

def continuum_func(X, a, b, c, d, e):
    x, y = X
    return a + b*(1+y)**e*(1+np.exp((c-x)*d))
    #return   a +  10**((b-x)*c)
    #return  a*( b*(1+y)**d +  10**(c-x))/growthfactor(y)**e
    #return  ( a +  10**((b-x)*c))/growthfactor(y)**d
    #return  a*(1 + (1+y)*10**(b-x))/growthfactor(y)**e
    #return  a*( 1 + 10**((b-x)*c))/growthfactor(y)**d
    #return  a*( (1+y)**d +  10**((b-x)*c))/growthfactor(y)**e
    #return  a*( 1 +  10**((b-x)*c))/growthfactor(y)*(1+y)**d


def physical_func(X, a, b, c,d, e):
    x, y = X
    return a + b*(1+y)**e*(1+np.exp((x-c)*d))
    #return  a*( 1 +  10**((x-b)*c))/growthfactor(y)*(1+y)**d
    #return  a*( (1+y)**d +  10**((x-b)*c))/growthfactor(y)**e
    #return  a*( 1 +  10**((x-b)*c*growthfactor(y)**d))/growthfactor(y)**e
    #return  a*( 1 +  10**((x-b*(1+y)**d)*c))*growthfactor(y)**e
    #return  a*( 1 +  10**((x-b)*c))
    #return  a*( c +  10**(x-b))

def hhmf(Min,Max,dlm,fitfunc,Z):

    mf = MassFunction()
    #Min = np.log10(10**Min*h0)
    #Max = np.log10(10**Max*h0)
    mf.update(cosmo_model=new_model,n=0.9653,sigma_8=0.8150,delta_wrt='crit',dlog10m=dlm,Mmin=Min,Mmax=Max, z=Z) #update baryon density and redshift
    #mf.update(transfer_model="FromFile",transfer_params={"fname":'./waves_input_tk.dat'},hmf_model="Tinker10")
    mf.update(transfer_model="FromFile",transfer_params={"fname":'./camb_transfer_out.dat'},hmf_model=fitfunc)
    #mf.update(transfer_model="FromFile",transfer_params={"fname":'./camb_transfer_out.dat'},hmf_model="Tinker10")

    return mf.m, mf.sigma, mf.dndlog10m

def bias(nu):
    Delta = 200
    y = np.log10(Delta)
    A = 1.0 + (0.24*y*np.exp(-((4/y)**4)))
    a = (0.44*y) - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + (0.107*y) + (0.19*np.exp(-((4/y)**4)))
    c = 2.4
    F = 1-(A*((nu**a)/((nu**a)+(delta_c**a))))+(B*(nu**b))+(C*(nu**c))
    return F

def interptb(x1,x2,x3,x4,x5,M):
    m, x , dndlog = hhmf(x1,x2,x3,x4,x5)
    y = bias(delta_c/x)
    tck = scipy.interpolate.splrep(m, y)
    bias_M = scipy.interpolate.splev(M, tck)
    return bias_M

def interpfb(filepath,X,M):
    fit = np.loadtxt(filepath+'/mhhalo_0.2.txt')
    popt = fit[:5]

    m, z = X
    y = physical_func((m,np.full(len(m),z)), *popt)
    tck = scipy.interpolate.splrep(m, y)
    bias_M = scipy.interpolate.splev(M, tck)
    return bias_M

def interpgb(filepath,M):
    print filepath+'/mhhalo_pb.txt'
    if os.path.isfile(filepath+'/mhhalo_pb.txt'):
      #biass = np.genfromtxt(filepath+'/vhalo_F_xb.txt')
      biass = np.genfromtxt(filepath+'/mhhalo_pb.txt')
      m = biass[:,0]
      y = biass[:,1]
      m = m[np.nonzero(~np.isnan(y))]
      y = y[np.nonzero(~np.isnan(y))]
      #m = [9.9, 10.1, 10.3,10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.7, 11.9, 12.1, 12.3,
      #12.5, 12.7, 12.9, 13.1,13.3, 13.5, 13.7, 13.9, 14.1, 14.3, 14.5]
      #y = [0.8, 0.78, 0.81, 0.82, 0.87, 0.87, 0.89, 0.87, 0.88, 0.97, 0.93, 0.98, 1.04,
      #1.06, 1.24, 1.21, 1.5, 1.78, 2.1, 2.62, 3.37, 4.01, 5.29, 7.6]
      tck = scipy.interpolate.splrep(m, y)
      bias_M = scipy.interpolate.splev(M, tck)
      return bias_M
    else:
      print('file mhhalo_pb.txt does not exist')
      return np.array([0,0])

if __name__ == '__main__':

    for z in [0,1,2,3]:
        for fitfunc in ['ST','Tinker10']:
            if z == 0: snapnum = 199
            if z == 1: snapnum = 156
            if z == 2: snapnum = 131
            if z == 3: snapnum = 113

            Min= 10; Max =14; dlm = 0.2
            #xdata = np.linspace(Min, Max, 100)
            #outdir = './output/B%s/H_%s'%(snapnum,dlm)
            outdir = './output/B%s/T'%(snapnum)
            #fitfunc='ST'
            #fitfunc = 'Tinker10'
            m, sigma_m, dndlog = hhmf(Min+dlm/2.,Max,dlm,fitfunc,z)
            bias_m = bias(delta_c/sigma_m)
            #bias_x = interptb(Min+dlm/2.,Max,dlm,fitfunc,z, xdata)
            tinker = [sigma_m,dndlog,bias_m]
            fig1 = plt.figure(figsize=(16, 6), dpi=80)
            ytit = ['$\sigma(z,M)$','$dn$/$d \, \log M$ $\, [h^3 \, Mpc^{-3}]$','$bias(M)$']

            for i in range(len(tinker)):
                ax = fig1.add_subplot(131+i)
                ax.plot(m, tinker[i],'b',label='z=0')
                #ax.set_title('Halo Bias', fontsize=16)
                ax.set_xlabel('$M$ $[M_{\odot}]$', fontsize=15)
                ax.set_ylabel(ytit[i], fontsize=15)
                #ax.set_ylabel('$b(z,M)$', fontsize=15)
                if i == 1 : ax.set_yscale('log')
                ax.set_xlim(10**9,10**16)
                #ax.set_ylim(0,10)
                ax.set_xscale('log')
                ax.grid(b=True, which='major', color='grey', linestyle='-.')
                ax.legend(loc='best')

            np.savetxt(outdir+'/%s.txt'%(fitfunc),np.vstack((np.log10(m), dndlog, sigma_m, bias_m)).T)
            #np.savetxt('B%s_tinker10.txt'%(snapnum),np.vstack((np.log10(m),sigma_m, dndlog, bias_m)).T)
            #np.savetxt('B%s_tinker10_wave.txt'%(snapnum),np.vstack((np.log10(m),sigma_m, dndlog, bias_m)).T)
            fig1.tight_layout()
            fig1.savefig(outdir+'/%s.png'%(fitfunc))
            #fig1.savefig('B%s_tinker10.png'%(snapnum))
    print "well done !!!"

