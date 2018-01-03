#import illustris_python as il
import h5py
import numpy as np
import struct
import matplotlib.pyplot as plt
import sys

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
    if( sys.argv[iarg] == '-lbox' ):
        lbox = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-h0' ):
        h0 = sys.argv[iarg+1]
    if( sys.argv[iarg] == '-bands' ):
        bands = sys.argv[iarg+1]
        minb = sys.argv[iarg+2]
        maxb = sys.argv[iarg+3]
        db = sys.argv[iarg+4]
    iarg = iarg + 1

#############################
#
# setting parameters
#indir = '/home/hpan/data/HALO/waves_512'

snapnum = int(zlist.split()[0])
if   snapnum == 199 : z=0
elif snapnum == 156 : z=1
elif snapnum == 131 : z=2

h0   = float(h0)
lbox = float(lbox)
volh = lbox**3 # In Mpc^3 h^3
#vol  = volh/pow(h0,3.) # In Mpc^3
print z, lbox, h0

#input
dhalo  = np.loadtxt('./test/dmass_mxyzc.txt')
#dhalo  = np.loadtxt('./test/dmass_com.txt')
dhmass = dhalo[:,0]
dhx    = dhalo[:,1]
dhy    = dhalo[:,2]
dhz    = dhalo[:,3]

#output
#outdir = './B%s'%(snapnum)
#outname = 'halo_%s'%(snapnum)
outname = bands

#Halo Masses
hmin = float(minb.split()[0])
hmax = float(maxb.split()[0])
dh   = float(db)
hbins = np.arange(hmin,hmax,dh)
dndlogM = []

#for j in range(1):
for j in range(len(hbins)):
  a = 10**(hbins[j])
  b = 10**(hbins[j] + dh)

  subffix = "%05.2f"%(hbins[j]+dh/2.)
  output_file = outdir+'/'+outname+'_%s'%(subffix.strip(" "))
  #output_file = outdir+'/'+outname
  print output_file
  out_file = open(output_file,"wb")

  #continue
  Xc = dhx
  Yc = dhy
  Zc = dhz
  Mc = dhmass

  indice = np.where( (Mc >= a) & (Mc < b))
  Xc = Xc[indice]
  Yc = Yc[indice]
  Zc = Zc[indice]
  Mc = Mc[indice]
  pos = np.vstack((Xc,Yc,Zc,Mc)).T

  out_file.write(struct.pack('f'*pos.size, *pos.flatten()))
  num = len(Xc)
  #print num
  #Tm = [hbins[j]+dh/2.,num]
  Tm = [hbins[j]+dh/2.,num/(dh*volh)]
  dndlogM= np.append(dndlogM,Tm)

out_file.close()
dndlogM = dndlogM.reshape((len(dndlogM)/2,2))

#plot host halo mass function
fig1 = plt.figure(figsize=(8, 6), dpi=80)
ax = fig1.add_subplot(111)

vhmf = dndlogM
ax.plot(vhmf[:,0],np.log10(vhmf[:,1]),'b',label='Velocriaptor(Mass_tot) z=%s'%(z))

hhmf = np.loadtxt('./B%s_hhalom_f.txt'%(snapnum))
thmf = np.loadtxt('./B%s_tinker10.txt'%(snapnum))
ax.plot(hhmf[:,0],np.log10(hhmf[:,1]),'green',label='dhalo')
ax.plot(thmf[:,0],np.log10(thmf[:,2]),'black',label='Tinker10')

ax.set_xlabel(r'$logM(h^{-1}M_{\bigodot})$', fontsize=15)
ax.set_ylabel(r'dn/dlogM($h^{3}Mpc^{-3}$)', fontsize=15)
#ax.set_yscale('log')
ax.set_xlim(9,16)

ax.grid(b=True, which='major', color='grey', linestyle='-.')
ax.legend(loc='best')

fig1.savefig('B%s_dhmf_tot.png'%(snapnum))

print "************************* DONE *************************"


