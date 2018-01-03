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


snapshot = 'snapshot_%s'%(snapnum)
fi=h5py.File(indir+'/L210_1536/'+snapshot+'.VELOCIraptor.properties.0','r')
#fi=h5py.File(indir+'/L210_N1536/'+snapshot+'.VELOCIraptor.hdf.properties.0','r')

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
  num = 0
  for i in range(fi["Num_of_files"][0]):
    fi=h5py.File(indir+'/L210_1536/'+snapshot+'.VELOCIraptor.properties.%d'%(i),'r')
    #fi=h5py.File(indir+'/L210_N1536/'+snapshot+'.VELOCIraptor.hdf.properties.%d'%(i),'r')
    #print np.log10(np.array(fi["Mvir"]).max()*10**10)
    #continue
    Xc = np.array(fi["Xc"])*h0*(1+z)
    Yc = np.array(fi["Yc"])*h0*(1+z)
    Zc = np.array(fi["Zc"])*h0*(1+z)
    #Mc = np.array(fi["Mass_200crit"])*1e10*h0
    #Mc = np.array(fi["Mass_tot"])*1e10*h0
    Mc = np.array(fi["Mass_FOF"])*1e10*h0
    #Mc = np.array(fi["Mvir"])*1e10*h0

    xnind = np.where(Xc<0.)
    xpind = np.where(Xc>210.)
    ynind = np.where(Yc<0.)
    ypind = np.where(Yc>210.)
    znind = np.where(Zc<0.)
    zpind = np.where(Zc>210.)
    Xc[xnind] = Xc[xnind]+210.0
    #Xc[xpind] = Xc[xpind]-210.0
    Yc[ynind] = Yc[ynind]+210.0
    #Yc[ypind] = Yc[ypind]-210.0
    Zc[znind] = Zc[znind]+210.0
    #Zc[zpind] = Zc[zpind]-210.0

    indice = np.where((np.array(fi["hostHaloID"])==-1) & (Mc >= a) & (Mc < b))
    #indice = np.where((np.array(fi["hostHaloID"])==-1) & (Mc < b))
    #indice = np.where((np.array(fi["hostHaloID"])==-1) & (np.array(fi["Mvir"]) >= a) & (np.array(fi["Mvir"]) < b))
    Xc = Xc[indice]
    Yc = Yc[indice]
    Zc = Zc[indice]
    Mc = Mc[indice]
    #Mc = np.array(fi["Mass_200crit"])[indice]
    pos = np.vstack((Xc,Yc,Zc,Mc)).T

    out_file.write(struct.pack('f'*pos.size, *pos.flatten()))
    num = num + len(Xc)
  #print num
  #Tm = [hbins[j]+dh/2.,num]
  Tm = [hbins[j]+dh/2.,num/(dh*volh)]
  dndlogM= np.append(dndlogM,Tm)
out_file.close()
dndlogM = dndlogM.reshape((len(dndlogM)/2,2))

#plot host halo mass function
np.savetxt(outdir+'/%s_f.txt'%(outname),dndlogM)
print "************************* DONE *************************"


