import numpy as np
from astropy.io import ascii

bands_r = ['mstars_tot','mcold','mstardot',
                 'L_tot_Halpha_ext',
                'L_tot_Lyalpha_ext',
                  'L_tot_Hbeta_ext',
                'L_tot_OII3727_ext',
          'mag_GALEX-FUV_r_tot_ext',
          'mag_GALEX-NUV_r_tot_ext',
             'mag_SDSS-u_r_tot_ext',
             'mag_SDSS-g_r_tot_ext',
             'mag_SDSS-r_r_tot_ext',
             'mag_SDSS-i_r_tot_ext',
             'mag_SDSS-z_r_tot_ext',
           'mag_UKIDSS-Y_r_tot_ext',
            'mag_UKIRT-H_r_tot_ext',
            'mag_UKIRT-J_r_tot_ext',
            'mag_UKIRT-K_r_tot_ext']

bands_o = ['mstars_tot','mcold','mstardot',
                 'L_tot_Halpha_ext',
                'L_tot_Lyalpha_ext',
                  'L_tot_Hbeta_ext',
                'L_tot_OII3727_ext',
          'mag_GALEX-FUV_o_tot_ext',
          'mag_GALEX-NUV_o_tot_ext',
             'mag_SDSS-u_o_tot_ext',
             'mag_SDSS-g_o_tot_ext',
             'mag_SDSS-r_o_tot_ext',
             'mag_SDSS-i_o_tot_ext',
             'mag_SDSS-z_o_tot_ext',
           'mag_UKIDSS-Y_o_tot_ext',
            'mag_UKIRT-H_o_tot_ext',
            'mag_UKIRT-J_o_tot_ext',
            'mag_UKIRT-K_o_tot_ext']

physicals = ['mstars_tot','mcold','mstardot',
                 'L_tot_Halpha_ext',
                'L_tot_Lyalpha_ext',
                  'L_tot_Hbeta_ext',
                'L_tot_OII3727_ext']


labels = ['ms','mcold','sfr','Halpha','Lyalpha','Hbeta','OII','FUV', 'NUV', 'u', 'g', 'r', 'i','z','Y','H','J','K']


xlabel = [r'LogM$[M_{\odot}/h]$',r'LogM$[M_{\star}/h]$',r'LogSFR$[M_{\star}/h/Gyr]$',
       r'Log$H_{\alpha}[erg/s]$',r'Log$Ly_{\alpha}[erg/s]$',r'Log$H_{\beta}[erg/s]$',
       r'Log$OII[erg/s]$','FUV', 'NUV', 'u', 'g', 'r', 'i','z','Y','H','J','K']

#xlabel = ['Log M $[M_{\odot}/h]$','log M(solar/h)','log SFR (Msolar/h/Gyr)','log Halpha(erg/s)','log Lyalpha(erg/s)','log Hbeta(erg/s)']
latexs = ['$M_{\star}$','$SFR$','$M_{gas}$',r'$H_{\alpha}$',r'$Ly_{\alpha}$',r'$H_{\beta}$',r'$O_{\Rmnum{2}}$','FUV', 'NUV', 'u', 'g', 'r', 'i','z','Y','H','J','K']
#xlow  = [9, 9, 8, 39,38,38,-24,-24,-24,-24,-24,-24,-24,-24,-24]
#xupp  = [12,11,11,43,44,44,-14,-14,-14,-14,-14,-14,-14,-14,-14]

mins = [9,  9,  9,  41, 41, 41, 41,-24,-24,-24,-24,-24,-24,-24,-24,-24,-24,-24]
maxs = [12, 11, 11, 43, 43, 43, 43,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18]


a = len(physicals)

#bands = [ i.rjust(15) for i in bands]
#labels = [ i.rjust(15) for i in labels]

#data = np.array([bands,labels,mins,maxs])
data = [physicals,labels[:a],xlabel[:a],latexs[:a],mins[:a],maxs[:a]]
#data = [bands_r,bands_o,labels,xlabel,latexs,mins,maxs]

#print data
#ascii.write(data,'bands.txt', names=['bands_r','bands_o','labels','xlabel','latexs','mins','maxs'],
#                                      formats={'bands_r': '%15s','bands_o': '%15s','labels': '>10','xlabel': '>10','latexs':'>5' ,'mins':'>5', 'maxs':'>5'},format='fixed_width',delimiter=' ')
ascii.write(data,'physicals.txt', names=['physicals','labels','xlabel','latexs','mins','maxs'],
                                      formats={'physicals': '%15s','labels': '>10','xlabel': '>10','latexs':'>5' ,'mins':'>5', 'maxs':'>5'},format='fixed_width',delimiter=' ')

