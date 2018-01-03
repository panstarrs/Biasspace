path.bias = '/Users/hengxingpan/Work/danail/ProCorr/Bias'
#sn = c(156)  #snapshot c(199 174 156 131) :redshift 0 0.5 1 2 # no 131 for dark halo
sn = c(199,156,131,113)  #snapshot c(199 174 156 131) :redshift 0 0.5 1 2 # no 131 for dark halo
#sn = sn[c(4)]
cso = c('centra','sat_cen','orp_sat_cen')
cso = cso[2]

par(mfcol=c(2,1))
#for plotting bias
yrange=c(0,30)
ap=3; bp=12;

runxyz = F
run_ps = F

run_ls = F #non linear least squares
run_ml = T #maximum likihoods


psuffix = "*_p.txt"
xsuffix = "*_x.txt"

#bins = c(3:9) #1,2,3,$4 #5:13
#bins = c(6,7,8) #1,2,3,$4 #5:13
bins = c(9:length(xlow)) #1,2,3,$4 #5:13
#bins  = c(3:9) #1,2,3,$4 #5:13
xlow  = c(10, 11,  8,  9,  7,  39, 39, 39, 39, -23,-23,-23,-23,-23,-23,-23,-23,-23,-23,-23)
xupp  = c(14, 14, 12, 11, 11, 43, 43, 43, 43, -14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14)

bands = c('Vhalo_F','mhhalo','mstars_tot','mcold','mstardot','L_tot_Halpha_ext','L_tot_Lyalpha_ext','L_tot_Hbeta_ext','L_tot_OII3727_ext',#8
          'mag_GALEX-FUV_r_tot_ext','mag_GALEX-NUV_r_tot_ext','mag_SDSS-u_r_tot_ext',#11
          'mag_SDSS-g_r_tot_ext','mag_SDSS-r_r_tot_ext','mag_SDSS-i_r_tot_ext','mag_SDSS-z_r_tot_ext',#15
          'mag_UKIDSS-Y_r_tot_ext','mag_UKIRT-H_r_tot_ext','mag_UKIRT-J_r_tot_ext','mag_UKIRT-K_r_tot_ext')#19

rlow = c(10, 11,  8,  9,  7,  38, 38, 38, 38,-24,-24,-24,-24,-24,-24,-24,-24,-24,-24,-24)
rupp = c(14, 14, 12, 11, 11,  43, 44, 44, 44,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14)
#dx   = c(0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
dx   = c(0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)

example <- function() {

  #parameters of computting bias
  #if (k != 4 || k != 5) next
  for (i in 1:length(bands)) {
    
    if (!(is.element(i,bins)) ) next
    xrange = c(xlow[i],xupp[i])
    xvar = seq(xlow[i],xupp[i]-dx[i],dx[i])+dx[i]/2
    rvar = seq(rlow[i],rupp[i]-dx[i],dx[i])+dx[i]/2
    dput(xvar)

    #power spectrum
    xlim=c(0.02,0.3);ylim=c(10^2,10^7)
    plot(xlim, ylim, type="n",log='y',xlab='Wavevector k [(simulation units)'^'-1'~']',ylab="Powerspectrum p(k)")
    title(paste("Power_s",bands[i],sep = " "))
    for (j in 1:length(sn)) {
      
      #if (sn[j] != 156 ) next
      opts = sprintf("'%s' '%s' '%s' '%s' '%s' '%s'",bands[i],rlow[i],rupp[i],dx[i],sn[j],cso)
      optp = sprintf("'%s' '%s' '%s' '%s'",bands[i],dx[i],sn[j],cso)
      
      print(optp)
      
      if (substr(bands[i],1,1) == 'V') {
        if (runxyz) runsh(path.bias,'runhalo.sh',opts)
        if (run_ps) runsh(path.bias,'runproh.pb',optp)
        path.sn = paste(path.bias,'/output/B',sn[j],'/H_',dx[i],sep = '')
      } else {
        if (runxyz) runsh(path.bias,'rungalf.sh',opts)
        if (run_ps) runsh(path.bias,'runprog.pb',optp)
        path.sn = paste(path.bias,'/output/B',sn[j],'/G_',dx[i],'/',cso,'/p_c',sep = '')
      }
  
      print(path.sn)
      setwd(path.sn)
      show.powerspectrum(path.sn,j,bands[i],psuffix,ap,bp)
    }
    
    #galaxy bias from powerspectrum
    plot(xrange, yrange, type="n",xlab=bands[i],ylab="galaxy bias from power_s")
    x = numeric();y = numeric();s = numeric();z = numeric() 
    for (j in 1:length(sn)) {
      
      #if (sn[j] != 156 ) next
      opts = sprintf("'%s' '%s' '%s' '%s' '%s' '%s'",bands[i],xlow[i],xupp[i],dx[i],sn[j],cso)
      if (substr(bands[i],1,1) == 'V') {
        path.sn = paste(path.bias,'/output/B',sn[j],'/H_',dx[i],sep = '')
      } else {
        path.sn = paste(path.bias,'/output/B',sn[j],'/G_',dx[i],'/',cso,'/p_c',sep = '')
      }

      setwd(path.sn)
      rp = powerspectrum(path.sn,j,bands[i],psuffix,ap,bp)
      show.biass(bands[i],rp,rvar,j,'pb')
      #if (substr(bands[i],1,1) == 'V') next
      if (substr(bands[i],1,3) == 'mag') {
        #runsh(path.bias,'runplotr.sh',opts)
      } else {
        #runsh(path.bias,'runplotn.sh',opts)
      }
      
      # bias.fit inputs
      input1 = read.table(paste('./',bands[i],"_p_b00.txt",sep = ""),skip=0,sep=' ')
      #input1 = read.table(paste('./',bands[i],"_p_b118.txt",sep = ""),skip=0,sep=' ')
      input1 = input1[xlow[i]< input1[,1] & input1[,1] <xupp[i],]
      x = append(x,input1[,1])
      y = append(y,input1[,2])
      s = append(s,input1[,3])
      z = append(z,rep(sn_to_z(sn[j]),length(input1[,1])))
      
    }
    print(z)
    opts = sprintf("%s %s %s",bands[i],dx[i],cso)
    for (j in 1:length(sn)) {
      #opts = sprintf("%s %s %s %s",bands[i],dx[i],cso,sn[j])
      #if (run_ls) runpy(path.bias,'bias_lsfit.py',opts)
      #if (run_ml) bias.fit(bands[i],dx[i],x,y,s,z)
    }
    if (run_ls) runpy(path.bias,'bias_lsfit.py',opts)
    if (run_ml) bias.fit(bands[i],dx[i],x,y,s,z)
  }
}  

runsh <- function(path.sh,filename,options) {
  system(paste0('cd ',path.sh,'; sh ',filename,' ',options),intern=FALSE)
}

runpy <- function(path.py,filename,options) {
  system(paste0('cd ',path.py,'; python ',filename,' ',options),intern=FALSE)
}

sn_to_z <- function(snapnum) {
  if ( snapnum == 199 ) {     redshift = 0
  } else if (snapnum == 156) {redshift = 1
  } else if (snapnum == 131) {redshift = 2
  } else if (snapnum == 113) {redshift = 3
  }
  return(redshift)
}

show.powerspectrum <- function(path.procorr,j,bands,psuffix,a,b) {
  
  input = list.files(path.procorr,pattern = glob2rx(paste(bands,psuffix,sep = "")))
  if (substr(bands,1,3) == 'mag') input = rev(input)
  input = c(input,paste(path.bias,'/output/B',sn[j],'/P_256/L210_N512_',sn[j],'_p.txt',sep = ""))
  print(input)
  ntrees <- length(input)
  colors <- rainbow(ntrees-1)
  colors = c(colors,"black")
  linetype <- c(rep(j,ntrees))
  
  r <- matrix(0, ntrees, b-a+1)
  # add lines 
  for (i in 1:ntrees) { 
    dat = read.table(input[i],skip=0,sep="")
    lines(dat[,1], dat[,2], type="o", lwd=1, lty=linetype[i], col=colors[i], cex=0.5)
  }
}

powerspectrum <- function(path.procorr,j,bands,psuffix,a,b) {
  
  input = list.files(path.procorr,pattern = glob2rx(paste(bands,psuffix,sep = "")))
  #str_sub(input[1],-11,-7)
  if (substr(bands,1,3) == 'mag') input = rev(input)
  input = c(input,paste(path.bias,'/output/B',sn[j],'/P_256/L210_N512_',sn[j],'_p.txt',sep = ""))
  ntrees <- length(input)
  r <- matrix(0, ntrees, b-a+1)
  # add lines 
  for (i in 1:ntrees) { 
    dat = read.table(input[i],skip=0,sep="")
    r[i,] = dat[a:b,2]
  }
  return(r)
}

show.biass <- function(bands,r,xvar,j,suffix) {
  
  ntrees = nrow(r)-1
  colors <- rainbow(ntrees)
  #print(ntrees)
  br <- r[1:ntrees,]
  
  for (i in 1:ntrees) {br[i,]=br[i,]/r[ntrees+1,]}
  bias = apply(br^0.5, 1, function(x) mean(na.omit(x)))
  bsd  = apply(br^0.5, 1, function(x) sd(na.omit(x)))
  
  # plot bias
  dput(round(bias,2))
  for (i in 1:ntrees) {points(xvar[i], bias[i], type="p",pch=j,col=colors[i], cex=0.5)}
  #arrows(xlab, bias-bsd-0.0001, xlab, bias+bsd+0.0001, length=0.02, angle=90, code=3)
  abline(h=1,lty=2)
  #title("galaxy bias from direct measure")
  #outputb = paste('./',bands,'_',suffix,".txt",sep = "")
  #print(cbind(xvar,bias,bsd))
  #write.table(cbind(xvar,bias,bsd), outputb,row.names=F,col.names=F)
}

bias.fit <- function(bands,dx,x,y,s,z) {
  #print(growthfactor(z))
  #input2 = read.table(paste(path.bias,'/output/',bands,"_",dx,".txt",sep = ""),skip=0,sep=' ')

  #ptrue = input2[1:5]
  #ptrue = c(0.93, 104571573527, 0.89 , 0.0016)
  ptrue = c(1, 0.1, mean(x), 0.5*sign(cor(x,y)), 1.5)
  print(ptrue)
  print(x)
  print(y)
  
  neglogL = function(p) {
    #return(sum((y-fitf(bands,x,z,p))^2/s^2+log(2*pi*s^2))/2)
    return(sum((y-fitf(bands,x,z,p))^2/2/(s^2+0.01)))
    #return(sum((b.observed-b.model)^2/2/(b.sigma^2+sigma^2)))
  }
  #opt = optim(ptrue,neglogL, hessian = TRUE)
  opt = optim(ptrue,neglogL,control = list(reltol = 1e-13, maxit = 1e5), hessian = TRUE)
  #opt = optim(ptrue,neglogL,control = list(parscale=abs(ptrue)), hessian = TRUE)
  pfit = opt$par
  #C = solve(opt$hessian,tol=1e-90)
  C = solve(opt$hessian)
  pfit.sigma = sqrt(diag(C))
  outputp = paste(path.bias,'/output/',bands,"_",dx,".txt",sep = "")
  write.table(t(c(pfit,pfit.sigma)), outputp,row.names=F,col.names=F)
  options("scipen"=-100, "digits"=4)
  print(c(pfit,pfit.sigma))
  options(scipen = 999)
  
}

growthfactor <- function(z) {
  gf = c(rep(1,length(z)))
  gf[z == 0] = 1
  gf[z == 1] = 0.607372387536
  gf[z == 2] = 0.417471529031
  gf[z == 3] = 0.315501130698
  return(gf)
}

fitf <- function(bands,x,z,p) {
  if (substr(bands,1,3) == 'mag') {
    return(p[1]+p[2]*(1+z)^p[5]*(1+exp((p[3]-x)*p[4])))
    #return(p[1]+(p[2]+10^((p[3]-x)*p[4]))/growthfactor(z)*(1+z)^p[5])
  } else {
    #return(p[1]+(p[2]+10^((x-p[3])*p[4]))/growthfactor(z)*(1+z)^p[5])
    return(p[1]+p[2]*(1+z)^p[5]*(1+exp((x-p[3])*p[4])))
    #return(p[1]*(1+10^((x-p[3]*(1+z)^p[2])*p[4]))/growthfactor(z)^p[5])
  }
}
example()
