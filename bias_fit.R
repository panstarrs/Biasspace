path.bias = '/Users/hengxingpan/Work/danail/ProCorr/Bias'
#png(filename="/Users/hengxingpan/Work/danail/ProCorr/Bias/name1.png")
#bands = 'mag_GALEX-FUV_r_tot'
bands = 'mstars_tot'
bias = read.table(paste(path.bias,'/output/B199/G_1/sat_cen/p_c/',bands,"_pb.txt",sep = ""),skip=0,sep=' ')
x = bias[,1]
y = bias[,2]
s = bias[,3]

dput(x)
dput(y)
dput(s)

f = function(x,p) {
  #p[1]+(10^x/p[2])^p[3]
  p[1]+p[2]*10^(x*p[3])
  #p[1]+(10^(-0.4*(x-p[2])))^p[3]
}

#ptrue = c(0,1,2)

#ptrue = c(1.04792723e+00 ,  3.58992492e+11, 5.69988478e-01)
ptrue = c(1.04792723e+00 ,  2.59259421e-07, 5.69988472e-01)
#ptrue = c( 0.98952624,-21.48585834, 0.94514184)

#s = array(0.1,length(x))
#y = f(x,ptrue)


n = length(ptrue)

  neglogL = function(p) {
    return(sum((y-f(x,p))^2/s^2+log(2*pi*s^2))/2)
  }
  #opt = optim(ptrue,neglogL, hessian = TRUE)
  opt = optim(ptrue,neglogL,control = list(abstol=1e-10), hessian = TRUE)
  pfit = opt$par
  C = solve(opt$hessian)
  pfit.sigma = sqrt(diag(C))
  
  
    plot(x,y)
    
    #lines(x,f(x,pfit))
    #x = seq(0,12,0.1)
    lines(x,f(x,ptrue))
    for (i in seq(n)) {
      cat(sprintf('p[%d] = %e+-%e\n',i,pfit[i],pfit.sigma[i]))
      #cat(sprintf('p[%d] = %e+-%6.3f\n',i,pfit[i],pfit.sigma[i]))
    }
    #dev.off()