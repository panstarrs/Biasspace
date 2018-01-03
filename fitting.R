par(mfcol=c(2,1))

f = function(x,p) {
  p[1]+(x/p[2])^p[3]
}

n.iterations = 100
x = seq(8,10,0.05)
ptrue = c(5,7,20)
s = array(20,length(x))

xlim=c(7,10);ylim=c(0.2,10)

n = length(ptrue)
pfit.mean = array(0,n)
C.mean = array(0,c(n,n))

for (iteration in seq(n.iterations)) {
  
  set.seed(iteration)
  y = f(x,ptrue)+rnorm(length(x))*s
  
  neglogL = function(p) {
    return(sum((y-f(x,p))^2/s^2+log(2*pi*s^2))/2)
  }
  #opt = optim(c(2,1,0),neglogL,control = list(abstol=1e-10), hessian = TRUE)
  opt = optim(ptrue,neglogL,hessian = TRUE)
  #opt = optim(ptrue,neglogL,control = list(parscale=abs(ptrue)), hessian = TRUE)
  pfit = opt$par
  C = solve(opt$hessian)
  pfit.mean = pfit.mean+pfit
  C.mean = C.mean+C
  pfit.sigma = sqrt(diag(C))
  
  if (iteration==1) {
    #plot(xlim,ylim)
    #plot(xlim,ylim,log='x')
    plot(x,y)
    lines(x,f(x,pfit))
    for (i in seq(n)) {
      cat(sprintf('p[%d] = %6.3f+-%6.3f\n',i,pfit[i],pfit.sigma[i]))
    }
    plot(ptrue[1],ptrue[2],pch=3,cex=3)
    #plot(ptrue[1],ptrue[2],xlim=c(-5,5),ylim=c(-1,4),pch=3,cex=3)
  }
  points(pfit[1],pfit[2],pch=20,col='#0000ff22')
}

pfit.mean = pfit.mean/n.iterations
C.mean = C.mean/n.iterations

points(pfit.mean[1],pfit.mean[2],pch=3,cex=3,col='blue')
#points(pfit.mean[1],pfit.mean[2],xlim=c(-5,5),ylim=c(-1,4),pch=3,cex=3,col='blue')

library(ellipse)
pts = 
  ellipse::ellipse(C.mean[1:2,1:2],centre=pfit.mean[1:2],level=0.68,draw=F)
lines(pts[,1],pts[,2])