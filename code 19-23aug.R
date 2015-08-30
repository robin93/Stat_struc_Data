n = 300
x = rnorm(n)

# Empirical distribution
x = sort(x)
y = (1:(n-1))/n
#par(mfrow=c(1,1))
plot(c(rep(x,each=2)),c(0,rep(y,each=2),1),type="l")
lines(x,dnorm(x))   #plots the probability density function for the random values of x
lines(x,pnorm(x))   #plots the cumulative distribution function for the random values of x

# Normal plot (QQ plot for normal distribution)
qqnorm(x)
qqline(x)

# QQ plot
z = rchisq(300, df = 3)
qqplot(qchisq(ppoints(300), df = 3), z)
qqline(z, distribution = function(p) qchisq(p, df = 3))

# PP plot
z = sort(z)
plot(pchisq(z,df=3),(1:n)/n,xlim=c(0,1),ylim=c(0,1)); lines(c(0,1),c(0,1))
plot(pchisq(x,df=3),(1:n)/n,xlim=c(0,1),ylim=c(0,1)); lines(c(0,1),c(0,1))

# Kolmogorov-Smirnov test based on max separation between EDF & theoretical CDF
n = 30
x = rnorm(n)
ks.test(x,pnorm)

# Empirical distribution with KS test p-value
edf = ecdf(x)
gridpt = min(x) + (max(x) - min(x))*(0:100000)*.00001
plot(gridpt,edf(gridpt),type="l",xlab="value",ylab="distribution function")
lines(gridpt,pnorm(gridpt))
pval = ks.test(x,pnorm)$p; pval = floor(1000*pval+.5)*0.001
text(fivenum(x)[2],0.9,paste("KS test p-value = ",pval))

# Shapiro-Wilk test
n = 30
x = rnorm(n,0,2)
shapiro.test(x)

# normal plot with KS and SW tests
qqnorm(x)
lines(c(-3,3),c(-3,3),col=4)
psw = shapiro.test(x)$p; psw = floor(1000*psw+.5)*0.001
pks = ks.test(x,pnorm)$p; pks = floor(1000*pks+.5)*0.001
text(fivenum(x)[2],fivenum(x)[5]-.5,paste("SW test p-value = ",psw))
text(fivenum(x)[2],fivenum(x)[5]-1,paste("KS test p-value = ",pks))

# smooth density from histogram
par(mfrow=c(1,2))
hist(x,probability=T)
plot(density(x),ylim=c(0,0.42),main="density estimate")

# chi-sq test for discrete data
n = 1000
x = rbinom(n,5,.5)
obscount = hist(x,breaks=(c(0:6)-0.5),plot=F)$counts
prob = dbinom((0:5),5,.45)
chisq.test(obscount, p = prob)

# another example
x = floor(10*runif(1000))
obscount = hist(x,breaks=(c(0:10)-0.5),plot=F)$counts
prob = rep(1/length(obscount), length(obscount))
chisq.test(obscount, p = prob)

