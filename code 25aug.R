n = 20
x = rnorm(n)
y = rnorm(n) + x
z = rnorm(n) + x - y
X = cbind(x,y,z)
X

colMeans(X) # sample mean of each variable
cov(X) # variances and covariances written as a matrix
cor(X) # correlation matrix of all variables

pairs(X) # matrix of scatter plots

# line plot
xs = (x - min(x))/(max(x) - min(x))
ys = (y - min(y))/(max(y) - min(y))
zs = (z - min(z))/(max(z) - min(z))
Xs = cbind(xs,ys,zs)
p = length(Xs[1,])
plot(c(1,p),c(0,1),type="n",xlab="variable number",ylab = "normalized value")
for (i in 1:n) lines((1:p),Xs[i,],col=i)

scatterplot3d(x,y,z, main="3D Scatterplot") # 3d scatterplot

stars(X) # star plot
