n <- 10
x <- sort(rnorm(n))
y <- (1:(n-1))/n
plot(c(rep(x,each=2)),c(0,rep(y,each=2),1),type="l")
plot(x,y,type="l")