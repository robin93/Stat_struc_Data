menH <- rnorm(1000,mean=70,sd=3)
menW <- runif(1000,15,70)+ 4*menH + 144 - 280 -42
men_cor <- cor(menH,menW)

womenH <- rnorm(1000,mean=64,sd=3)
womenW <- runif(1000,15,70)+ 4*womenH + 120 -256 -42
women_cor <- cor(womenH,womenW)


comb_H <- c(menH,womenH)
comb_W <- c(menW,womenW)
comb_cor <- cor(comb_H,comb_W)

men_women_data <- cbind(menH,menW,womenH,womenW)