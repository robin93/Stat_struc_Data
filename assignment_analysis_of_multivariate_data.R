#national track records for women by country analysis
#reading the data files
ntr_data <- data.frame(read.table("national_track_record_women.dat",header = FALSE,fill = TRUE))

#removing bad rows
clean_data <- ntr_data[-c(30:33),]

clean_data[,5:8] <- (clean_data[,5:8])*60 #converting values to a scale

#clean the data 
clean_data[,3:4] <- (clean_data[,3:4])*1  
clean_data[,2] <- as.numeric(clean_data[,2])

#attaching the column names
colnames(clean_data)<-c("Country","100m","200m","400m","800m","1500m","3000m","Marathon")

means <- colMeans(clean_data[,2:8])
variance <- cov(clean_data[,2:8])
correlation <- cor(clean_data[,2:8])

means
variance
correlation

