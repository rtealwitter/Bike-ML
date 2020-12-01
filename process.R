library(dplyr)

data <- read.csv("data/data.csv")
weather <- read.csv("data/weather.csv")

data$standarddate <- as.Date(data$date, "%m/%d/%Y")
weather$standarddate <- as.Date(weather$DATE, "%m/%d/%y")

skinyweather <- weather %>%
  select(standarddate, AWND, PRCP, SNOW, SNWD, TAVG, TMAX, TMIN, WDF2, WSF2)

data1 <- inner_join(data, skinyweather, by="standarddate")

write.csv(data1, "data/data.csv")

