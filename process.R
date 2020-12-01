library(dplyr)

combined1 <- read.csv("data/input.csv")
weather <- read.csv("data/weather.csv")

combined1$standarddate <- as.Date(combined1$date, "%m/%d/%Y")
weather$standarddate <- as.Date(weather$DATE, "%m/%d/%y")

skinyweather <- weather %>%
  select(standarddate, AWND, PRCP, SNOW, SNWD, TAVG, TMAX, TMIN, WDF2, WSF2)

combined2 <- inner_join(combined1, skinyweather, by="standarddate")

write.csv(combined2, "data/data.csv")


