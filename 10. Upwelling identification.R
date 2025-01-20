library(RmarineHeatWaves); 
library(plyr); 
library(dplyr); 
library(ggplot2)
library(lubridate)

######
setwd("C:/Users/david/Downloads/Paper New Lien/raw_data/")
getwd()
data<- read.csv("./polygon_4_combined.csv")

ts_new <- data |>
  mutate(time = as.Date(time, format = "%Y-%m-%d")) |>
  rename(temp = analysed_sst, t=time) |>
  mutate(temp= temp-273.15,
         doy = yday(t))|>
  select(doy, t, temp)


mcs<- detect(ts_new,
             doy=doy,
             x=t,
             y=temp,
             climatology_start = "2003-01-01",
             climatology_end = "2023-12-31",
             pctile = 85,
             cold_spells = TRUE,
             min_duration = 3,
             #window_half_width = 5, 
             smooth_percentile = TRUE,
             #join_across_gaps = TRUE, 
             #max_gap = 2, 
             #max_pad_length = 3, 
)


clim<- mcs$clim |> ungroup()

events<- mcs$event |> ungroup()

write.csv(clim, file = "Clim_Upwelling_R4.csv", row.names = FALSE)
write.csv(events, file = "Events_Upwelling_R4.csv", row.names = FALSE)
getwd()

# Polygon 1
setwd("C:/Users/david/Downloads/Paper New Lien/raw_data/")
getwd()
data<- read.csv("./polygon_3_combined.csv")

ts_new <- data |>
  mutate(time = as.Date(time, format = "%Y-%m-%d")) |>
  rename(temp = analysed_sst, t=time) |>
  mutate(temp= temp-273.15,
         doy = yday(t))|>
  select(doy, t, temp)


mcs<- detect(ts_new,
             doy=doy,
             x=t,
             y=temp,
             climatology_start = "2003-01-01",
             climatology_end = "2023-12-31",
             pctile = 85,
             cold_spells = TRUE,
             min_duration = 3,
             #window_half_width = 5, 
             smooth_percentile = TRUE,
             #join_across_gaps = TRUE, 
             #max_gap = 2, 
             #max_pad_length = 3, 
)


clim<- mcs$clim |> ungroup()

events<- mcs$event |> ungroup()

write.csv(clim, file = "Clim_Upwelling_R1.csv", row.names = FALSE)
write.csv(events, file = "Events_Upwelling_R1.csv", row.names = FALSE)
getwd()
