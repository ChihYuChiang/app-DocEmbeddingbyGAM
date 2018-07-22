#--Set up
library(tidyverse)

#Read in all sources of reviews 
review_spot <- read_csv("../data/df_cb_main.csv")
review_radar <- read_csv("../data/df_cb_main_GamesRadar.csv")
review_poly <- read_csv("../data/df_cb_main_Polygon.csv")
review_core <- read_csv("../data/raw_coregame/core_game_reviews.csv")


#--Combine all sources of reviews
#Select only necessary columns
spot <- select(review_spot, Author=`Author Name`, Game=`Game Title`, Review)
radar <- select(review_radar, Author=`Author Name`, Game=`Game Title`, Review)
poly <- select(review_poly, Author=`Author Name`, Game=`Game Title`, Review)
core <- select(review_core, Author=Site, Game=`Game Title`, Review)

#Source: 1=spot, 2=radar, 3=poly, 4=core
review_combined <- bind_rows(spot, radar, poly, core, .id="Source")


#--Assign coreID
lookup <- read_csv("../data/raw_coregame/core_game_lookup_list.csv")
review_combined <- left_join(x=review_combined, y=lookup, by="Game")

review_combined$CoreID[is.na(review_combined$CoreID)] <- 0


#--Save
write.csv(review_combined, file="../data/df_cb_main_combined.csv", fileEncoding="UTF-8")
