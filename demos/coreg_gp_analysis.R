rm(list=ls())
library(ggplot2)
library(maps)
library(dplyr)

UK <- map_data(map = "world", region = "UK") 
preds <- read.csv('Documents/phd/Doubly-Stochastic-GPs/demos/corregionalised_gp_results.csv')
print(head(preds))

ggplot(data = preds, aes(x = lon, y = lat)) + 
  geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
  geom_point(aes(colour=var), alpha=0.5, size=2) + 
  scale_color_gradient(low = "orange", high = "red") +
  facet_wrap(.~date) +
  coord_map() +
  theme_bw() 

preds %>% 
  ggplot(aes(var)) +
  geom_density()
