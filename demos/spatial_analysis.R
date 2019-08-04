rm(list=ls())
library(ggplot2)
library(maps)
library(dplyr)
library(gridExtra)
source('Documents/phd/Doubly-Stochastic-GPs/demos/plotters.R')

preds <- read.csv('Documents/phd/Doubly-Stochastic-GPs/colab_results/spatial_preds_1000.csv')
linspace <- read.csv('Documents/phd/Doubly-Stochastic-GPs/colab_results/spatial_linspace_1000.csv')
cams_truth <- read.csv('Documents/phd/Doubly-Stochastic-GPs/demos/cams_time_fullmonth.csv') %>% 
  dplyr::filter(date == 93)
     
ylims = c(min(preds$lat), max(preds$lat))
xlims = c(min(preds$lon), max(preds$lon))

spatial_p <- plot_spatial_heldout(spatial_preds, 'Spatial GP\'s Error Surface', ylims, xlims)
spatial_p
linspace_p <- plot_spatial_linspaces(linspace, "Spatial GP\'s Response Surface", ylims, xlims)
linspace_p
cams_p <- plot_spatial_cams(cams_truth, "CAMS Output",  ylims, xlims)
cams_p
grid.arrange(linspace_p, cams_p, ncol=2)

ggsave('Documents/phd/Doubly-Stochastic-GPs/plots/spatial_analysis/spatial_linsp_vs_cams.pdf', arrangeGrob(linspace_p, cams_p, ncol=2), device='pdf', dpi=100)
ggsave('Documents/phd/Doubly-Stochastic-GPs/plots/spatial_analysis/spatial_gp_error_surf.pdf', spatial_p, device='pdf', dpi=100)

ggsave('Documents/phd/Doubly-Stochastic-GPs/plots/spatial_analysis/spatial_linsp_vs_cams.jpg', arrangeGrob(linspace_p, cams_p, ncol=2), device='jpg')
ggsave('Documents/phd/Doubly-Stochastic-GPs/plots/spatial_analysis/spatial_gp_error_surf.jpg', spatial_p, device='jpg')

truth_vs_error <- preds %>% 
  ggplot(aes(x = truth, y = sq_error)) +
  geom_point(alpha = 0.5) +
  labs(x='Truth PM2.5 Value', y = 'Squared Error', title = 'Observed vs. Prediction Error') +
  theme_bw()

ggsave('~/Documents/phd/Doubly-Stochastic-GPs/plots/spatial_analysis/spatial_truth_vs_error.jpg', plot = truth_vs_error, device = 'jpg')
