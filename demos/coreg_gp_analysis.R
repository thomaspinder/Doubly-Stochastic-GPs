rm(list=ls())
library(ggplot2)
library(maps)
library(dplyr)
uni_dir <- '~/Documents/models/Doubly-Stochastic-DGP/'
home_dir <- 'Documents/phd/Doubly-Stochastic-GPs/'
setwd('~/')
setwd(uni_dir)
source('demos/plotters.R')

UK <- map_data(map = "world", region = "UK")

# Load spatiotemporal data
preds <- read.csv('colab_results/ard_linspace_300.csv')
cams_truth <- read.csv('demos/cams_time_fullmonth.csv')
held_out_preds <- read.csv('colab_results/ard_preds_300.csv') %>% 
  mutate(col_ind = ifelse(sq_error > quantile(sq_error, .99), sq_error, 0))

# Load coregionalised data
coreg_lin_data <- read.csv('demos/corregionalised_gp_nonsep_results_1week_sparse2000_linspace.csv')
coreg_ho_data <- read.csv('demos/corregionalised_nonsep_gp_results_1week_sparse2000.csv') %>% 
  mutate(indicator = ifelse(indicator==0, "AURN", "CAMS"))

# Compute date ranges
ylims = c(min(preds$lat), max(preds$lat))
xlims = c(min(preds$lon), max(preds$lon))
dlims = c(94, 111)
cdlims = c(min(coreg_ho_data$date), max(coreg_ho_data$date)+1)

# Plot spatiotemporal results
held_out_error <- plot_heldout(df = held_out_preds, title = "Sparse Gaussian Process Error Surface", dlims=dlims, ylims=ylims, xlims=xlims)
predictions <- plot_linspaces(preds, "Sparse Gaussian Process Response", dlims, ylims, xlims)
truth <- plot_cams(cams_truth, "CAMS Output", dlims, ylims, xlims)

# Plot coregionalised results
coreg_linspace <- plot_linspaces(coreg_lin_data, title="Coregionalised GP Response Surface", dlims=cdlims, ylims=ylims, xlims=xlims)
# Indicator 0 is for AURN, 1 for CAMS
coreg_heldout <- plot_co_heldout(coreg_ho_data, title="Coregionalised GP Error Surface", dlims=cdlims, ylims=ylims, xlims=xlims)
coreg_var <- plot_var_linspaces(coreg_lin_data, title="Coregionalised GP Uncertainty Surface", dlims=cdlims, ylims=ylims, xlims=xlims)


ggsave('plots/st_univariate_analysis/cams_uni_error_surf.pdf', plot=held_out_error, device='pdf', dpi=100)
ggsave('plots/st_univariate_analysis/cams_uni.pdf', plot=truth, device='pdf', dpi=100)
ggsave('plots/st_univariate_analysis/sgp_uni.pdf', plot=predictions, device='pdf', dpi=100)
ggsave('plots/coregionalised_plots/coreg_linspace.pdf', plot=coreg_linspace, device='pdf', dpi=100)
ggsave('plots/coregionalised_plots/coreg_heldout.pdf', plot=coreg_heldout, device='pdf', dpi=100)

ggsave('plots/st_univariate_analysis/cams_uni_error_surf.jpg', plot=held_out_error, device='jpg')
ggsave('plots/st_univariate_analysis/cams_uni.jpg', plot=truth, device='jpg')
ggsave('plots/st_univariate_analysis/sgp_uni.jpg', plot=predictions, device='jpg')
ggsave('plots/coregionalised_plots/coreg_linspace.jpg', plot=coreg_linspace, device='jpg', dpi=100)
ggsave('plots/coregionalised_plots/coreg_heldout.jpg', plot=coreg_heldout, device='jpg', dpi=100)


preds %>% 
  dplyr::filter(date > 94 & date < 111) %>% 
  ggplot(aes(x = lon, y = lat)) + 
  geom_point(aes(colour=var), alpha=0.5, size=8) + 
  scale_color_gradient(low = "white", high = "red") +
  facet_wrap(.~date) +
  geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
  coord_map() +
  theme_bw() 

preds %>% 
  dplyr::filter(date == 94) %>%
  dplyr::filter(sq_error > 5) %>%
  ggplot(aes(x = lon, y = lat)) + 
  geom_point(aes(colour=sq_error), size=5) + 
  scale_color_gradient(low = "white", high = "red") +
  facet_wrap(.~date) +
  geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
  coord_map() +
  theme_bw() 


# Where did the errors occur
preds %>% 
  dplyr::filter(col_ind > 0) %>% 
  ggplot(aes(x = lon, y = lat)) + 
    geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
    geom_point(aes(colour=sq_error), size=1) + 
    scale_color_gradient(low = "white", high = "red") +
    facet_wrap(.~date) +
    coord_map() +
    theme_bw() 

error_bplot <- held_out_preds %>% 
  dplyr::filter(date > 94 & date < 111) %>% 
  ggplot(aes(x = as.factor(date), y=sq_error)) +
  geom_boxplot(alpha=0.5) +
  coord_flip() +
  theme_bw() 

ggsave('plots/st_univariate_analysis/hout_error_bplot.pdf', plot = error_bplot, device='pdf')

truth_vs_error <- held_out_preds %>% 
  ggplot(aes(x=mu, y = sq_error)) +
  geom_point(alpha = 0.5) +
  labs(x='Truth PM2.5 Value', y = 'Squared Error', title = 'Observed vs. Prediction Error') +
  theme_bw()
truth_vs_error
ggsave('plots/st_univariate_analysis/truth_vs_error.jpg', plot = truth_vs_error, device = 'jpg')
