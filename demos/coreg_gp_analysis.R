rm(list=ls())
library(ggplot2)
library(maps)
library(dplyr)
source('Documents/phd/Doubly-Stochastic-GPs/demos/plotters.R')

UK <- map_data(map = "world", region = "UK") 
preds <- read.csv('Documents/phd/Doubly-Stochastic-GPs/colab_results/ard_linspace_300.csv')
cams_truth <- read.csv('Documents/phd/Doubly-Stochastic-GPs/demos/cams_time_fullmonth.csv')
held_out_preds <- read.csv('Documents/phd/Doubly-Stochastic-GPs/colab_results/ard_preds_300.csv') %>% 
  mutate(col_ind = ifelse(sq_error > quantile(sq_error, .99), sq_error, 0))
coreg_lin_data <- read.csv('Documents/phd/Doubly-Stochastic-GPs/demos/corregionalised_gp_nonsep_results_1week_sparse2000_linspace.csv')
coreg_ho_data <- read.csv('Documents/phd/Doubly-Stochastic-GPs/demos/corregionalised_nonsep_gp_results_1week_sparse2000.csv')

ylims = c(min(preds$lat), max(preds$lat))
xlims = c(min(preds$lon), max(preds$lon))
dlims = c(94, 111)
cdlims = c(min(coreg_ho_data$date), max(coreg_ho_data$date)+1)

held_out_error <- plot_heldout(df = held_out_preds, title = "Sparse Gaussian Process Error Surface", dlims=dlims, ylims=ylims, xlims=xlims)
predictions <- plot_linspaces(preds, "Sparse Gaussian Process Response", dlims, ylims, xlims)
truth <- plot_cams(cams_truth, "CAMS Output", dlims, ylims, xlims)
coreg_linspace <- plot_linspaces(coreg_lin_data, title="Coregionalised GP Response Surface", dlims=cdlims, ylims=ylims, xlims=xlims)

# Indicator 0 is for AURN, 1 for CAMS
coreg_heldout <- plot_co_heldout(coreg_ho_data, title="Coregionalised GP Error Surface", dlims=cdlims, ylims=ylims, xlims=xlims)


ggsave('Documents/phd/Doubly-Stochastic-GPs/plots/st_univariate_analysis/cams_uni_error_surf.pdf', plot=held_out_error, device='pdf', dpi=100)
ggsave('Documents/phd/Doubly-Stochastic-GPs/plots/st_univariate_analysis/cams_uni.pdf', plot=truth, device='pdf', dpi=100)
ggsave('Documents/phd/Doubly-Stochastic-GPs/plots/st_univariate_analysis/sgp_uni.pdf', plot=predictions, device='pdf', dpi=100)

ggsave('Documents/phd/Doubly-Stochastic-GPs/plots/st_univariate_analysis/cams_uni_error_surf.jpg', plot=held_out_error, device='jpg')
ggsave('Documents/phd/Doubly-Stochastic-GPs/plots/st_univariate_analysis/cams_uni.jpg', plot=truth, device='jpg')
ggsave('Documents/phd/Doubly-Stochastic-GPs/plots/st_univariate_analysis/sgp_uni.jpg', plot=predictions, device='jpg')


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

ggsave('~/Documents/phd/Doubly-Stochastic-GPs/plots/st_univariate_analysis/hout_error_bplot.pdf', plot = error_bplot, device='pdf')

truth_vs_error <- held_out_preds %>% 
  ggplot(aes(x=mu, y = sq_error)) +
  geom_point(alpha = 0.5) +
  labs(x='Truth PM2.5 Value', y = 'Squared Error', title = 'Observed vs. Prediction Error') +
  theme_bw()
truth_vs_error
ggsave('~/Documents/phd/Doubly-Stochastic-GPs/plots/st_univariate_analysis/truth_vs_error.jpg', plot = truth_vs_error, device = 'jpg')
