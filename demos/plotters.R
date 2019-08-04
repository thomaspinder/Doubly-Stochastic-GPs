plot_linspaces <- function(df, title, dlims, ylims, xlims){
  UK <- map_data(map = "world", region = "UK") 
  predictions <- df %>% 
    dplyr::filter(date > dlims[1] & date < dlims[2]) %>% 
    ggplot(aes(x = lon, y = lat)) + 
    geom_point(aes(colour=mu), alpha=0.5, size=8) + 
    scale_color_gradient(low = "white", high = "red") +
    labs(x='Longitude', y='Latitude', colour= 'PM2.5 Levels', title=title) +
    ylim(ylims[1], ylims[2]) +
    xlim(xlims[1], xlims[2]) +
    facet_wrap(.~date) +
    geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
    coord_map() +
    theme_bw() %+replace% 
    theme(plot.margin=grid::unit(c(0,0,0,0), "mm"))
  return(predictions)
}

plot_spatial_linspaces <- function(df, title, ylims, xlims){
  UK <- map_data(map = "world", region = "UK") 
  predictions <- df %>% 
    ggplot(aes(x = lon, y = lat)) + 
    geom_point(aes(colour=mu), alpha=0.5, size=8) + 
    scale_color_gradient(low = "white", high = "red") +
    labs(x='Longitude', y='Latitude', colour= 'PM2.5 Levels', title=title) +
    ylim(ylims[1], ylims[2]) +
    xlim(xlims[1], xlims[2]) +
    geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
    coord_map() +
    theme_bw() %+replace% 
    theme(plot.margin=grid::unit(c(0,0,0,0), "mm"))
  return(predictions)
}

plot_heldout <- function(df, title, dlims, ylims, xlims){
  UK <- map_data(map = "world", region = "UK") 
  held_out_error <- df %>% 
    dplyr::filter(date > dlims[1] & date < dlims[2]) %>% 
    ggplot(aes(x = lon, y = lat)) + 
    geom_point(aes(colour=sq_error), size=3) + 
    scale_color_gradient(low = "white", high = "red") +
    labs(x='Longitude', y='Latitude', colour= 'Squared Error', title=title) +
    ylim(ylims[1], ylims[2]) +
    xlim(xlims[1], xlims[2]) +
    facet_wrap(.~date) +
    geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
    coord_map() +
    theme_bw() 
  return(held_out_error)
}

plot_spatial_heldout <- function(df, title, ylims, xlims){
  UK <- map_data(map = "world", region = "UK") 
  held_out_error <- df %>% 
    ggplot(aes(x = lon, y = lat)) + 
    geom_point(aes(colour=sq_error), size=3) + 
    scale_color_gradient(low = "white", high = "red") +
    labs(x='Longitude', y='Latitude', colour= 'Squared Error', title=title) +
    ylim(ylims[1], ylims[2]) +
    xlim(xlims[1], xlims[2]) +
    geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
    coord_map() +
    theme_bw() 
  return(held_out_error)
}



plot_co_heldout <- function(df, title, dlims, ylims, xlims){
  UK <- map_data(map = "world", region = "UK") 
  held_out_error <- df %>% 
    dplyr::filter(date > dlims[1] & date < dlims[2]) %>% 
    ggplot(aes(x = lon, y = lat)) + 
    geom_point(aes(colour=sq_error), size=3) + 
    scale_color_gradient(low = "white", high = "red") +
    labs(x='Longitude', y='Latitude', colour= 'Squared Error', title=title) +
    ylim(ylims[1], ylims[2]) +
    xlim(xlims[1], xlims[2]) +
    facet_wrap(.~date+indicator) +
    geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
    coord_map() +
    theme_bw() 
  return(held_out_error)
}

plot_cams <- function(df, title, dlims, ylims, xlims){
  UK <- map_data(map = "world", region = "UK") 
  cams_plot <- df %>% 
    dplyr::filter(date > dlims[1] & date < dlims[2]) %>% 
    ggplot(aes(x = lon, y = lat)) + 
    geom_point(aes(colour=val), alpha=0.5, size=8) + 
    ylim(ylims[1], ylims[2]) +
    xlim(xlims[1], xlims[2]) +
    scale_color_gradient(low = "white", high = "red") +
    labs(x='Longitude', y='Latitude', colour= 'PM2.5 Levels', title=title) +
    facet_wrap(.~date) +
    geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
    coord_map() +
    theme_bw() 
  return(cams_plot)
}

plot_spatial_cams <- function(df, title, ylims, xlims){
  UK <- map_data(map = "world", region = "UK") 
  cams_plot <- df %>% 
    ggplot(aes(x = lon, y = lat)) + 
    geom_point(aes(colour=val), alpha=0.5, size=8) + 
    ylim(ylims[1], ylims[2]) +
    xlim(xlims[1], xlims[2]) +
    scale_color_gradient(low = "white", high = "red") +
    labs(x='Longitude', y='Latitude', colour= 'PM2.5 Levels', title=title) +
    geom_polygon(data = UK, aes(x = long, y = lat, group = group), fill = NA, color = "black") +
    coord_map() +
    theme_bw() 
  return(cams_plot)
}
