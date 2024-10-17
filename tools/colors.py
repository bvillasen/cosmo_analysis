import pylab
import palettable

# Colors
bright_green = pylab.cm.viridis(.7)
light_blue = pylab.cm.cool(.3)
dark_blue = pylab.cm.viridis(.3) 
dark_purple = pylab.cm.Purples(.7)
blue = 'C0'
orange = 'C1'
green = 'C2'
red = 'C3'
purple = 'C4'
cyan = 'C9' 


cmap_haline = palettable.cmocean.sequential.Haline_10.mpl_colormap
cmap_deep_r = palettable.cmocean.sequential.Deep_20_r.mpl_colormap
cmap_ice    = palettable.cmocean.sequential.Ice_20.mpl_colormap

cmap_davos =  palettable.scientific.sequential.Davos_20.mpl_colormap
cmap_devon =  palettable.scientific.sequential.Devon_20.mpl_colormap
cmap_oslo =  palettable.scientific.sequential.Oslo_20.mpl_colormap
cmap_lapaz =  palettable.scientific.sequential.LaPaz_20.mpl_colormap
cmap_turku =  palettable.scientific.sequential.Turku_20.mpl_colormap
cmap_lajolla_r =  palettable.scientific.sequential.LaJolla_20_r.mpl_colormap

haline = palettable.cmocean.sequential.Haline_10_r.mpl_colors
matter = palettable.cmocean.sequential.Matter_20.mpl_colors
purple_blue = palettable.colorbrewer.sequential.PuBu_9.mpl_colors
purples = palettable.colorbrewer.sequential.Purples_9.mpl_colors
yellows = palettable.colorbrewer.sequential.YlOrRd_9.mpl_colors
greens = palettable.colorbrewer.sequential.YlGn_9.mpl_colors 
blues = palettable.colorbrewer.sequential.Blues_9.mpl_colors 
reds = palettable.colorbrewer.sequential.Reds_9.mpl_colors 
yellow_green_blues =  palettable.colorbrewer.sequential.YlGnBu_5.mpl_colors 

ocean_blue = haline[-3]
ocean_green = haline[4]
sky_blue = purple_blue[4]
dark_purple = purples[-1]

dense = palettable.cmocean.sequential.Dense_20_r
gray = palettable.cmocean.sequential.Gray_20_r

light_orange = yellows[3]
dark_green = haline[3]

# sky_blue = blues[4]
yellow = yellows[2]
light_green = greens[5]
light_red = reds[5]