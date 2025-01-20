import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from netCDF4 import Dataset
import numpy as np
from netCDF4 import num2date
from mpl_toolkits.basemap import Basemap
import os
print(os.getcwd())

ncfile='./GRIDONE_2D.nc'
file = Dataset(ncfile,mode='r') # abrir el archivo en modo lectura 
print(file.variables)
lon=file.variables["lon"][:]
lat=file.variables["lat"][:]
ele= file.variables["elevation"][:]
lon1=lon[5100:7501]
lat1=lat[5700:7201]
ele1=ele[5700:7201,5100:7501]
np.max(ele1)
np.min(ele1)
lons,lats=np.meshgrid(lon1,lat1)

def custom_ocean_cmap(numcolors=11, name='custom_ocean_cmap',
                      mincol='Teal', midcol='Turquoise', maxcol='White'):
    """ Create a custom colormap for ocean visualization.
    
    Default is teal to turquoise to white with 11 colors.
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap

# Define new color levels for the ocean
blevels = [-8000, -7000, -6000, -5000, -4000, -3000, -2000, -1500, -1000, -500, -200, -100, 0]
N = len(blevels) - 1
cmap_ocean = custom_ocean_cmap(N, mincol='Teal', midcol='Turquoise', maxcol='White')

m=Basemap(projection='cyl',llcrnrlon=-92,llcrnrlat=6,urcrnrlon=-60,urcrnrlat=24, resolution ='h')

from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings("ignore")

fig, ax = plt.subplots(figsize=(12, 10))

m.drawlsmask(land_color='gray')
#m.drawcountries()
#m.etopo()
h=m.pcolormesh(lons,lats,ele1,latlon=True, vmin=-8000,vmax=0,cmap=cmap_ocean)# shading='flat'
m.fillcontinents(color='gray')
m.drawcoastlines(color='gray', zorder=0)
m.drawparallels(np.arange(6.,25.,2.),labels=[1,0,0,0],linewidth=0.01)
m.drawmeridians(np.arange(-92.,-60+1.,2),labels=[0,0,0,1],linewidth=0.01)

# POLIGONO 1
delta_x = 1; delta_y = 1;
x1,y1 = m(-79-delta_x,10.5+delta_y);x2,y2 = m(-79-delta_x,10.5-delta_y)
x3,y3 = m(-79+delta_x,10.5-delta_y);x4,y4 = m(-79+delta_x,10.5+delta_y)
print('Limites poligono 1 en lon: ', -79-delta_x ,' a ', -79+delta_x, ' en lat: ', 10.5-delta_y, ' a ', 10.5+delta_y)
poly_1 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)],facecolor='None',edgecolor='orange',linewidth=2,linestyle='--')
plt.gca().add_patch(poly_1)

# POLIGONO 2
x1,y1 = m(-82-delta_x,21+delta_y);x2,y2 = m(-82-delta_x,21-delta_y)
x3,y3 = m(-82+delta_x,21-delta_y);x4,y4 = m(-82+delta_x,21+delta_y)
print('Limites poligono 2 en lon: ', -82-delta_x ,' a ', -82+delta_x, ' en lat: ', 21-delta_y, ' a ', 21+delta_y)
poly_2 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)],facecolor='None',edgecolor='orange',linewidth=2,linestyle='--')
plt.gca().add_patch(poly_2)

# POLIGONO 3
x1,y1 = m(-67-delta_x,11.5+delta_y);x2,y2 = m(-67-delta_x,11.5-delta_y)
x3,y3 = m(-67+delta_x,11.5-delta_y);x4,y4 = m(-67+delta_x,11.5+delta_y)
print('Limites poligono 3 en lon: ', -67-delta_x ,' a ', -67+delta_x, ' en lat: ', 11.5-delta_y, ' a ', 11.5+delta_y)
poly_3 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)],facecolor='None',edgecolor='orange',linewidth=2,linestyle='--')
plt.gca().add_patch(poly_3)

# POLIGONO 4
x1,y1 = m(-72.25-delta_x,12.75+delta_y);x2,y2 = m(-72.25-delta_x,12.75-delta_y)
x3,y3 = m(-72.25+delta_x,12.75-delta_y);x4,y4 = m(-72.25+delta_x,12.75+delta_y)
print('Limites poligono 4 en lon: ', -72.25-delta_x ,' a ', -72.25+delta_x, ' en lat: ', 12.75-delta_y, ' a ', 12.75+delta_y)
poly_4 = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)],facecolor='None',edgecolor='orange',linewidth=2,linestyle='--')
plt.gca().add_patch(poly_4)

# Paises
#ax = plt.subplot(111)
ax.text(-74.5, 9, "Colombia", size=11, va="center", ha="center", rotation=0)
ax.text(-67, 9, "Venezuela", size=11, va="center", ha="center", rotation=0)
ax.text(-78, 8.5, "Panama", size=9, va="center", ha="center", rotation=-36)
ax.text(-82, 8.5, "Costa Rica", size=9, va="center", ha="center", rotation=-30)
ax.text(-85.4, 13, "Nicaragua", size=9, va="center", ha="center", rotation=0)
ax.text(-86, 15, "Honduras", size=9, va="center", ha="center", rotation=0)
#ax.text(-89, 14.3, "El", size=9, va="center", ha="center", rotation=0)
#ax.text(-89, 13.8, "Salvador", size=9, va="center", ha="center", rotation=0)
#ax.text(-90, 15.5, "Guatemala", size=9.6, va="center", ha="center", rotation=0)
#ax.text(-88.5, 17, "Belice", size=9.6, va="center", ha="center", rotation=0)
#ax.text(-93, 17.5, "Mexico", size=9.6, va="center", ha="center", rotation=0)
ax.text(-70.4, 19, "Hispaniola", size=9.6, va="center", ha="center", rotation=0)
ax.text(-66, 18.3, "Puerto", size=9.0, va="center", ha="center", rotation=0)
ax.text(-66, 17.9, "Rico", size=9.0, va="center", ha="center", rotation=0)
ax.text(-61, 14, "Lesser Antilles", size=9.0, va="center", ha="center", rotation=-90)
ax.text(-73, 19, "Haiti", size=9.0, va="center", ha="center", rotation=0)
#ax.text(-81.5, 28, "EEUU", size=9.0, va="center", ha="center", rotation=0)
ax.text(-80, 22.2, "Cuba", size=10.0, va="center", ha="center", rotation=-25)
# Cuencas
ax.text(-77, 13, "Colombia Basin", size=9, va="center", ha="center", rotation=0,color='black')
ax.text(-69, 13.5, "Venezuela Basin", size=9, va="center", ha="center", rotation=0,color='black')
ax.text(-86, 17, "Caiman Basin", size=9, va="center", ha="center", rotation=20,color='black')
ax.text(-84, 21, "Yucatan Basin", size=9, va="center", ha="center", rotation=0,color='black')
#ax.text(-90, 25, "Gulf of Mexico", size=9, va="center", ha="center", rotation=0,color='black')
ax.text(-73, 15.5, "Beata Ridge", size=8, va="center", ha="center", rotation=47,color='black')

# Oceano
#ax.text(-70, 29, "Atlantic Ocean", size=11, va="center", ha="center", rotation=0,color='black')
#ax.text(-90, 8, "Pacific Ocean", size=11, va="center", ha="center", rotation=0,color='black')

# Flechaspara los pasos y canales
#ax.annotate('Golfo Darien', xy=(-76, 10), xytext=(-76, 7),arrowprops=dict(facecolor='blue', shrink=0.001),size=8)
ax.text(-77, 11, "Gulf of", size=9, va="center", ha="center", rotation=0,color='black')
ax.text(-77, 10.5, "Darien", size=9, va="center", ha="center", rotation=0,color='black')
#ax.annotate('Golfo Mosquitos', xy=(-82, 9.5), xytext=(-83,11.5),arrowprops=dict(facecolor='blue', shrink=0.001),size=6)
#ax.annotate('Paso Vientos', xy=(-74, 20), xytext=(-72, 22),arrowprops=dict(facecolor='blue', shrink=0.001),size=8)
#ax.annotate('Paso Mona', xy=(-68, 18.5), xytext=(-66, 20),arrowprops=dict(facecolor='blue', shrink=0.001),size=8)
#ax.annotate('Paso Anegada', xy=(-63, 17.5), xytext=(-61, 19),arrowprops=dict(facecolor='blue', shrink=0.001),size=9)
#ax.annotate('Can. San. Lucía', xy=(-61, 13), xytext=(-59, 13),arrowprops=dict(facecolor='blue', shrink=0.001),size=6)
#ax.annotate('Gol. de Honduras', xy=(-87, 17), xytext=(-90, 22),arrowprops=dict(facecolor='blue', shrink=0.001),size=6)
#ax.annotate('Cuenca Granada', xy=(-63, 13), xytext=(-60, 10),arrowprops=dict(facecolor='blue', shrink=0.001),size=7)

# Otros
ax.text(-81, 17, "Chibcha Channel", size=9, va="center", ha="center", rotation=25,color='black')
ax.text(-63, 14, "Aves", size=9, va="center", ha="center", rotation=0,color='black')
ax.text(-63, 13.3, "Ridge", size=9, va="center", ha="center", rotation=0,color='black')
ax.text(-84, 19.5, "Caimen", size=9, va="center", ha="center", rotation=0,color='black')
ax.text(-84, 19, "Ridge", size=9, va="center", ha="center", rotation=0,color='black')
ax.text(-64, 11, "Cariaco Trench", size=9, va="center", ha="center", rotation=0,color='black')
ax.text(-63.5, 12.5, "Granada Basin", size=9, va="center", ha="center", rotation=0,color='black')
ax.text(-63, 17.5, "Anegada Passage", size=9, va="center", ha="center", rotation=45,color='black')
ax.text(-68, 18.5, "Mona Passage", size=9, va="center", ha="center", rotation=45,color='black')
ax.text(-74, 20, "Windwart Passage", size=9, va="center", ha="center", rotation=45,color='black')
#ax.text(-60, 13, "San. Lucia Chan.", size=9, va="center", ha="center", rotation=-90,color='black')
ax.text(-82, 11, "Gulf of", size=9, va="center", ha="center", rotation=-30,color='black')
ax.text(-82, 10, "Mosquitos", size=9, va="center", ha="center", rotation=-30,color='black')
#################################################################
# Notacion para isobatas de 200, 1000 y 3000 m
# 200 m
#ax.text(-82.5, 13.5, "200 m", size=10, va="center", ha="center", rotation=59,color='lime')
# 1000 m
#ax.text(-80.8, 13, "1000 m", size=10, va="center", ha="center", rotation=57,color='olive')
# 3000 m
#ax.text(-77, 14.3, "3000 m", size=10, va="center", ha="center", rotation=50,color='b')

#################################################################
# Numeros en cajas
ax.text(-67, 11.5, "1", size=18, va="center", ha="center", rotation=0,color='orange')
ax.text(-82, 21, "2", size=18, va="center", ha="center", rotation=0,color='orange')
ax.text(-79, 10.5, "3", size=18, va="center", ha="center", rotation=0,color='orange')
ax.text(-72.25, 12.75, "4", size=18, va="center", ha="center", rotation=0,color='orange')


cbar = plt.colorbar(h, orientation='horizontal', fraction=0.08, pad=0.05)
cbar.set_label('(m)', weight='bold', fontsize=12)
plt.title('Caribbean Sea Bathymetry')

fig.savefig('Figure_1_Transformers.jpeg', dpi=300, bbox_inches='tight')