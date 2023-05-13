import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D   
from matplotlib.ticker import AutoMinorLocator


def visualize_location(df, bin_size, flare_class, models):
    cmap=plt.cm.Blues

    plt.rcParams.update({'font.size': 16})

    
    df = df.copy()
    # df[['lon', 'lat']] = df[['fl_lon', 'fl_lat']]
    df[["lon", "lat"]] = df["fl_location"].str.strip(r"[()]").str.split(",", expand=True).astype(str)
    df['lon'] = pd.to_numeric(df['lon']).round(decimals=2).astype(str).replace(r'\.0$', '', regex=True)
    df[["lon", "lat"]] = df[['lon', 'lat']].astype(float)

    #To include the limb locations in the bins with in limb-location. Just handling border cases.
    #This actually makes, grid of [-70 to 70] and also [85 to 90] inclusive of both the ends hence 6 by 5 grid in the border
    df.loc[df['lon'] == 70, 'lon'] -= 1
    df.loc[df['lon'] == 90, 'lon'] -= 1

    print(df.lon.max(), df.lon.min())
    print(df.lat.max(), df.lat.min())

    # Create a new column 'result' based on the condition
    df['result'] = np.where(df['flare_prob'] >= 0.5, 'TP', 'FN')

    # Define the grid and group the DataFrame by the grid and 'result'
    grid = bin_size
    df_grouped = df.groupby([np.floor(df['lat']/grid)*grid, np.floor(df['lon']/grid)*grid, 'result']).size().unstack(fill_value=0)

    # Compute TP/(TP+FN)
    df_grouped['TP/(TP+FN)'] = df_grouped['TP'] / (df_grouped['TP'] + df_grouped['FN'])
    vmin = df_grouped['TP/(TP+FN)'].min()
    vmax = df_grouped['TP/(TP+FN)'].max()
    # print(df_grouped.to_markdown())

    # Create the heatmap
    heatmap, xedges, yedges = np.histogram2d(df_grouped.index.get_level_values(1), 
                                            df_grouped.index.get_level_values(0), 
                                            bins=[np.arange(-90, 95, grid), np.arange(-35, 40, grid)], 
                                            weights=df_grouped['TP/(TP+FN)'])

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # fig = figure(figsize=(16, 4), dpi=300)
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    # Add stars at the center of any grid where TP is 0 but FN is greater than 0
    for i, x in enumerate(np.arange(-90, 95, grid)):
        for j, y in enumerate(np.arange(-35, 40, grid)):
            if (y, x) in df_grouped.index:
                if df_grouped.loc[(y, x), 'TP'] == 0 and df_grouped.loc[(y, x), 'FN'] != 0:
                    star_artist = ax.scatter(x+(bin_size/2), y+(bin_size/2), marker='X', s=80, color='red', edgecolors='red')
                        

    # instance is used to divide axes
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size = "5%",
                            pad = 1.2,
                            pack_start = True)
    fig.add_axes(cax)
    
    # creating colorbar
    fig.colorbar(im, cax = cax, orientation = "horizontal", label='Recall')


    ax.axvline(x=-70, color='red', linestyle='-')
    ax.axvline(x=70, color='red',  linestyle='-')


    ax.set_xticks(np.arange(-90, 95, 10))
    ax.set_xticks(np.arange(-90, 95, grid), minor=True)
    ax.set_yticks(np.arange(-35, 40, 10))
    ax.set_yticks(np.arange(-35, 40, grid), minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    if models=='alex':
        legend_elements = [Line2D([], [], marker='X', color='red', label='No Correct Predictions', markersize=9, linestyle='None')]
        l = ax.legend(handles=legend_elements, loc=(0.4, 1.004))
        l.get_frame().set_alpha(0.7)

    if models!= 'resnet':
        ax.tick_params(left=True,
                bottom=True,
                labelleft=True,
                labelbottom=False)
        ax.grid(True)
    else:
        ax.set_xlabel('Heliographic Longitude (Central Meridian Distance)', fontsize=18)
        ax.set_xticks(np.arange(-90, 95, 10))
        ax.set_xticklabels(np.arange(-90, 95, 10), fontsize=15, rotation=45)

    fig.tight_layout(pad=0.3)
    fig.savefig(f'plots/{flare_class}flare_bin_{bin_size}_{models}.svg', dpi=300, transparent=True)
    # plt.show()

bin = 5
models = ['alex', 'vgg', 'resnet']
for items in models:
    print(items)
    df1 = pd.read_csv(f'results/{items}_x_class.csv')
    df2 = pd.read_csv(f'results/{items}_m_class.csv')
    visualize_location(df1, bin,  'X', items)
    visualize_location(df2, bin,  'M', items)
    visualize_location(pd.concat([df1, df2]), bin, 'Combined', items)
