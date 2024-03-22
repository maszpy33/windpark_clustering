

# %%
operator_marker_map = [
    'o',  
    's',
    '^',
    'v',
    '<',
    '>',
    'p',
    '*',
    'h',
    'H',
    '+',
    'x', 
    'D',
    'd',
    '|',
    '_']


# %%
def detail_data_plot_ULevel(df, lat_min, lat_max, lon_min, lon_max):
    # Function to categorize Spannungsebene
    @cache
    def categorize_spannungsebene(row):
        if 'Niederspannung' in row:
            return 'Niederspannung'
        elif 'Mittelspannung' in row:
            return 'Mittelspannung'
        elif 'Hochspannung' in row:
            return 'Hochspannung'
        elif 'Höchstspannung' in row:
            return 'Höchstspannung'
        else:
            return 'Other'

    # Apply the function to create a new column
    df['Spannungsebene_Category'] = df['Spannungsebene'].apply(categorize_spannungsebene)

    # Slice Data for data count
    if SLICE_DATA:
        df = df[(df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) & 
                    (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)]

    # Define color mapping
    color_map = {
        'Niederspannung': 'green',
        'Mittelspannung': 'blue',
        'Hochspannung': 'orange',
        'Höchstspannung': 'red',
        'Other': 'gray'  # For any category that doesn't match the specified ones
    }

    # Calculate aspect ratio
    image_width_to_height_ratio = (lon_max - lon_min) / (lat_max - lat_min)

    # Create a figure with the same aspect ratio
    fig, ax = plt.subplots(figsize=(8, 8 * image_width_to_height_ratio))

    # Set the background color as transparent
    # fig.patch.set_alpha(0)
    # ax.patch.set_alpha(0)

    # Create a Basemap instance that aligns with Google Maps projection and bounds
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                llcrnrlon=lon_min, urcrnrlon=lon_max, lat_ts=20, resolution='i', ax=ax)

    m.drawcoastlines()
    m.drawcountries()

    # Convert lat/long to map projection coordinates
    x, y = m(df.longitude, df.latitude)

    if df.shape[0] < 50:
        maker_size = 10
    else:
        maker_size = 3

    marker_category_list = {}
    scatter_legends = []
    for category, color in color_map.items():
        subset = df[df['Spannungsebene_Category'] == category]
        temp_label = f"[#{subset.shape[0]}] - {category}" if subset.shape[0] != 0 else f"{category}"

        # Iterate through each subset row to plot with the corresponding marker shape
        for index, row in subset.iterrows():
            # operator_name = row['Name des Anlagenbetreibers (nur Org.)']
            operator_name = row['Name des Anschluss-Netzbetreibers']
            if operator_name not in marker_category_list:
                marker_category_list[operator_name] = operator_marker_map.pop(0)
            # marker = operator_marker_map.get(operator_name, 'o')  # Default to circle if operator not in map
            
            x_temp, y_temp = m([row.longitude], [row.latitude])
            scatter = m.scatter(x_temp, y_temp, alpha=0.9, color=color, s=maker_size, 
                                label=temp_label if index == subset.first_valid_index() else "", 
                                edgecolors='black', linewidth=0.2, marker=marker_category_list[operator_name])
            scatter_legends.append(scatter)
            # To avoid duplicate labels in the legend, only add the scatter object once per category
            # if index == subset.first_valid_index():
            #     scatter_legends.append(scatter)

        # x_temp, y_temp = m(subset.longitude, subset.latitude)
        # scatter = m.scatter(x_temp, y_temp, alpha=0.9, color=color, s=maker_size, label=temp_label, edgecolors='black', linewidth=0.2)
        # scatter_legends.append(scatter)
    
    text_box_string = f"Slice Coordinates:\nLat: {round(lat_min, 2)} - {round(lat_max, 2)}\nLon: {round(lon_min, 2)} - {round(lon_max, 2)}"

    # Add a text box
    plt.text(0.805, 0.015, text_box_string, bbox=dict(facecolor='grey', alpha=0.3, pad=5), 
            horizontalalignment='left', verticalalignment='bottom', 
            transform=plt.gca().transAxes, fontsize=8)

    # Setting labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Wind Farm Data Slice - differentiated in voltage level [#{df.shape[0]}]')
    ax.grid(True)
    ax.legend(handles=scatter_legends, fontsize="x-small", loc='lower left')

    plot_id = str(uuid.uuid4())
    # plt.savefig(plot_save_path + "wind_ms_all_data_plot.png", dpi=plot_save_resolution, bbox_inches='tight', transparent=True)
    plt.savefig(PLOT_SAVE_PATH + f"wind_ms_detail_data_plot_{plot_id}.png", dpi=PLOT_SAVE_RESOLUTION, bbox_inches='tight')
    
    plt.show()


# # RUN PLOT FUNCTION
# data_plot_ULevel(wind_ms_df, lat_min, lat_max, lon_min, lon_max)
detail_data_plot_ULevel(detail_df, detail_df.latitude.min()-0.01, detail_df.latitude.max()+0.01, detail_df.longitude.min()-0.01, detail_df.longitude.max()+0.01)

# %%
detail_df = wind_ms_df[wind_ms_df['cluster'] == 606]
detail_df.shape

# %%
detail_df.to_csv(SUB_DATA_PATH + "detail_data_606.csv", index=False)


# %%
detail_df.columns

# %%
print(detail_df["Name des Anschluss-Netzbetreibers"].unique())
print(detail_df["Name des Anlagenbetreibers (nur Org.)"].unique())

# %%
detail_data_plot_ULevel(detail_df, detail_df.latitude.min()-0.01, detail_df.latitude.max()+0.01, detail_df.longitude.min()-0.01, detail_df.longitude.max()+0.01)

# %%
detail_df.columns

# %%
print("Name des Anlagenbetreibers (nur Org.):")
print(detail_df["Name des Anlagenbetreibers (nur Org.)"].unique())
print()
print("Name des Anschluss-Netzbetreibers:")
print(detail_df["Name des Anschluss-Netzbetreibers"].unique())

# %% [markdown]
# ### Plot Cluster based on "Name des Anlagenbetreibers (nur Org.) or "Name des Anschluss-Netzbetreibers"

# %%
clustered_wind_ms_df.columns

# %%
print(len(clustered_wind_ms_df['Name des Anlagenbetreibers (nur Org.)'].unique()))
clustered_wind_ms_df['Name des Anlagenbetreibers (nur Org.)'].unique()

# %%
detail

# %%


# %%
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def plot_anlagenbetreiber(data, cluster_radius, min_samples):
    # Turn the date column into a datetime object
    data['Inbetriebnahmedatum der Einheit'] = pd.to_datetime(data['Inbetriebnahmedatum der Einheit'], format='%d/%m/%Y')

    # TODO: use threshold dates to plot each cluster devided in "Inbetriebnahme Datum"
    threshold_date_str = ["01/01/2013", "01/01/2016", "01/01/2017", "01/01/2020"]
    threshold_dates = [pd.to_datetime(date) for date in threshold_date_str]

    # Convert latitude and longitude to radians and then to a matrix
    coords = data[['latitude', 'longitude']].to_numpy()
    coords_radiant = np.radians(coords)

    eps_km = cluster_radius 

    # Extract the cluster labels
    # labels = db.labels_
    labels = data.cluster.unique()

    # Get unique cluster labels
    unique_labels = np.unique(labels)

    for cluster in unique_labels:
        if SLICE_DATA:
            # Get mean coordinates for the cluster to place the label
            temp_cluster_df = data[data['cluster'] == cluster]
            mean_x = temp_cluster_df['longitude'].mean()
            mean_y = temp_cluster_df['latitude'].mean()
            x_cluster_label, y_cluster_label = m(mean_x, mean_y)

        
            # Plot the label
            temp_cluster_label = f"{cluster}"
            cluster_label_font_size = 5
            plt.annotate(temp_cluster_label, (x_cluster_label, y_cluster_label), xytext=(5, -5), textcoords='offset points', fontsize=cluster_label_font_size)

    # Create a colormap for the clusters
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # Plot Variables
    image_width_to_height_ratio = (lon_max - lon_min) / (lat_max - lat_min)

    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(8, 8 * image_width_to_height_ratio))

    # Create a Basemap instance that aligns with Google Maps projection and bounds
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                llcrnrlon=lon_min, urcrnrlon=lon_max, lat_ts=20, resolution='i', ax=ax)

    m.drawcoastlines()
    m.drawcountries()


    scatter_legends = []
    color_cache = {}

    if SLICE_DATA:
        temp_marker_size = 1
    else:
        temp_marker_size = 0.1

    # Plot labels for each cluster
    # First print out count for each cluster and define colour
    unique_anlagenbetreiber = data['Name des Anlagenbetreibers (nur Org.)'].unique()
    print(f"Algorithm found: {len(unique_anlagenbetreiber)} cluster with {data.shape[0]} data points.")
    for index, temp_anlagenbetreiber in enumerate(unique_anlagenbetreiber):
        if SLICE_DATA:
            marker_size = 3
        else:
            marker_size = 0.1
        
        # Get the number of data points in the cluster
        anlagenbetreiber_count = data[data['Name des Anlagenbetreibers (nur Org.)'] == temp_anlagenbetreiber].shape[0]

        # threshold_date_marker = markers[index]
        # print(f"Cluster: {temp_anlagenbetreiber} [#{anlagenbetreiber_count}]")

        # Generate a random color
        random_color = get_random_color()
        # color_cache[temp_anlagenbetreiber] = random_color

        temp_sliced_df = data[data['Name des Anlagenbetreibers (nur Org.)'] == temp_anlagenbetreiber]
    
        # Convert lat/lon to map projection coordinates
        x, y = m(temp_sliced_df.longitude, temp_sliced_df.latitude)
        temp_label = f"Anlagenbetreiber {temp_anlagenbetreiber}"
        # temp_scatter_legend = m.scatter(x, y, color=color_cache[cluster], marker=threshold_date_marker, s=2, edgecolors='black', linewidth=0.1, label=temp_cluster_label)
        # temp_scatter_legend = m.scatter(x, y, color=color_cache[temp_anlagenbetreiber], s=temp_marker_size, edgecolors='black', linewidth=0.01, label=temp_label)
        temp_scatter = m.scatter(x, y, color=random_color, marker="o", s=marker_size, edgecolors='black', linewidth=0.01)
        scatter_legends.append(temp_scatter)


    # Setting labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Anlagenbetreiber Wind Turbine Data - {CLUSTER_ALGO_DECISION}, eps_km={eps_km}km, min_samp={min_samples} - [#{data.shape[0]}]')
    ax.grid(True)
    # ax.legend(handles=scatter_legends, fontsize="xx-small", loc='lower left')

    plt.savefig(PLOT_SAVE_PATH + f"windpark_cluster_plot_anlagenbetreiber_{CLUSTER_ALGO_DECISION}_{eps_km}km_{min_samples}_minSamp.png", dpi=1500, bbox_inches='tight')

    # clear_output(wait=True)

    plt.show()


# len(clustered_wind_ms_df.cluster.unique())
plot_anlagenbetreiber(clustered_wind_ms_df, CLUSTER_RADIUS_KM, MIN_SAMPLES)
# clustered_wind_ms_df.columns

# %% [markdown]
# -------------------------------

# %%
count = clustered_wind_ms_df[clustered_wind_ms_df['cluster'] != -1]['cluster'].shape[0]
count2 = clustered_wind_ms_df[clustered_wind_ms_df['cluster'] == -1]['cluster'].shape[0]
print(count)
print(count2)


# %%
clustered_wind_ms_df.cluster.unique()

# %%
import matplotlib.pyplot as plt
import numpy as np

def netzebene_plot_single_marker(df):
    # Define color mapping for 'Spannungsebene_Category' if needed for other visual elements

    # Create a figure with a specific aspect ratio
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust aspect ratio as needed

    # Set the background color as transparent
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Aggregate data to find centroids and counts per cluster
    cluster_groups = df.groupby('cluster')
    cluster_info = cluster_groups[['latitude', 'longitude']].mean()  # Centroid of each cluster
    cluster_sizes = cluster_groups.size()  # Count of values in each cluster

    # Plot one marker per cluster with size based on count
    for cluster_id, centroid in cluster_info.iterrows():
        size = cluster_sizes.loc[cluster_id] * 10  # Scale size of the marker as needed
        ax.scatter(centroid['longitude'], centroid['latitude'], s=size, label=f'Cluster {cluster_id} (n={cluster_sizes.loc[cluster_id]})')

    # Setting labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Clusters based on Coordinates')
    # ax.legend(fontsize="x-small", loc='lower left')

    plt.show()

# Example usage with your DataFrame
# netzebene_plot(wind_df)



# %%
# RUN NETZEBENE PLOT FUNCTION
# netzebene_plot_single_marker(wind_ms_df)

# %%
data = pd.read_csv(DATA_PATH + "cleaned_data.csv")
data.head()

# %%
wind_ms_df.to_csv(DATA_PATH + "clustered_wind_ms_df.csv", index=False)


# %%
cluster_counts = wind_ms_df['cluster'].value_counts()
print(cluster_counts)


# %% [markdown]
# ### Plot cluster and noise

# %%
import matplotlib.pyplot as plt

# Filter the dataframe based on the cluster column
noise_df = wind_ms_df[wind_ms_df['cluster'] == -1]
filtered_df = wind_ms_df[wind_ms_df['cluster'] != -1]

# Create a scatter plot with different colors for each dataframe
plt.scatter(noise_df['longitude'], noise_df['latitude'], color='blue', label='Noise', s=1)
plt.scatter(filtered_df['longitude'], filtered_df['latitude'], color='red', label='Filtered', s=1)

print(f"Noise: {noise_df.shape[0]} | Filtered: {filtered_df.shape[0]}")

# Set the x and y axis labels
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Set the title of the plot
plt.title('Windpark Clustering')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# %%


# %%



