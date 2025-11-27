# Dataset

We use the publicly available river monitoring dataset proposed by the Hydro-informer team. It contains measurements collected from the Topľa River in eastern Slovakia, which is roughly 130 km long. The dataset records hourly observations from four locations along the river between 2008 and the end of 2019. Two major flood events are captured in July 2008 and June 2010, while the period from 2016 to 2018 contains only gap years. The Hydro-informer authors provide a processed version of the data with interpolated gaps and a split that uses data from 2009 through 2018 for training.

For our experiment we rely on the processed dataset, but limit the scope to a single station (Bardejov) so that the data matches our edge deployment constraints. The Bardejov station exposes four input features, each of which is recorded at hourly intervals. The Bardejov station data has 4 features:

| Feature | Symbol | Units | Description |
| --- | --- | --- | --- |
| Precipitation | P | mm | The amount of water (rain, sleet, or snow) expected to fall from the atmosphere. |
| Temperature | T | °C | Air temperature around the station. |
| Discharge rate | Q | m³/s | The volume of water flowing through the river at the station per second. |
| Water level | H | cm | The river water level measured relative to a local benchmark. |

We directly consume these pre-processed readings for training and evaluating our models, focusing on short-term forecasting tasks in the Bardejov watershed.

## Files in this dataset
| Filename | Description |
| --- | --- |
| `data_diff_locations_inter.csv` | Processed Hydro-informer CSV for all four stations with interpolated missing values; provided as the original publication split. |
| `data_diff_locations_no_inter.csv` | Same multi-station structure without interpolation, kept for reference or further experimentation. |
| `one_station_train_data.csv` | Training slice (2009–2015 plus 2019) extracted for the Bardejov station, retaining the hourly timestamp and each of the four features. |
| `one_station_test_data.csv` | Evaluation slice covering the Bardejov station for testing short-term forecasting performance. |
| `data_set_eplanation_and_explore.ipynb` | Notebook outlining the exploratory steps, data filtering, and rationale for station and feature selection. |

## Data layout
The published data is stored as a single CSV with hourly timestamps and a station identifier. We filter that file down to Bardejov, ensuring the rows retain the four columns listed above plus the timestamp. The training/evaluation split follows the Hydro-informer convention by skipping 2016–2018, so downstream pipelines see a consistent feature order and time index.
