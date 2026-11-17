# Suncompass

This repository contains code for estimating the direction of the sun from a single outdoor image.
![Suncompass](Figure_1.png)

## Why?

In many areas of the world the sun shines a lot and humans have used it as a navigational aid for thousands of years. So maybe it would be fun to imbue our robotic creations with this ability too.

Applications:
- You can estimate the yaw angle between multiple cameras on the same rig even if they don't have overlap
- You can stabilise SLAM algorithms by adding an absolute yaw reference
- If you know the time, you know which way the sun is relative to you, and you have a reasonable estimate of where in the world you are then you can tell global heading

### What happens when the sun isn't shining?
The algorithm doesn't work 🤷‍♂️

## Repository Contents

1. **pretrained_resnet.py**: Defines a regression model based on a pre-trained ResNet model and includes functions for training and loading the model.
2. **3_224_224_resnet.pth**: A pre-trained ResNet model for estimating the direction of the sun from a single outdoor image.
3. **suncompass.py**: Class for interacting with the pre-trained ResNet model.
4. **demo.py**: Demonstrates how to use the pre-trained ResNet model to estimate the direction of the sun from a single outdoor image.

## Installation

### From pypi
```sh
pip install suncompass
```

### Manual installation

1. Clone the repository:
    ```sh
    git clone https://github.com/hugohadfield/suncompass.git
    cd suncompass
    ```

2. Install the package:
    ```sh
    pip install .
    ```

## Example usage

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

from suncompass import SunCompass

image_path = "croatia.jpg"
suncompass = SunCompass()
suncompass.set_eval(dropout=False)

image = cv2.imread(image_path)
img_with_suncompass, theta_rad = suncompass.predict_and_draw(image)
plt.figure()
plt.imshow(cv2.cvtColor(img_with_suncompass, cv2.COLOR_BGR2RGB))
plt.title(f"Sun direction: {np.degrees(theta_rad):.1f} degrees")
plt.show()
```


## Requirements

- Python 3.7+
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib
- tqdm
- click
- PIL
- tensorboard
- suncalc

## Usage

### add_sun_vector.py

This script calculates the sun vector in the ENU frame and the relative frame based on heading.

Example usage:
```sh
python add_sun_vector.py
```

### extract_motion.py

This script extracts motion parameters such as speed, heading, and curvature from a given dataset.

Example usage:
```sh
python extract_motion.py
```

### pretrained_resnet.py

This script defines and trains a regression model based on a pre-trained ResNet model.

Example usage:
```sh
python pretrained_resnet.py
```

### run_model_on_images.py

This script runs the trained ResNet model on a set of images and saves the results to a CSV file.

Example usage:
```sh
python run_model_on_images.py --model_name 3_224_224_resnet.pth --image_dir /path/to/images --gt_csv /path/to/ground_truth.csv
```

## Function Details

### add_sun_vector.py

- `calculate_sun_vector_ENU(time: datetime, lat_degs: float, long_degs: float) -> np.ndarray`: Calculates the sun vector in the ENU frame.
- `calculate_sun_vector_relative(time_unixus: int, lat_degs: float, long_degs: float, heading_rads_from_east: Union[float, np.complex128]) -> np.ndarray`: Calculates the sun vector in the relative frame.
- `calculate_relative_azimuth_angle(time_unixus: int, lat_degs: float, long_degs: float, heading_rads_from_east: float) -> float`: Calculates the azimuth angle of the sun vector in the relative frame.

### extract_motion.py

- `haversine(lat1, lon1, lat2, lon2) -> float`: Calculates the great circle distance between two points on the Earth.
- `calculate_bearing_clockwise_from_north(lat1, lon1, lat2, lon2) -> float`: Calculates the bearing between two points.
- `calculate_heading_anticlockwise_from_east(lat1, lon1, lat2, lon2) -> float`: Calculates the heading anti-clockwise from east.
- `extract_motion(data: pd.DataFrame) -> pd.DataFrame`: Extracts speed, heading, and curvature from the given data.

### pretrained_resnet.py

- `ResNetRegression`: Defines a regression model based on a pre-trained ResNet model.
- `train_network(device, n_epochs: int = 10, image_size: Tuple[int, int] = (128, 128))`: Trains the network for a set number of epochs.

### run_model_on_images.py

- `run_model_on_images(model_name: Path, image_dir: Path, image_size: Tuple[int, int, int] = (1, 100, 100))`: Runs the model on the images and saves the results to a CSV file.
- `plot_results_csv(gt_csv: Optional[str] = None)`: Plots the results from the CSV file.

## Historical weather lookup

SunCompass includes `suncompass.historical_weather.get_historical_weather()` for screening
outdoor image datasets by historical weather. It uses the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
with the ERA5 reanalysis model by default. No API key or additional Python
package is required.

From a clone where SunCompass is installed (`pip install -e .`):

```bash
python -m suncompass.historical_weather \
  --lat 37.7255 --lon -122.4721 \
  --time 2018-08-02T16:14:00Z
```

Or from Python:

```python
from datetime import datetime, timezone
from suncompass.historical_weather import get_historical_weather

weather = get_historical_weather(
    37.7255, -122.4721,
    datetime(2018, 8, 2, 16, 14, tzinfo=timezone.utc),
)
print(weather)
```

The response includes `timestamp_utc`, `weather_hour_utc`, the requested
coordinates, the model grid-cell coordinates, and these hourly weather fields:

- `cloud_cover` and `cloud_cover_low`: instantaneous cloud coverage (%).
- `precipitation`: precipitation during the preceding hour (mm).
- `direct_normal_irradiance`: mean direct normal solar irradiance during the
  preceding hour (W/m²).
- `sunshine_duration`: sunshine during the preceding hour (seconds).

The nearest UTC hourly timestamp is selected, including across UTC midnight;
missing weather values remain `None`. The input time **must** include a timezone.
Weather is *reanalysis data at model-grid resolution*, not a local observation
or proof that the sun was visible in an image. In particular, fog may vary
substantially within a model grid cell. For a high-frame-rate driving sequence,
reuse one hourly weather lookup across nearby frames rather than making one
API request per frame.

Run the offline unit tests with:

```bash
python -m unittest discover -s tests -v
```

## License

This project is licensed under the MIT License - see the [licence.txt](licence.txt) file for details.

## Research workflows

- [WayveScenes101 evaluation](PILOT.md)
