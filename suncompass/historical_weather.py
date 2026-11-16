"""Historical weather lookup for evaluating outdoor imagery.

Uses Open-Meteo's historical reanalysis API (not local weather observations).
See https://open-meteo.com/en/docs/historical-weather-api
"""

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = (
    "cloud_cover",
    "cloud_cover_low",
    "precipitation",
    "direct_normal_irradiance",
    "sunshine_duration",
)


def get_historical_weather(lat, lon, timestamp, *, model="era5", timeout=30):
    """Return hourly historical weather nearest a specified UTC instant.

    Args:
        lat, lon: WGS84 geographic coordinates in decimal degrees.
        timestamp: Timezone-aware datetime; converted to UTC before lookup.
        model: Open-Meteo reanalysis model (default ``era5`` for consistency).
        timeout: HTTP request timeout in seconds.

    The return value is a dictionary with the request time, selected hourly
    weather time, grid-cell coordinates, model, and the following fields:

    - cloud_cover, cloud_cover_low: instantaneous cloud cover in percent.
    - precipitation: mm accumulated over the preceding hour.
    - direct_normal_irradiance: W/m² averaged over the preceding hour.
    - sunshine_duration: seconds over the preceding hour.

    Missing weather values are returned as ``None``. The data describes a
    reanalysis grid cell, not a measurement of whether the sun was visible in
    the camera image. For long recordings, reuse one lookup per hour/area
    rather than calling this function separately for every frame.
    """
    try:
        latitude, longitude = float(lat), float(lon)
    except (TypeError, ValueError) as exc:
        raise ValueError("lat and lon must be numeric") from exc

    if not (math.isfinite(latitude) and -90 <= latitude <= 90):
        raise ValueError("lat must be a finite number between -90 and 90")
    if not (math.isfinite(longitude) and -180 <= longitude <= 180):
        raise ValueError("lon must be a finite number between -180 and 180")
    if not isinstance(timestamp, datetime) or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be a timezone-aware datetime")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty Open-Meteo model name")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    utc_time = timestamp.astimezone(timezone.utc)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": utc_time.date().isoformat(),
        # Include tomorrow so nearest-hour matching works near midnight UTC.
        "end_date": (utc_time + timedelta(days=1)).date().isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "models": model,
    }
    url = ARCHIVE_URL + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "suncompass-historical-weather/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    if payload.get("error"):
        raise ValueError("Open-Meteo error: " + str(payload.get("reason", "unknown")))
    hourly = payload.get("hourly", {})
    timestamps = hourly.get("time", [])
    if not timestamps:
        raise ValueError("Open-Meteo returned no hourly timestamps")

    for variable in HOURLY_VARIABLES:
        if variable not in hourly or len(hourly[variable]) != len(timestamps):
            raise ValueError("Open-Meteo response missing or incomplete: " + variable)

    hourly_times = []
    for value in timestamps:
        hour = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if hour.tzinfo is None:
            hour = hour.replace(tzinfo=timezone.utc)
        hourly_times.append(hour.astimezone(timezone.utc))

    nearest_index = min(
        range(len(hourly_times)),
        key=lambda index: abs(hourly_times[index] - utc_time),
    )
    result = {
        "timestamp_utc": utc_time.isoformat(),
        "weather_hour_utc": hourly_times[nearest_index].isoformat(),
        "requested_latitude": latitude,
        "requested_longitude": longitude,
        "grid_latitude": payload.get("latitude"),
        "grid_longitude": payload.get("longitude"),
        "model": model,
    }
    result.update({variable: hourly[variable][nearest_index] for variable in HOURLY_VARIABLES})
    return result


def main():
    parser = argparse.ArgumentParser(description="Look up historical weather at a GPS location and time")
    parser.add_argument("--lat", type=float, required=True, help="WGS84 latitude in degrees")
    parser.add_argument("--lon", type=float, required=True, help="WGS84 longitude in degrees")
    parser.add_argument(
        "--time", "--timestamp", dest="timestamp", required=True,
        help="ISO 8601 timestamp with timezone, e.g. 2018-08-02T16:14:00Z",
    )
    parser.add_argument("--model", default="era5", help="Open-Meteo model (default: era5)")
    args = parser.parse_args()
    timestamp = datetime.fromisoformat(args.timestamp.replace("Z", "+00:00"))
    result = get_historical_weather(args.lat, args.lon, timestamp, model=args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
