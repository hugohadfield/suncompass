"""Offline tests for the Open-Meteo historical weather utility."""

import io
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from suncompass.historical_weather import get_historical_weather


class TestHistoricalWeather(unittest.TestCase):
    @staticmethod
    def payload(times=None):
        times = times or ["2018-08-02T16:00", "2018-08-02T17:00", "2018-08-03T00:00"]
        return {
            "latitude": 37.75,
            "longitude": -122.5,
            "hourly": {
                "time": times,
                "cloud_cover": [90, 40, 10],
                "cloud_cover_low": [80, 30, 5],
                "precipitation": [0.3, 0.0, 0.0],
                "direct_normal_irradiance": [15.0, 400.0, 0.0],
                "sunshine_duration": [0.0, 3300.0, 0.0],
            },
        }

    @staticmethod
    def mock_response(payload):
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    @patch("suncompass.historical_weather.urlopen")
    def test_selects_nearest_hour_and_converts_to_utc(self, urlopen):
        urlopen.return_value = self.mock_response(self.payload())
        # 09:14 local time is 16:14 UTC.
        local = datetime(2018, 8, 2, 9, 14, tzinfo=timezone(timedelta(hours=-7)))
        result = get_historical_weather(37.7255, -122.4721, local)
        self.assertEqual(result["timestamp_utc"], "2018-08-02T16:14:00+00:00")
        self.assertEqual(result["weather_hour_utc"], "2018-08-02T16:00:00+00:00")
        self.assertEqual(result["cloud_cover_low"], 80)
        self.assertEqual(result["direct_normal_irradiance"], 15.0)
        self.assertEqual(result["grid_latitude"], 37.75)
        self.assertEqual(result["model"], "era5")

        request = urlopen.call_args.args[0]
        params = parse_qs(urlsplit(request.full_url).query)
        self.assertEqual(params["timezone"], ["UTC"])
        self.assertEqual(params["models"], ["era5"])
        self.assertEqual(params["start_date"], ["2018-08-02"])
        self.assertEqual(params["end_date"], ["2018-08-03"])
        self.assertIn("sunshine_duration", params["hourly"][0])

    @patch("suncompass.historical_weather.urlopen")
    def test_nearest_hour_can_be_next_utc_day(self, urlopen):
        payload = self.payload()
        payload["hourly"]["time"] = ["2018-08-02T23:00", "2018-08-03T00:00", "2018-08-03T01:00"]
        urlopen.return_value = self.mock_response(payload)
        result = get_historical_weather(37, -122, datetime(2018, 8, 2, 23, 50, tzinfo=timezone.utc))
        self.assertEqual(result["weather_hour_utc"], "2018-08-03T00:00:00+00:00")
        self.assertEqual(result["cloud_cover"], 40)

    @patch("suncompass.historical_weather.urlopen")
    def test_null_weather_value_is_preserved(self, urlopen):
        payload = self.payload()
        payload["hourly"]["cloud_cover_low"][0] = None
        urlopen.return_value = self.mock_response(payload)
        result = get_historical_weather(37, -122, datetime(2018, 8, 2, 16, 10, tzinfo=timezone.utc))
        self.assertIsNone(result["cloud_cover_low"])

    def test_rejects_naive_time_and_invalid_coordinates(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            get_historical_weather(37, -122, datetime(2018, 8, 2))
        for lat, lon in [(91, 0), (0, -181), (float("nan"), 0), (0, float("inf"))]:
            with self.subTest(lat=lat, lon=lon):
                with self.assertRaises(ValueError):
                    get_historical_weather(lat, lon, datetime(2018, 8, 2, tzinfo=timezone.utc))

    @patch("suncompass.historical_weather.urlopen")
    def test_api_error_is_explicit(self, urlopen):
        urlopen.return_value = self.mock_response({"error": True, "reason": "unsupported model"})
        with self.assertRaisesRegex(ValueError, "unsupported model"):
            get_historical_weather(37, -122, datetime(2018, 8, 2, tzinfo=timezone.utc))

    @patch("suncompass.historical_weather.urlopen")
    def test_incomplete_response_is_rejected(self, urlopen):
        payload = self.payload()
        del payload["hourly"]["sunshine_duration"]
        urlopen.return_value = self.mock_response(payload)
        with self.assertRaisesRegex(ValueError, "sunshine_duration"):
            get_historical_weather(37, -122, datetime(2018, 8, 2, 16, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
