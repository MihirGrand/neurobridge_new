"""Weather tool - simple placeholder weather lookup."""

import datetime as dt
import os

import requests


def get_weather(location: str = "") -> str:
    """
    Fetch current weather and air quality data using ipinfo.io for geolocation
    and Open-Meteo for weather + AQI. No API keys required.

    If a location is provided, it is used as a display label. The actual
    data is always based on the server's IP geolocation (ipinfo.io), so the
    weather/AQI reflects where this agent is running.

    If no location is provided, ipinfo.io also supplies the city and region.

    Args:
        location: Optional display label, e.g. "Mumbai" or "Home". If empty,
                  the city from ipinfo.io is used.

    Returns:
        A formatted string with weather details and AQI information.
    """
    # --- Step 1: Get lat/lon from ipinfo.io ---
    try:
        ipinfo_headers = {}
        ipinfo_key = os.getenv("IPINFO_API_KEY")
        if ipinfo_key:
            ipinfo_headers["Authorization"] = f"Bearer {ipinfo_key}"

        ip_resp = requests.get(
            "https://ipinfo.io/json", headers=ipinfo_headers, timeout=10
        )
        ip_resp.raise_for_status()
        ip_data = ip_resp.json()
    except requests.RequestException as e:
        return f"ERROR: Failed to get location from ipinfo.io: {e}"

    loc_str = ip_data.get("loc", "")
    if not loc_str:
        return "ERROR: ipinfo.io did not return location data."

    lat, lon = loc_str.split(",")
    lat = float(lat)
    lon = float(lon)

    # Use provided location or fall back to ipinfo city/region
    display_city = location or ip_data.get("city", "Unknown")
    display_region = ip_data.get("region", "")
    display_country = ip_data.get("country", "")
    display_loc = f"{display_city}, {display_region}, {display_country}"

    # --- Step 2: Fetch weather from Open-Meteo ---
    weather_text = ""
    try:
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,weather_code,wind_speed_10m,wind_direction_10m,"
                    "surface_pressure"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        w = weather_resp.json()
        c = w.get("current", {})

        # Open-Meteo WMO weather codes -> readable descriptions
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Rime fog",
            51: "Light drizzle",
            53: "Drizzle",
            55: "Dense drizzle",
            56: "Freezing drizzle",
            57: "Dense freezing drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with heavy hail",
        }
        code = c.get("weather_code", 0)
        condition = weather_codes.get(code, f"Unknown ({code})")

        # Wind direction in compass
        def _wind_dir(deg):
            dirs = [
                "N",
                "NNE",
                "NE",
                "ENE",
                "E",
                "ESE",
                "SE",
                "SSE",
                "S",
                "SSW",
                "SW",
                "WSW",
                "W",
                "WNW",
                "NW",
                "NNW",
            ]
            if deg is None:
                return "N/A"
            return dirs[round(deg / 22.5) % 16]

        temp = c.get("temperature_2m", "N/A")
        feels_like = c.get("apparent_temperature", "N/A")
        humidity = c.get("relative_humidity_2m", "N/A")
        pressure = c.get("surface_pressure", "N/A")
        wind_speed = c.get("wind_speed_10m", "N/A")
        wind_dir = _wind_dir(c.get("wind_direction_10m"))
        precip = c.get("precipitation", 0)

        weather_text = (
            f"Weather in {display_loc} (lat={lat}, lon={lon})\n"
            f"  Condition : {condition}\n"
            f"  Temp      : {temp} C (feels like {feels_like} C)\n"
            f"  Humidity  : {humidity}%\n"
            f"  Pressure  : {pressure} hPa\n"
            f"  Wind      : {wind_speed} km/h {wind_dir}\n"
            f"  Precip    : {precip} mm"
        )
    except requests.RequestException as e:
        weather_text = f"Weather: Could not fetch data - {e}"

    # --- Step 3: Fetch AQI from Open-Meteo Air Quality API ---
    aqi_text = ""
    try:
        aqi_resp = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide",
                "timezone": "auto",
            },
            timeout=10,
        )
        aqi_resp.raise_for_status()
        aqi_data = aqi_resp.json()
        hourly = aqi_data.get("hourly", {})

        # Find the index for the current hour
        times = hourly.get("time", [])
        now_hour_str = dt.datetime.now().strftime("%Y-%m-%dT%H:00")
        current_idx = None
        for i, t in enumerate(times):
            if t == now_hour_str:
                current_idx = i
                break

        if current_idx is None and times:
            # Fallback: use the last available hour
            current_idx = len(times) - 1

        if current_idx is not None:

            def _val(key):
                vals = hourly.get(key, [])
                if current_idx < len(vals):
                    v = vals[current_idx]
                    return f"{v}" if v is not None else "N/A"
                return "N/A"

            pm10 = _val("pm10")
            pm25 = _val("pm2_5")
            no2 = _val("nitrogen_dioxide")
            so2 = _val("sulphur_dioxide")
            o3 = _val("ozone")
            co = _val("carbon_monoxide")

            # Simple AQI estimation from PM2.5 (US EPA scale)
            def _aqi_from_pm25(v):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return None
                if v <= 12.0:
                    return int(v * 50 / 12)
                elif v <= 35.4:
                    return int(50 + (v - 12.0) * 50 / 23.4)
                elif v <= 55.4:
                    return int(100 + (v - 35.4) * 50 / 20.0)
                elif v <= 150.4:
                    return int(150 + (v - 55.4) * 100 / 95.0)
                elif v <= 250.4:
                    return int(200 + (v - 150.4) * 100 / 100.0)
                elif v <= 350.4:
                    return int(300 + (v - 250.4) * 100 / 100.0)
                else:
                    return int(400 + (v - 350.4) * 500 / 150.0)

            def _aqi_label(aqi):
                if aqi is None:
                    return "N/A"
                if aqi <= 50:
                    return "Good"
                elif aqi <= 100:
                    return "Moderate"
                elif aqi <= 150:
                    return "Unhealthy for Sensitive Groups"
                elif aqi <= 200:
                    return "Unhealthy"
                elif aqi <= 300:
                    return "Very Unhealthy"
                else:
                    return "Hazardous"

            aqi_val = _aqi_from_pm25(pm25)
            aqi_label = _aqi_label(aqi_val)
            aqi_display = str(aqi_val) if aqi_val is not None else "N/A"

            aqi_text = (
                f"\n\nAir Quality Index (current hour)\n"
                f"  AQI        : {aqi_display} - {aqi_label}\n"
                f"  PM2.5      : {pm25} ug/m3\n"
                f"  PM10       : {pm10} ug/m3\n"
                f"  NO2        : {no2} ug/m3\n"
                f"  SO2        : {so2} ug/m3\n"
                f"  O3         : {o3} ug/m3\n"
                f"  CO         : {co} ug/m3"
            )
        else:
            aqi_text = "\n\nAQI: No hourly data available for current time."
    except requests.RequestException as e:
        aqi_text = f"\n\nAQI: Could not fetch data - {e}"

    return weather_text + aqi_text
