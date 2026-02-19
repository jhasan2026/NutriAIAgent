from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime
import math
import requests
import pandas as pd

from langchain_core.tools import tool

# ---------------------------
# Helper: season by month
# ---------------------------
def infer_season(month: int, hemisphere: str = "north") -> str:
    # Very simple mapping. You can improve for Bangladesh/local contexts later.
    if hemisphere.lower() == "south":
        # swap seasons
        if month in (12, 1, 2): return "summer"
        if month in (3, 4, 5): return "autumn"
        if month in (6, 7, 8): return "winter"
        return "spring"
    else:
        if month in (12, 1, 2): return "winter"
        if month in (3, 4, 5): return "spring"
        if month in (6, 7, 8): return "summer"
        return "autumn"


# ---------------------------
# TOOL 1: User location
# ---------------------------
@tool("get_user_location")
def get_user_location(country_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Get approximate user location using IP-based geolocation.
    No API key required (uses ipapi.co). If it fails, returns unknown.
    """
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=6)
        resp.raise_for_status()
        data = resp.json()
        return {
            "status": "ok",
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country_name"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "source": "ipapi.co",
            "note": "Approximate IP-based location."
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "city": None,
            "region": None,
            "country": country_hint,
            "latitude": None,
            "longitude": None,
            "timezone": None
        }


# ---------------------------
# TOOL 2: Weather + season
# ---------------------------
@tool("get_weather_and_season")
def get_weather_and_season(latitude: float, longitude: float, hemisphere: str = "north") -> Dict[str, Any]:
    """
    Current weather + inferred season.
    Uses Open-Meteo (no key needed).
    """
    season = infer_season(datetime.utcnow().month, hemisphere=hemisphere)
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
        )
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        return {
            "status": "ok",
            "season": season,
            "temperature_c": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "wind_kph": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
            "source": "open-meteo.com"
        }
    except Exception as e:
        return {
            "status": "error",
            "season": season,
            "error": str(e)
        }


# ---------------------------
# TOOL 3: Local food prices
# ---------------------------
@tool("get_local_food_prices")
def get_local_food_prices(
    location_name: str,
    csv_path: str = "sample_data/local_food_prices.csv",
    max_items: int = 30
) -> Dict[str, Any]:
    """
    Retrieve local food prices from a local CSV (offline-friendly).
    CSV columns recommended: location, item, unit, price, currency
    """
    try:
        df = pd.read_csv(csv_path)
        df_loc = df[df["location"].str.lower() == location_name.lower()].copy()
        if df_loc.empty:
            # fallback: return most common location sample
            df_loc = df.copy()

        df_loc = df_loc.sort_values("price", ascending=True).head(max_items)
        items = df_loc[["item", "unit", "price", "currency"]].to_dict(orient="records")
        return {"status": "ok", "location": location_name, "items": items, "source": "local_csv"}
    except Exception as e:
        return {"status": "error", "location": location_name, "error": str(e)}
