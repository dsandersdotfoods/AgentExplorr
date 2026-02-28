"""
API MCP Server — Wrap External APIs as LLM Tools
==================================================

WHAT THIS DOES:
  This MCP server wraps the Open-Meteo weather API — a completely free,
  open-source weather API that requires NO API key. It demonstrates how
  to turn any REST API into MCP tools that an LLM can use.

WHY OPEN-METEO?
  - 100% free, no API key required
  - Open source (https://github.com/open-meteo/open-meteo)
  - Good documentation
  - Reliable uptime
  - Perfect for demos and learning

THE PATTERN:
  The pattern for wrapping ANY API as MCP tools is:
  1. Define the tool with a clear description (so the LLM knows when to use it)
  2. Map tool parameters to API parameters
  3. Make the HTTP request
  4. Format the response for the LLM (JSON → human-readable)

  This same pattern works for:
  - GitHub API → code search, PR management
  - Slack API → send messages, read channels
  - Any REST API → wrap it as MCP tools

LEARNING RESOURCES:
  - Open-Meteo docs: https://open-meteo.com/en/docs
  - httpx (async HTTP client): https://www.python-httpx.org/
  - REST API design: https://restfulapi.net/
  - VIDEO: "REST APIs Explained" — https://www.youtube.com/watch?v=-mN3VyJuCjM
"""

from __future__ import annotations

import httpx
from mcp.server.fastmcp import FastMCP

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Server Setup
# ---------------------------------------------------------------------------

server = FastMCP(
    name="weather-api",
    instructions="Weather data from Open-Meteo (free, no API key). Get current weather and forecasts.",
)

# Open-Meteo API base URL — completely free, no API key!
OPEN_METEO_BASE = "https://api.open-meteo.com/v1"

# Common city coordinates for convenience
# In production, you'd use a geocoding API to convert city names to coordinates
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "new york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "tokyo": (35.6762, 139.6503),
    "paris": (48.8566, 2.3522),
    "sydney": (-33.8688, 151.2093),
    "san francisco": (37.7749, -122.4194),
    "berlin": (52.5200, 13.4050),
    "toronto": (43.6532, -79.3832),
    "singapore": (1.3521, 103.8198),
    "mumbai": (19.0760, 72.8777),
}


def _resolve_coordinates(location: str) -> tuple[float, float] | None:
    """Resolve a city name to latitude/longitude coordinates.

    Args:
        location: City name (case-insensitive).

    Returns:
        (latitude, longitude) tuple, or None if not found.
    """
    return CITY_COORDINATES.get(location.lower().strip())


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------


@server.tool()
def get_current_weather(
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
) -> str:
    """Get current weather conditions for a location.

    Provide either (latitude, longitude) OR a city name.
    Supported cities: New York, London, Tokyo, Paris, Sydney,
    San Francisco, Berlin, Toronto, Singapore, Mumbai.

    Args:
        latitude: Latitude (-90 to 90).
        longitude: Longitude (-180 to 180).
        city: City name (alternative to lat/lon).

    Returns:
        Current weather conditions including temperature, wind, and humidity.
    """
    # Resolve coordinates from city name if provided
    if city and (latitude is None or longitude is None):
        coords = _resolve_coordinates(city)
        if coords is None:
            available = ", ".join(sorted(CITY_COORDINATES.keys()))
            return f"City '{city}' not found. Available cities: {available}"
        latitude, longitude = coords

    if latitude is None or longitude is None:
        return "Error: Provide either (latitude, longitude) or a city name."

    # Build API request
    # Open-Meteo uses query parameters to specify what data you want
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
    }

    logger.info("fetching_weather", lat=latitude, lon=longitude)

    try:
        response = httpx.get(f"{OPEN_METEO_BASE}/forecast", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})

        # Weather codes → human-readable descriptions
        # Full list: https://open-meteo.com/en/docs#weathervariables
        weather_codes: dict[int, str] = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            95: "Thunderstorm",
        }

        weather_code = current.get("weather_code", 0)
        condition = weather_codes.get(weather_code, f"Code {weather_code}")

        location_name = city.title() if city else f"({latitude}, {longitude})"

        return f"""Current Weather for {location_name}:
  Condition:  {condition}
  Temperature: {current.get('temperature_2m', 'N/A')}°F
  Humidity:    {current.get('relative_humidity_2m', 'N/A')}%
  Wind Speed:  {current.get('wind_speed_10m', 'N/A')} mph"""

    except httpx.HTTPStatusError as e:
        return f"API Error: {e.response.status_code} — {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Connection Error: {e}"


@server.tool()
def get_forecast(
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    days: int = 3,
) -> str:
    """Get a multi-day weather forecast.

    Args:
        latitude: Latitude (-90 to 90).
        longitude: Longitude (-180 to 180).
        city: City name (alternative to lat/lon).
        days: Number of forecast days (1-7).

    Returns:
        Daily forecast with high/low temperatures and conditions.
    """
    if city and (latitude is None or longitude is None):
        coords = _resolve_coordinates(city)
        if coords is None:
            return f"City '{city}' not found."
        latitude, longitude = coords

    if latitude is None or longitude is None:
        return "Error: Provide either (latitude, longitude) or a city name."

    days = max(1, min(days, 7))  # Clamp to 1-7

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
        "temperature_unit": "fahrenheit",
        "forecast_days": days,
    }

    try:
        response = httpx.get(f"{OPEN_METEO_BASE}/forecast", params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_probability_max", [])

        location_name = city.title() if city else f"({latitude}, {longitude})"

        lines = [f"{days}-Day Forecast for {location_name}:", ""]
        lines.append(f"  {'Date':<12} {'High':>6} {'Low':>6} {'Rain %':>7}")
        lines.append("  " + "-" * 35)

        for i in range(len(dates)):
            lines.append(
                f"  {dates[i]:<12} {highs[i]:>5.0f}°F {lows[i]:>5.0f}°F {precip[i]:>6}%"
            )

        return "\n".join(lines)

    except httpx.HTTPStatusError as e:
        return f"API Error: {e.response.status_code}"
    except httpx.RequestError as e:
        return f"Connection Error: {e}"


# ---------------------------------------------------------------------------
# Run the server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server.run()
