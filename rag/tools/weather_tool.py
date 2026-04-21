# rag/tools/weather_tool.py

import requests
from datetime import datetime, timedelta
from config.settings import DEFAULT_WEATHER_LOCATION


def parse_date(query: str) -> tuple:
    """
    Detect if query asks for past, today, or future weather.
    Returns (start_date, end_date, label) as strings YYYY-MM-DD.
    """
    today = datetime.now().date()
    query_lower = query.lower()

    if "yesterday" in query_lower:
        date = today - timedelta(days=1)
        return str(date), str(date), "yesterday"

    elif "tomorrow" in query_lower:
        date = today + timedelta(days=1)
        return str(date), str(date), "tomorrow"

    elif "day after tomorrow" in query_lower:
        date = today + timedelta(days=2)
        return str(date), str(date), "day after tomorrow"

    elif "week" in query_lower or "7 days" in query_lower:
        end = today + timedelta(days=7)
        return str(today), str(end), "next 7 days"

    else:
        # Default to today
        return str(today), str(today), "today"


def get_weather_condition(code: int) -> str:
    """Convert WMO weather code to human readable condition."""
    if code == 0:
        return "Clear Sky ☀️"
    elif code in [1, 2, 3]:
        return "Partly Cloudy ⛅"
    elif code in [45, 48]:
        return "Foggy 🌫️"
    elif code in [51, 53, 55]:
        return "Drizzle 🌦️"
    elif code in [61, 63, 65]:
        return "Rainy 🌧️"
    elif code in [71, 73, 75]:
        return "Snowy ❄️"
    elif code in [80, 81, 82]:
        return "Rain Showers 🌦️"
    elif code in [95, 96, 99]:
        return "Thunderstorm ⛈️"
    else:
        return "Mixed Conditions 🌤️"


def get_weather(location: str = DEFAULT_WEATHER_LOCATION, query: str = "", reranker=None) -> str:
    """
    Fetch weather for any city in India.
    Supports today, yesterday, tomorrow and 7 day forecast.
    """
    start_date, end_date, label = parse_date(query or location)

    # Open-Meteo for all cases
    # Note: wttr.in removed — it returns minimal one-line format with no
    # humidity, high/low, or detailed condition. Open-Meteo returns full data.
    try:
        # Geocode city
        geocode_url = (
            f"https://geocoding-api.open-meteo.com/v1/search"
            f"?name={location}&count=1&country=IN"
        )
        geo_response = requests.get(geocode_url, timeout=5)
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find location: {location}"

        lat       = geo_data["results"][0]["latitude"]
        lon       = geo_data["results"][0]["longitude"]
        city_name = geo_data["results"][0]["name"]

        # ── Today ─────────────────────────────────────────────────────────
        if label == "today":
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,apparent_temperature,"
                f"relative_humidity_2m,wind_speed_10m,weather_code"
                f"&daily=temperature_2m_max,temperature_2m_min"
                f"&forecast_days=1"
                f"&timezone=Asia/Kolkata"
            )
            weather_response = requests.get(weather_url, timeout=5)
            weather_data     = weather_response.json()

            current    = weather_data.get("current", {})
            daily      = weather_data.get("daily",   {})

            temp       = current.get("temperature_2m",      "N/A")
            feels_like = current.get("apparent_temperature", None)
            humidity   = current.get("relative_humidity_2m", "N/A")
            wind       = current.get("wind_speed_10m",       "N/A")
            condition  = get_weather_condition(current.get("weather_code", 0))

            temp_max   = daily.get("temperature_2m_max", [None])[0]
            temp_min   = daily.get("temperature_2m_min", [None])[0]

            feels_str  = f"\n  Feels Like   : {feels_like}°C" if feels_like is not None else ""
            high_low   = f"\n  High / Low   : {temp_max}°C / {temp_min}°C" if temp_max and temp_min else ""

            now_str = datetime.now().strftime("%A, %d %B %Y %I:%M %p")
            return (
                f"📅 Fetched: {now_str}\n"
                f"Current weather in {city_name}, India:\n"
                f"  Condition    : {condition}\n"
                f"  Temperature  : {temp}°C{feels_str}{high_low}\n"
                f"  Humidity     : {humidity}%\n"
                f"  Wind Speed   : {wind} km/h"
            )

        # ── Yesterday ─────────────────────────────────────────────────────
        elif label == "yesterday":
            weather_url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={start_date}&end_date={end_date}"
                f"&daily=temperature_2m_max,temperature_2m_min,"
                f"relative_humidity_2m_max,wind_speed_10m_max,weather_code"
                f"&timezone=Asia/Kolkata"
            )
            weather_response = requests.get(weather_url, timeout=5)
            weather_data     = weather_response.json()

            if weather_data.get("error"):
                return f"Could not fetch yesterday's weather: {weather_data.get('reason')}"

            daily    = weather_data.get("daily", {})
            temp_max = daily.get("temperature_2m_max",      [None])[0]
            temp_min = daily.get("temperature_2m_min",      [None])[0]
            humidity = daily.get("relative_humidity_2m_max",[None])[0]
            wind     = daily.get("wind_speed_10m_max",      [None])[0]
            code     = daily.get("weather_code",            [0])[0]
            condition = get_weather_condition(code or 0)

            return (
                f"Weather in {city_name}, India (yesterday - {start_date}):\n"
                f"  Condition    : {condition}\n"
                f"  Max Temp     : {temp_max}°C\n"
                f"  Min Temp     : {temp_min}°C\n"
                f"  Humidity     : {humidity}%\n"
                f"  Wind Speed   : {wind} km/h"
            )

        # ── Tomorrow / Next 7 days ─────────────────────────────────────────
        else:
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max,temperature_2m_min,"
                f"relative_humidity_2m_max,wind_speed_10m_max,weather_code"
                f"&start_date={start_date}&end_date={end_date}"
                f"&timezone=Asia/Kolkata"
            )
            weather_response = requests.get(weather_url, timeout=5)
            weather_data     = weather_response.json()

            daily    = weather_data.get("daily", {})
            dates    = daily.get("time",                [])
            temp_max = daily.get("temperature_2m_max",  [])
            temp_min = daily.get("temperature_2m_min",  [])
            humidity = daily.get("relative_humidity_2m_max", [])
            wind     = daily.get("wind_speed_10m_max",  [])
            codes    = daily.get("weather_code",        [])

            if not dates:
                return f"No forecast data found for {city_name}."

            lines = [f"Weather forecast for {city_name}, India:"]
            for i, date in enumerate(dates):
                condition = get_weather_condition(codes[i] if i < len(codes) else 0)
                lines.append(
                    f"\n  {date}:\n"
                    f"    Condition : {condition}\n"
                    f"    High      : {temp_max[i] if i < len(temp_max) else 'N/A'}°C\n"
                    f"    Low       : {temp_min[i] if i < len(temp_min) else 'N/A'}°C\n"
                    f"    Humidity  : {humidity[i] if i < len(humidity) else 'N/A'}%\n"
                    f"    Wind      : {wind[i] if i < len(wind) else 'N/A'} km/h"
                )

            return "\n".join(lines)

    except Exception as e:
        return f"Weather fetch failed: {e}"


if __name__ == "__main__":
    print(get_weather("Nagpur",    "today"))
    print("---")
    print(get_weather("Mumbai",    "yesterday"))
    print("---")
    print(get_weather("Delhi",     "tomorrow"))
    print("---")
    print(get_weather("Bangalore", "next 7 days"))

# from rag.tools.sports_tool import get_sports_scores, get_cricket_scores

# print(get_sports_scores("manchester united latest score"))
# print(get_sports_scores("golden state warriors latest score"))
# print(get_sports_scores('south africa cricket team latest score'))


# from rag.tools.stocks_tool import get_stock_price

# test_queries = [
#     "tata consultancy services share price",
#     "reliance share price",
#     "adani port",  
#     "mahindra financial",      # typo test
#     "eternal stock",
#     "apple share price",
#     "nifty 50",
#     "hdfc bank",
#     "infosys",
#     "TSLA",                   # direct ticker
#     "TCS.NS",                 # direct NSE ticker
# ]

# for query in test_queries:
#     print(f"\nQuery: '{query}'")
#     print(get_stock_price(query))
#     print("-" * 40)