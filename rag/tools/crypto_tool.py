# rag/tools/crypto_tool.py

import requests
import certifi
from typing import Optional

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

COIN_MAPPING = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    "cardano": "cardano",
    "ada": "cardano",
    "bnb": "binancecoin",
    "binance": "binancecoin",
    "xrp": "ripple",
    "ripple": "ripple",
    "polygon": "matic-network",
    "matic": "matic-network",
    "shiba": "shiba-inu",
    "shib": "shiba-inu",
}


def get_crypto_price(coin: str, reranker=None) -> str:
    coin = coin.strip().lower()
    coin_id = COIN_MAPPING.get(coin, coin)

    try:
        url = f"{COINGECKO_BASE}/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd,inr",
            "include_24hr_change": "true"
        }
        response = requests.get(url, params=params, timeout=10, verify=certifi.where())  # ✅ fix here
        data = response.json()

        if coin_id not in data:
            return f"Could not find crypto: {coin}"

        price_data = data[coin_id]
        usd = price_data.get("usd", 0)
        inr = price_data.get("inr", 0)
        change = price_data.get("usd_24h_change", 0)

        change_symbol = "📈" if change >= 0 else "📉"

        return (
            f"💰 {coin.title()}\n"
            f"USD: ${usd:,.2f}\n"
            f"INR: ₹{inr:,.2f}\n"
            f"24h Change: {change_symbol} {change:.2f}%"
        )

    except Exception as e:
        return f"Crypto fetch failed: {e}"