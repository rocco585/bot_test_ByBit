from pybit.unified_trading import HTTP

TESTNET = True   # ВАЖНО: demo = True

API_KEY = "qGPP7E6AF0I2HGpPEl"
API_SECRET = "NqYF10DDGjpgF1PsU1D7bGSDylXojgKuJroQ"

session = HTTP(
    testnet=TESTNET,
    api_key=API_KEY.strip(),
    api_secret=API_SECRET.strip()
)

print("MODE:", "TESTNET")

print("\n--- PUBLIC (должно работать всегда) ---")
print(session.get_tickers(category="linear", symbol="BTCUSDT"))

print("\n--- PRIVATE (ключи) ---")
print(session.get_positions(category="linear", symbol="BTCUSDT"))
