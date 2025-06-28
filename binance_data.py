import aiohttp
import asyncio
from typing import List, Dict, Any
from datetime import datetime


class BinanceDataFetcher:
    """
    Fetches BTC data from Binance API
    """
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    async def get_klines(self, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 100) -> List[List[str]]:
        """
        Fetch kline/candlestick data from Binance
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT, ETHUSDT)
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of klines to fetch (max 1000)
            
        Returns:
            List of kline data
        """
        url = f"{self.base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to fetch data: {response.status}")
    
    def format_klines_data(self, klines: List[List[str]]) -> Dict[str, List[float]]:
        """
        Format klines data into OHLCV structure
        
        Args:
            klines: Raw klines data from Binance API
            
        Returns:
            Formatted OHLCV data
        """
        formatted_data = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "timestamp": []
        }
        
        for kline in klines:
            formatted_data["timestamp"].append(int(kline[0]))
            formatted_data["open"].append(float(kline[1]))
            formatted_data["high"].append(float(kline[2]))
            formatted_data["low"].append(float(kline[3]))
            formatted_data["close"].append(float(kline[4]))
            formatted_data["volume"].append(float(kline[5]))
            
        return formatted_data