from enum import Enum
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    symbol: str = Field(
        ...,
        description="Cryptocurrency symbol to analyze (e.g., 'BTC', 'ETH', 'BTCUSDT')",
        example="BTC",
        min_length=2,
        max_length=20
    )
    interval: Optional[str] = Field(
        "1h",
        description="Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)",
        example="1h"
    )
    limit: Optional[int] = Field(
        90,
        description="Number of klines to fetch (max 1000)",
        example=90,
        ge=1,
        le=1000
    )

    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v):
        valid_intervals = {
            '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',
            '6h', '8h', '12h', '1d', '3d', '1w', '1M'
        }
        if v not in valid_intervals:
            raise ValueError(f'Invalid interval. Must be one of: {", ".join(sorted(valid_intervals))}')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC",
                "interval": "1h",
                "limit": 90
            }
        }


class AnalyzeResponse(BaseModel):
    symbol: str = Field(..., description="The analyzed trading pair symbol")
    data: Dict[str, Any] = Field(..., description="Raw OHLCV data from Binance")
    technical_indicators: Dict[str, Any] = Field(...,
                                                 description="Traditional technical indicators (RSI, MACD, Bollinger Bands)")
    smc_indicators: Dict[str, Any] = Field(...,
                                           description="Smart Money Concepts indicators (FVG, Order Blocks, Liquidity, etc.)")
    timestamp: str = Field(..., description="ISO timestamp when the analysis was performed")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "data": {
                    "open": [45000.0, 45100.0],
                    "high": [45200.0, 45300.0],
                    "low": [44900.0, 45000.0],
                    "close": [45100.0, 45250.0],
                    "volume": [1000.5, 1200.3],
                    "timestamp": [1640995200000, 1640998800000]
                },
                "technical_indicators": {
                    "rsi": [65.5, 67.2],
                    "macd": {
                        "macd_line": [100.5, 105.2],
                        "signal_line": [98.3, 102.1],
                        "histogram": [2.2, 3.1]
                    },
                    "bollinger_bands": {
                        "upper_band": [46000.0, 46100.0],
                        "middle_band": [45000.0, 45100.0],
                        "lower_band": [44000.0, 44100.0]
                    }
                },
                "smc_indicators": {
                    "fair_value_gaps": {
                        "bullish_fvg": [],
                        "bearish_fvg": []
                    },
                    "swing_highs_lows": {
                        "swing_highs": [],
                        "swing_lows": []
                    }
                },
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class SignalStrength(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class TrendDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


class TechnicalSignal(BaseModel):
    rsi_level: float = Field(..., description="Current RSI value")
    rsi_signal: SignalStrength = Field(..., description="RSI signal interpretation")
    macd_signal: str = Field(..., description="MACD signal (bullish/bearish/neutral)")
    bb_position: str = Field(..., description="Price position relative to Bollinger Bands")
    overall_technical: SignalStrength = Field(..., description="Overall technical analysis signal")


class MarketData(BaseModel):
    current_price: float = Field(..., description="Latest close price")
    price_change_1h: float = Field(..., description="Price change % over 1 hour")
    price_change_4h: float = Field(..., description="Price change % over 4 hours")
    price_change_24h: float = Field(..., description="Price change % over 24 hours")
    volume_trend: str = Field(..., description="Volume trend analysis")
    trend_short: TrendDirection = Field(..., description="Short-term trend direction")
    trend_medium: TrendDirection = Field(..., description="Medium-term trend direction")


class SMCSummary(BaseModel):
    key_support: Optional[float] = Field(None, description="Key support level")
    key_resistance: Optional[float] = Field(None, description="Key resistance level")
    liquidity_sweep: Optional[str] = Field(None, description="Recent liquidity sweep events")
    fvg_count: int = Field(0, description="Number of unfilled fair value gaps")


class TradingSignalResponse(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol")
    market_data: MarketData = Field(..., description="Current market data and trends")
    technical_signals: TechnicalSignal = Field(..., description="Technical indicator signals")
    smc_summary: SMCSummary = Field(..., description="Smart Money Concepts summary")
    overall_signal: SignalStrength = Field(..., description="Overall trading signal")
    confidence: float = Field(..., description="Signal confidence (0-100)", ge=0, le=100)
    timeframe: str = Field(..., description="Analysis timeframe")
    timestamp: str = Field(..., description="Analysis timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "market_data": {
                    "current_price": 45250.0,
                    "price_change_1h": 0.5,
                    "price_change_4h": -1.2,
                    "price_change_24h": 3.8,
                    "volume_trend": "increasing",
                    "trend_short": "BULLISH",
                    "trend_medium": "BULLISH"
                },
                "technical_signals": {
                    "rsi_level": 58.5,
                    "rsi_signal": "NEUTRAL",
                    "macd_signal": "bullish_crossover",
                    "bb_position": "middle_band",
                    "overall_technical": "BUY"
                },
                "smc_summary": {
                    "key_support": 44800.0,
                    "key_resistance": 46500.0,
                    "liquidity_sweep": "recent_lows_swept",
                    "fvg_count": 2
                },
                "overall_signal": "BUY",
                "confidence": 75.0,
                "timeframe": "1h",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class KillzoneRequest(BaseModel):
    date: Optional[str] = Field(
        None,
        description="Target date in YYYY-MM-DD format (default: today)",
        example="2024-01-01"
    )
    count: Optional[int] = Field(
        10,
        description="Number of days to look back from date (default: 10)",
        example=10,
        ge=1,
        le=30
    )
    symbol: Optional[str] = Field(
        "BTCUSDT",
        description="Cryptocurrency symbol to analyze (e.g., 'BTC', 'ETH', 'BTCUSDT')",
        example="BTCUSDT",
        min_length=2,
        max_length=20
    )

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-01-01",
                "count": 10,
                "symbol": "BTCUSDT"
            }
        }


class KillzoneData(BaseModel):
    open: Optional[float] = Field(None, description="Opening price of the kill zone")
    high: Optional[float] = Field(None, description="Highest price of the kill zone")
    low: Optional[float] = Field(None, description="Lowest price of the kill zone")
    close: Optional[float] = Field(None, description="Closing price of the kill zone")

    class Config:
        json_schema_extra = {
            "example": {
                "open": 45000.0,
                "high": 45500.0,
                "low": 44800.0,
                "close": 45200.0
            }
        }


class DailyKillzones(BaseModel):
    Asia: KillzoneData = Field(..., description="Asia kill zone data (00:00-09:00 UTC)")
    London: KillzoneData = Field(..., description="London kill zone data (07:00-16:00 UTC)")
    NewYork: KillzoneData = Field(..., description="New York kill zone data (13:00-22:00 UTC)")

    class Config:
        json_schema_extra = {
            "example": {
                "Asia": {
                    "open": 45000.0,
                    "high": 45500.0,
                    "low": 44800.0,
                    "close": 45200.0
                },
                "London": {
                    "open": 45200.0,
                    "high": 45800.0,
                    "low": 45100.0,
                    "close": 45600.0
                },
                "NewYork": {
                    "open": 45600.0,
                    "high": 46000.0,
                    "low": 45400.0,
                    "close": 45900.0
                }
            }
        }


class KillzoneResponse(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol")
    data: Dict[str, DailyKillzones] = Field(..., description="Kill zone data by date")
    weekStart: str = Field(..., description="Monday of the week for the input date (YYYY-MM-DD format)")
    timestamp: str = Field(..., description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "data": {
                    "2024-01-01": {
                        "Asia": {
                            "open": 45000.0,
                            "high": 45500.0,
                            "low": 44800.0,
                            "close": 45200.0
                        },
                        "London": {
                            "open": 45200.0,
                            "high": 45800.0,
                            "low": 45100.0,
                            "close": 45600.0
                        },
                        "NewYork": {
                            "open": 45600.0,
                            "high": 46000.0,
                            "low": 45400.0,
                            "close": 45900.0
                        }
                    }
                },
                "weekStart": "2023-12-25",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
