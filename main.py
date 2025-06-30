import os
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from binance_data import BinanceDataFetcher
from smc_analysis import SMCAnalyzer
from technical_analysis import TechnicalAnalyzer

# Get API key from environment variable
API_KEY = os.getenv("API_KEY", "")

# Define API key security scheme for Swagger
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key for protected endpoints"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

app = FastAPI(
    title="Cryptocurrency Technical Analysis API",
    description="A comprehensive API for analyzing cryptocurrency data using technical indicators and Smart Money Concepts (SMC)",
    version="1.0.0",
    contact={
        "name": "DataMaker API",
        "url": "https://github.com/mchen81/cat-datamaker",
    },
    license_info={
        "name": "MIT",
    },
)


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


def calculate_trend_direction(close_prices: List[float], short_period: int = 5, medium_period: int = 20) -> tuple[
    TrendDirection, TrendDirection]:
    """Calculate short and medium term trend directions"""
    if len(close_prices) < medium_period:
        return TrendDirection.SIDEWAYS, TrendDirection.SIDEWAYS

    current_price = close_prices[-1]
    short_avg = sum(close_prices[-short_period:]) / short_period
    medium_avg = sum(close_prices[-medium_period:]) / medium_period

    # Short term trend
    short_trend = TrendDirection.SIDEWAYS
    if current_price > short_avg * 1.002:  # 0.2% threshold
        short_trend = TrendDirection.BULLISH
    elif current_price < short_avg * 0.998:
        short_trend = TrendDirection.BEARISH

    # Medium term trend
    medium_trend = TrendDirection.SIDEWAYS
    if current_price > medium_avg * 1.005:  # 0.5% threshold
        medium_trend = TrendDirection.BULLISH
    elif current_price < medium_avg * 0.995:
        medium_trend = TrendDirection.BEARISH

    return short_trend, medium_trend


def determine_technical_sentiment(technical_indicators: Dict[str, Any]) -> tuple[TechnicalSignal, SignalStrength]:
    """Analyze technical indicators and determine overall sentiment"""
    # Extract current values (assuming they're at the end of arrays)
    rsi_values = technical_indicators.get('rsi', [50])
    macd_data = technical_indicators.get('macd', {})
    bb_data = technical_indicators.get('bollinger_bands', {})

    current_rsi = rsi_values[-1] if rsi_values else 50

    # RSI Signal
    rsi_signal = SignalStrength.NEUTRAL
    if current_rsi > 70:
        rsi_signal = SignalStrength.SELL
    elif current_rsi > 60:
        rsi_signal = SignalStrength.NEUTRAL
    elif current_rsi < 30:
        rsi_signal = SignalStrength.BUY
    elif current_rsi < 40:
        rsi_signal = SignalStrength.NEUTRAL

    # MACD Signal
    macd_signal = "neutral"
    if macd_data:
        macd_line = macd_data.get('macd_line', [])
        signal_line = macd_data.get('signal_line', [])
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
                macd_signal = "bullish_crossover"
            elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
                macd_signal = "bearish_crossover"
            elif macd_line[-1] > signal_line[-1]:
                macd_signal = "bullish"
            else:
                macd_signal = "bearish"

    # Bollinger Bands position
    bb_position = "middle_band"
    if bb_data:
        upper_band = bb_data.get('upper_band', [])
        lower_band = bb_data.get('lower_band', [])
        middle_band = bb_data.get('middle_band', [])

        if upper_band and lower_band and middle_band:
            current_upper = upper_band[-1]
            current_lower = lower_band[-1]
            current_middle = middle_band[-1]

            # Need current price to determine position - will be passed separately
            bb_position = "middle_band"  # Default

    # Calculate overall technical sentiment
    signals = []
    if rsi_signal in [SignalStrength.BUY, SignalStrength.STRONG_BUY]:
        signals.append(1)
    elif rsi_signal in [SignalStrength.SELL, SignalStrength.STRONG_SELL]:
        signals.append(-1)
    else:
        signals.append(0)

    if "bullish" in macd_signal:
        signals.append(1)
    elif "bearish" in macd_signal:
        signals.append(-1)
    else:
        signals.append(0)

    avg_signal = sum(signals) / len(signals) if signals else 0

    if avg_signal > 0.5:
        overall_technical = SignalStrength.BUY
    elif avg_signal < -0.5:
        overall_technical = SignalStrength.SELL
    else:
        overall_technical = SignalStrength.NEUTRAL

    technical_signal = TechnicalSignal(
        rsi_level=current_rsi,
        rsi_signal=rsi_signal,
        macd_signal=macd_signal,
        bb_position=bb_position,
        overall_technical=overall_technical
    )

    return technical_signal, overall_technical


def extract_key_smc_levels(smc_indicators: Dict[str, Any], current_price: float) -> SMCSummary:
    """Extract key SMC levels and information"""
    support = None
    resistance = None
    liquidity_sweep = None
    fvg_count = 0

    # Extract swing highs and lows for support/resistance
    swing_data = smc_indicators.get('swing_highs_lows', {})
    if swing_data:
        swing_lows = swing_data.get('swing_lows', [])
        swing_highs = swing_data.get('swing_highs', [])

        # Find closest support (swing low below current price)
        valid_supports = [low for low in swing_lows if low < current_price]
        if valid_supports:
            support = max(valid_supports)  # Closest support below

        # Find closest resistance (swing high above current price)
        valid_resistances = [high for high in swing_highs if high > current_price]
        if valid_resistances:
            resistance = min(valid_resistances)  # Closest resistance above

    # Count fair value gaps
    fvg_data = smc_indicators.get('fair_value_gaps', {})
    if fvg_data:
        bullish_fvg = fvg_data.get('bullish_fvg', [])
        bearish_fvg = fvg_data.get('bearish_fvg', [])
        fvg_count = len(bullish_fvg) + len(bearish_fvg)

    return SMCSummary(
        key_support=support,
        key_resistance=resistance,
        liquidity_sweep=liquidity_sweep,
        fvg_count=fvg_count
    )


def calculate_price_changes(close_prices: List[float], timestamps: List[int], timeframe: str) -> Dict[str, float]:
    """Calculate price changes over different periods"""
    if len(close_prices) < 2:
        return {"1h": 0.0, "4h": 0.0, "24h": 0.0}

    current_price = close_prices[-1]

    # Calculate based on timeframe
    timeframe_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
        '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
    }

    interval_minutes = timeframe_minutes.get(timeframe, 60)

    # Calculate how many candles back for each period
    candles_1h = max(1, 60 // interval_minutes)
    candles_4h = max(1, 240 // interval_minutes)
    candles_24h = max(1, 1440 // interval_minutes)

    def get_price_change(candles_back: int) -> float:
        if len(close_prices) <= candles_back:
            return 0.0
        old_price = close_prices[-candles_back - 1]
        return ((current_price - old_price) / old_price) * 100 if old_price > 0 else 0.0

    return {
        "1h": get_price_change(candles_1h),
        "4h": get_price_change(candles_4h),
        "24h": get_price_change(candles_24h)
    }


def analyze_volume_trend(volume_data: List[float]) -> str:
    """Analyze volume trend"""
    if len(volume_data) < 10:
        return "insufficient_data"

    recent_volume = sum(volume_data[-5:]) / 5
    older_volume = sum(volume_data[-10:-5]) / 5

    if recent_volume > older_volume * 1.2:
        return "increasing"
    elif recent_volume < older_volume * 0.8:
        return "decreasing"
    else:
        return "stable"


def process_for_trading_decision(
        symbol: str,
        formatted_data: Dict[str, List[float]],
        technical_indicators: Dict[str, Any],
        smc_indicators: Dict[str, Any],
        timeframe: str
) -> TradingSignalResponse:
    """
    Main function to process comprehensive analysis data into GPT-optimized trading signals
    """
    # Extract price and volume data
    close_prices = formatted_data.get('close', [])
    volume_data = formatted_data.get('volume', [])
    timestamps = formatted_data.get('timestamp', [])

    if not close_prices:
        raise ValueError("No price data available")

    current_price = close_prices[-1]

    # Calculate price changes
    price_changes = calculate_price_changes(close_prices, timestamps, timeframe)

    # Analyze trends
    short_trend, medium_trend = calculate_trend_direction(close_prices)

    # Analyze volume
    volume_trend = analyze_volume_trend(volume_data)

    # Create market data
    market_data = MarketData(
        current_price=current_price,
        price_change_1h=price_changes["1h"],
        price_change_4h=price_changes["4h"],
        price_change_24h=price_changes["24h"],
        volume_trend=volume_trend,
        trend_short=short_trend,
        trend_medium=medium_trend
    )

    # Analyze technical indicators
    technical_signal, tech_sentiment = determine_technical_sentiment(technical_indicators)

    # Update Bollinger Bands position with current price
    bb_data = technical_indicators.get('bollinger_bands', {})
    if bb_data:
        upper_band = bb_data.get('upper_band', [])
        lower_band = bb_data.get('lower_band', [])
        middle_band = bb_data.get('middle_band', [])

        if upper_band and lower_band and middle_band:
            current_upper = upper_band[-1]
            current_lower = lower_band[-1]
            current_middle = middle_band[-1]

            if current_price > current_upper:
                technical_signal.bb_position = "above_upper_band"
            elif current_price < current_lower:
                technical_signal.bb_position = "below_lower_band"
            elif current_price > current_middle:
                technical_signal.bb_position = "above_middle_band"
            else:
                technical_signal.bb_position = "below_middle_band"

    # Extract SMC summary
    smc_summary = extract_key_smc_levels(smc_indicators, current_price)

    # Calculate overall signal and confidence
    signals = []
    weights = []

    # Technical signals (weight: 40%)
    if tech_sentiment == SignalStrength.STRONG_BUY:
        signals.append(2)
    elif tech_sentiment == SignalStrength.BUY:
        signals.append(1)
    elif tech_sentiment == SignalStrength.SELL:
        signals.append(-1)
    elif tech_sentiment == SignalStrength.STRONG_SELL:
        signals.append(-2)
    else:
        signals.append(0)
    weights.append(0.4)

    # Trend signals (weight: 30%)
    trend_signal = 0
    if short_trend == TrendDirection.BULLISH and medium_trend == TrendDirection.BULLISH:
        trend_signal = 2
    elif short_trend == TrendDirection.BULLISH or medium_trend == TrendDirection.BULLISH:
        trend_signal = 1
    elif short_trend == TrendDirection.BEARISH and medium_trend == TrendDirection.BEARISH:
        trend_signal = -2
    elif short_trend == TrendDirection.BEARISH or medium_trend == TrendDirection.BEARISH:
        trend_signal = -1
    signals.append(trend_signal)
    weights.append(0.3)

    # Price momentum signals (weight: 20%)
    momentum_signal = 0
    if price_changes["1h"] > 2 and price_changes["4h"] > 1:
        momentum_signal = 1
    elif price_changes["1h"] < -2 and price_changes["4h"] < -1:
        momentum_signal = -1
    signals.append(momentum_signal)
    weights.append(0.2)

    # Volume confirmation (weight: 10%)
    volume_signal = 0
    if volume_trend == "increasing":
        # Volume increasing strengthens the current trend
        if sum(signals) > 0:
            volume_signal = 1
        elif sum(signals) < 0:
            volume_signal = -1
    signals.append(volume_signal)
    weights.append(0.1)

    # Calculate weighted average
    weighted_sum = sum(s * w for s, w in zip(signals, weights))

    # Determine overall signal
    if weighted_sum > 1.0:
        overall_signal = SignalStrength.STRONG_BUY
        confidence = min(95, 60 + abs(weighted_sum) * 20)
    elif weighted_sum > 0.3:
        overall_signal = SignalStrength.BUY
        confidence = min(85, 50 + abs(weighted_sum) * 25)
    elif weighted_sum < -1.0:
        overall_signal = SignalStrength.STRONG_SELL
        confidence = min(95, 60 + abs(weighted_sum) * 20)
    elif weighted_sum < -0.3:
        overall_signal = SignalStrength.SELL
        confidence = min(85, 50 + abs(weighted_sum) * 25)
    else:
        overall_signal = SignalStrength.NEUTRAL
        confidence = max(30, 50 - abs(weighted_sum) * 30)

    return TradingSignalResponse(
        symbol=symbol,
        market_data=market_data,
        technical_signals=technical_signal,
        smc_summary=smc_summary,
        overall_signal=overall_signal,
        confidence=round(confidence, 1),
        timeframe=timeframe,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/",
         summary="API Health Check",
         description="Returns basic information about the API",
         response_description="API status and information")
async def root():
    """Root endpoint that returns API information and status."""
    return {
        "message": "Cryptocurrency Technical Analysis API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "/docs": "Swagger UI documentation",
            "/redoc": "ReDoc documentation",
            "/v1/analyze": "Comprehensive analysis endpoint",
            "/v1/trading-signal": "GPT-optimized trading signals endpoint"
        }
    }


@app.post("/v1/analyze",
          # response_model=AnalyzeResponse,
          summary="Analyze Cryptocurrency",
          description="Fetch cryptocurrency data from Binance and perform comprehensive technical analysis including traditional indicators and Smart Money Concepts",
          response_description="Complete analysis results with technical and SMC indicators")
async def analyze_crypto(request: AnalyzeRequest, _: bool = Depends(verify_api_key)):
    """
    Analyze cryptocurrency data with comprehensive technical indicators.
    
    This endpoint:
    1. Fetches klines data from Binance for the specified symbol, interval, and limit
    2. Calculates traditional technical indicators (RSI, MACD, Bollinger Bands)
    3. Performs Smart Money Concepts analysis (Fair Value Gaps, Order Blocks, Liquidity, etc.)
    4. Returns all data and calculated indicators
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., BTC, ETH, ADA, etc.)
    - **interval**: Optional kline interval (default: 1h) - Valid values: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    - **limit**: Optional number of klines to fetch (default: 90, max: 1000)
    
    **Note:** If the symbol doesn't end with 'USDT', it will be automatically appended.
    """
    try:
        # Initialize analyzers
        binance_fetcher = BinanceDataFetcher()
        technical_analyzer = TechnicalAnalyzer()
        smc_analyzer = SMCAnalyzer()

        # Fetch data from Binance using request parameters
        symbol = request.symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        klines_data = await binance_fetcher.get_klines(
            symbol=symbol,
            interval=request.interval,
            limit=request.limit
        )
        formatted_data = binance_fetcher.format_klines_data(klines_data)

        # Run technical analysis
        technical_indicators = technical_analyzer.analyze_all_indicators(formatted_data)

        # Run SMC analysis
        smc_indicators = smc_analyzer.analyze_all_smc(formatted_data)

        # Return combined results
        return {
            "symbol": symbol,
            "data": formatted_data,
            "technical_indicators": technical_indicators,
            "smc_indicators": smc_indicators,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/v1/trading-signal",
          response_model=TradingSignalResponse,
          summary="Get Trading Signal for GPT",
          description="Fetch cryptocurrency data and return GPT-optimized trading signals with key decision points",
          response_description="Concise trading signal optimized for automated decision making")
async def get_trading_signal(request: AnalyzeRequest, _: bool = Depends(verify_api_key)):
    """
    Get GPT-optimized trading signals for automated decision making.
    
    This endpoint:
    1. Fetches klines data from Binance for the specified symbol, interval, and limit
    2. Performs comprehensive technical and SMC analysis (same as /v1/analyze)
    3. Processes the data into concise, decision-focused signals
    4. Returns key indicators, trends, and overall trading recommendation
    
    **Optimized for GPT consumption - uses ~200-500 tokens vs ~2000+ tokens from /v1/analyze**
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., BTC, ETH, ADA, etc.)
    - **interval**: Optional kline interval (default: 1h) - Valid values: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    - **limit**: Optional number of klines to fetch (default: 90, max: 1000)
    
    **Returns:**
    - Current market data and price changes
    - Technical indicator signals (RSI, MACD, Bollinger Bands)
    - Smart Money Concepts summary (key levels, liquidity)
    - Overall trading signal with confidence score
    
    **Note:** If the symbol doesn't end with 'USDT', it will be automatically appended.
    """
    try:
        # Initialize analyzers (same as existing endpoint)
        binance_fetcher = BinanceDataFetcher()
        technical_analyzer = TechnicalAnalyzer()
        smc_analyzer = SMCAnalyzer()

        # Fetch data from Binance using request parameters
        symbol = request.symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        klines_data = await binance_fetcher.get_klines(
            symbol=symbol,
            interval=request.interval,
            limit=request.limit
        )
        formatted_data = binance_fetcher.format_klines_data(klines_data)

        # Run technical analysis
        technical_indicators = technical_analyzer.analyze_all_indicators(formatted_data)

        # Run SMC analysis
        smc_indicators = smc_analyzer.analyze_all_smc(formatted_data)

        # Process data for GPT optimization
        trading_signal = process_for_trading_decision(
            symbol=symbol,
            formatted_data=formatted_data,
            technical_indicators=technical_indicators,
            smc_indicators=smc_indicators,
            timeframe=request.interval
        )

        return trading_signal

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trading signal analysis failed: {str(e)}")
