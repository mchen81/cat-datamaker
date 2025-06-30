from datetime import datetime, timezone
from typing import Dict, Any, List

from models import *


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
