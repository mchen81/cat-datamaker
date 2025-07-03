from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field

from binance_data import BinanceDataFetcher
from utils import verify_api_key


class RSIStatus(str, Enum):
    OVERSOLD = "OVERSOLD"
    NEUTRAL = "NEUTRAL"
    OVERBOUGHT = "OVERBOUGHT"


class TrendDirection(str, Enum):
    RISING = "RISING"
    FALLING = "FALLING"
    SIDEWAYS = "SIDEWAYS"


class DivergenceType(str, Enum):
    BULLISH_DIVERGENCE = "BULLISH_DIVERGENCE"
    BEARISH_DIVERGENCE = "BEARISH_DIVERGENCE"
    HIDDEN_BULLISH = "HIDDEN_BULLISH"
    HIDDEN_BEARISH = "HIDDEN_BEARISH"


class DivergenceStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class VolumeProfile(str, Enum):
    VERY_HIGH = "VERY_HIGH"
    ABOVE_AVERAGE = "ABOVE_AVERAGE"
    AVERAGE = "AVERAGE"
    BELOW_AVERAGE = "BELOW_AVERAGE"
    VERY_LOW = "VERY_LOW"


class UnusualActivityType(str, Enum):
    VOLUME_SPIKE = "VOLUME_SPIKE"
    VOLUME_DRYUP = "VOLUME_DRYUP"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"


class VolumePattern(str, Enum):
    RISING_PRICE_RISING_VOLUME = "RISING_PRICE_RISING_VOLUME"
    RISING_PRICE_FALLING_VOLUME = "RISING_PRICE_FALLING_VOLUME"
    FALLING_PRICE_RISING_VOLUME = "FALLING_PRICE_RISING_VOLUME"
    FALLING_PRICE_FALLING_VOLUME = "FALLING_PRICE_FALLING_VOLUME"


class SmartMoneyDirection(str, Enum):
    INFLOW = "INFLOW"
    OUTFLOW = "OUTFLOW"
    NEUTRAL = "NEUTRAL"


class SmartMoneyStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class ATRStatus(str, Enum):
    EXTREME = "EXTREME"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class VolatilityRegime(str, Enum):
    EXPANSION = "EXPANSION"
    CONSOLIDATION = "CONSOLIDATION"
    BREAKOUT = "BREAKOUT"
    SQUEEZE = "SQUEEZE"


class MarketStrengthTrend(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    MODERATE_BULLISH = "MODERATE_BULLISH"
    WEAK_BULLISH = "WEAK_BULLISH"
    NEUTRAL = "NEUTRAL"
    WEAK_BEARISH = "WEAK_BEARISH"
    MODERATE_BEARISH = "MODERATE_BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class BuySellPressure(str, Enum):
    EXTREME = "EXTREME"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class MarketPhase(str, Enum):
    EARLY_TREND = "EARLY_TREND"
    MATURE_TREND = "MATURE_TREND"
    LATE_TREND = "LATE_TREND"
    REVERSAL = "REVERSAL"
    CONSOLIDATION = "CONSOLIDATION"


class SignalAlignment(str, Enum):
    STRONG_BULLISH_BIAS = "STRONG_BULLISH_BIAS"
    BULLISH_BIAS = "BULLISH_BIAS"
    NEUTRAL_BULLISH_BIAS = "NEUTRAL_BULLISH_BIAS"
    NEUTRAL = "NEUTRAL"
    NEUTRAL_BEARISH_BIAS = "NEUTRAL_BEARISH_BIAS"
    BEARISH_BIAS = "BEARISH_BIAS"
    STRONG_BEARISH_BIAS = "STRONG_BEARISH_BIAS"


class PrimarySignal(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL_BULLISH = "NEUTRAL_BULLISH"
    NEUTRAL = "NEUTRAL"
    NEUTRAL_BEARISH = "NEUTRAL_BEARISH"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class SignalStrength(str, Enum):
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    VERY_WEAK = "VERY_WEAK"


class RecommendedAction(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    LOOK_FOR_LONG_ENTRIES = "LOOK_FOR_LONG_ENTRIES"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"
    LOOK_FOR_SHORT_ENTRIES = "LOOK_FOR_SHORT_ENTRIES"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class MarketState(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    TRANSITIONING = "TRANSITIONING"
    VOLATILE = "VOLATILE"


class TrendMaturity(str, Enum):
    EARLY = "EARLY"
    MIDDLE = "MIDDLE"
    LATE = "LATE"
    EXHAUSTED = "EXHAUSTED"


class OptimalStrategy(str, Enum):
    TREND_FOLLOWING = "TREND_FOLLOWING"
    MEAN_REVERSION = "MEAN_REVERSION"
    BREAKOUT = "BREAKOUT"
    SCALPING = "SCALPING"
    WAIT_AND_SEE = "WAIT_AND_SEE"


class RiskEnvironment(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"


class MomentumStatus(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class MomentumAcceleration(str, Enum):
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STEADY = "STEADY"


class CrossSignal(str, Enum):
    BULLISH_CROSS = "BULLISH_CROSS"
    BEARISH_CROSS = "BEARISH_CROSS"
    NONE = "NONE"


class CurrentRange(str, Enum):
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


# Pydantic Models
class DivergenceData(BaseModel):
    detected: bool = Field(..., description="Whether divergence is detected")
    type: Optional[DivergenceType] = Field(None, description="Type of divergence")
    strength: Optional[DivergenceStrength] = Field(None, description="Strength of divergence")
    description: Optional[str] = Field(None, description="Description of the divergence")


class RSIKeyLevels(BaseModel):
    oversold: bool = Field(..., description="Whether RSI is in oversold territory")
    overbought: bool = Field(..., description="Whether RSI is in overbought territory")
    previous_extreme: Optional[str] = Field(None, description="Previous extreme level description")


class RSIData(BaseModel):
    status: RSIStatus = Field(..., description="Current RSI status")
    value_range: str = Field(..., description="Current RSI value range")
    trend: TrendDirection = Field(..., description="RSI trend direction")
    divergence: DivergenceData = Field(..., description="Divergence analysis")
    key_levels: RSIKeyLevels = Field(..., description="Key RSI levels")


class MomentumOscillator(BaseModel):
    status: MomentumStatus = Field(..., description="Momentum status")
    strength: SignalStrength = Field(..., description="Momentum strength")
    acceleration: MomentumAcceleration = Field(..., description="Momentum acceleration")
    cross_signal: CrossSignal = Field(..., description="Cross signal status")


class MomentumIndicators(BaseModel):
    rsi: RSIData = Field(..., description="RSI analysis")
    momentum_oscillator: MomentumOscillator = Field(..., description="Momentum oscillator analysis")


class UnusualActivity(BaseModel):
    detected: bool = Field(..., description="Whether unusual activity is detected")
    type: Optional[UnusualActivityType] = Field(None, description="Type of unusual activity")
    magnitude: Optional[str] = Field(None, description="Magnitude of unusual activity")
    interpretation: Optional[str] = Field(None, description="Interpretation of the activity")


class VolumePatterns(BaseModel):
    accumulation: bool = Field(..., description="Whether accumulation pattern is detected")
    distribution: bool = Field(..., description="Whether distribution pattern is detected")
    pattern: VolumePattern = Field(..., description="Current volume pattern")


class SmartMoneyFlow(BaseModel):
    direction: SmartMoneyDirection = Field(..., description="Smart money flow direction")
    strength: SmartMoneyStrength = Field(..., description="Smart money flow strength")
    persistence: str = Field(..., description="Persistence of the flow")


class VolumeAnalysis(BaseModel):
    current_volume_profile: VolumeProfile = Field(..., description="Current volume profile")
    volume_trend: TrendDirection = Field(..., description="Volume trend")
    volume_confirmation: bool = Field(..., description="Whether volume confirms price action")
    unusual_activity: UnusualActivity = Field(..., description="Unusual volume activity")
    volume_patterns: VolumePatterns = Field(..., description="Volume patterns")
    smart_money_flow: SmartMoneyFlow = Field(..., description="Smart money flow analysis")


class VolatilityMetrics(BaseModel):
    atr_status: ATRStatus = Field(..., description="ATR status")
    volatility_trend: TrendDirection = Field(..., description="Volatility trend")
    current_range: CurrentRange = Field(..., description="Current trading range")
    range_expansion: bool = Field(..., description="Whether range is expanding")
    suggested_stop_distance: str = Field(..., description="Suggested stop distance as percentage")
    volatility_regime: VolatilityRegime = Field(..., description="Current volatility regime")


class StrengthComponents(BaseModel):
    price_action: float = Field(..., description="Price action score", ge=0, le=10)
    volume: float = Field(..., description="Volume score", ge=0, le=10)
    momentum: float = Field(..., description="Momentum score", ge=0, le=10)
    volatility: float = Field(..., description="Volatility score", ge=0, le=10)


class MarketStrength(BaseModel):
    overall_score: float = Field(..., description="Overall market strength score", ge=0, le=10)
    trend_strength: MarketStrengthTrend = Field(..., description="Trend strength assessment")
    buy_pressure: BuySellPressure = Field(..., description="Buy pressure level")
    sell_pressure: BuySellPressure = Field(..., description="Sell pressure level")
    strength_components: StrengthComponents = Field(..., description="Breakdown of strength components")
    market_phase: MarketPhase = Field(..., description="Current market phase")


class Confluences(BaseModel):
    bullish_signals: List[str] = Field(..., description="List of bullish signals")
    bearish_signals: List[str] = Field(..., description="List of bearish signals")
    neutral_factors: List[str] = Field(..., description="List of neutral factors")
    signal_alignment: SignalAlignment = Field(..., description="Overall signal alignment")
    confidence_score: float = Field(..., description="Confidence score", ge=0, le=10)


class IndicatorSummary(BaseModel):
    primary_signal: PrimarySignal = Field(..., description="Primary signal direction")
    signal_strength: SignalStrength = Field(..., description="Signal strength")
    key_observation: str = Field(..., description="Key observation from analysis")
    caution_notes: str = Field(..., description="Caution notes and warnings")
    recommended_action: RecommendedAction = Field(..., description="Recommended trading action")
    invalidation_scenario: str = Field(..., description="Scenario that would invalidate the signal")


class TradingConditions(BaseModel):
    market_state: MarketState = Field(..., description="Current market state")
    trend_maturity: TrendMaturity = Field(..., description="Trend maturity level")
    optimal_strategy: OptimalStrategy = Field(..., description="Optimal trading strategy")
    risk_environment: RiskEnvironment = Field(..., description="Risk environment assessment")
    session_alignment: bool = Field(..., description="Whether session is aligned for trading")


class TechnicalIndicatorsResponse(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Analysis timeframe")
    timestamp: str = Field(..., description="Analysis timestamp")
    current_price: float = Field(..., description="Current price")
    momentum_indicators: MomentumIndicators = Field(..., description="Momentum indicators analysis")
    volume_analysis: VolumeAnalysis = Field(..., description="Volume analysis")
    volatility_metrics: VolatilityMetrics = Field(..., description="Volatility metrics")
    market_strength: MarketStrength = Field(..., description="Market strength analysis")
    confluences: Confluences = Field(..., description="Signal confluences")
    indicator_summary: IndicatorSummary = Field(..., description="Indicator summary")
    trading_conditions: TradingConditions = Field(..., description="Trading conditions")


# Create router
router = APIRouter(tags=["Technical Analysis"])


class TechnicalIndicatorAnalyzer:
    """
    Technical Indicator Analysis using pandas_ta library
    Does not extend SMC analyzer as it focuses on traditional indicators
    """

    def __init__(self):
        pass

    def prepare_dataframe(self, ohlcv_data: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Convert OHLCV data to pandas DataFrame for analysis
        """
        df = pd.DataFrame(ohlcv_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        return df

    def calculate_distance_percentage(self, level: float, current_price: float) -> str:
        """
        Calculate percentage distance from current price
        """
        if current_price == 0:
            return "0.00%"

        percentage = ((level - current_price) / current_price) * 100
        sign = "+" if percentage >= 0 else ""
        return f"{sign}{percentage:.2f}%"

    def analyze_rsi(self, df: pd.DataFrame) -> RSIData:
        """
        Analyze RSI with divergence detection
        """
        try:
            # Calculate RSI using pandas_ta
            rsi = ta.rsi(df['close'], length=14)

            # Get current RSI value
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50

            # Determine RSI status
            if current_rsi >= 70:
                status = RSIStatus.OVERBOUGHT
            elif current_rsi <= 30:
                status = RSIStatus.OVERSOLD
            else:
                status = RSIStatus.NEUTRAL

            # Determine value range
            if current_rsi >= 80:
                value_range = "80-100"
            elif current_rsi >= 70:
                value_range = "70-80"
            elif current_rsi >= 50:
                value_range = "50-70"
            elif current_rsi >= 30:
                value_range = "30-50"
            elif current_rsi >= 20:
                value_range = "20-30"
            else:
                value_range = "0-20"

            # Determine RSI trend
            if len(rsi) >= 5:
                recent_rsi = rsi.iloc[-5:].tolist()
                if recent_rsi[-1] > recent_rsi[0]:
                    trend = TrendDirection.RISING
                elif recent_rsi[-1] < recent_rsi[0]:
                    trend = TrendDirection.FALLING
                else:
                    trend = TrendDirection.SIDEWAYS
            else:
                trend = TrendDirection.SIDEWAYS

            # Detect divergences (simplified)
            divergence = self.detect_rsi_divergence(df, rsi)

            # Key levels
            key_levels = RSIKeyLevels(
                oversold=current_rsi <= 30,
                overbought=current_rsi >= 70,
                previous_extreme=self.get_previous_rsi_extreme(rsi)
            )

            return RSIData(
                status=status,
                value_range=value_range,
                trend=trend,
                divergence=divergence,
                key_levels=key_levels
            )

        except Exception as e:
            print(f"Error analyzing RSI: {e}")
            return RSIData(
                status=RSIStatus.NEUTRAL,
                value_range="40-60",
                trend=TrendDirection.SIDEWAYS,
                divergence=DivergenceData(detected=False),
                key_levels=RSIKeyLevels(oversold=False, overbought=False)
            )

    def detect_rsi_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> DivergenceData:
        """
        Detect RSI divergences with price
        """
        try:
            if len(df) < 20 or len(rsi) < 20:
                return DivergenceData(detected=False)

            # Get recent price and RSI data
            recent_prices = df['close'].iloc[-20:].tolist()
            recent_rsi = rsi.iloc[-20:].tolist()

            # Simple divergence detection (last 10 vs previous 10)
            if len(recent_prices) >= 20:
                price_change = recent_prices[-1] - recent_prices[-10]
                rsi_change = recent_rsi[-1] - recent_rsi[-10]

                # Bullish divergence: price lower, RSI higher
                if price_change < 0 and rsi_change > 0:
                    strength = DivergenceStrength.MODERATE if abs(price_change) > df['close'].iloc[
                        -1] * 0.02 else DivergenceStrength.WEAK
                    return DivergenceData(
                        detected=True,
                        type=DivergenceType.BULLISH_DIVERGENCE,
                        strength=strength,
                        description="Price made lower low, RSI made higher low"
                    )

                # Bearish divergence: price higher, RSI lower
                elif price_change > 0 and rsi_change < 0:
                    strength = DivergenceStrength.MODERATE if abs(price_change) > df['close'].iloc[
                        -1] * 0.02 else DivergenceStrength.WEAK
                    return DivergenceData(
                        detected=True,
                        type=DivergenceType.BEARISH_DIVERGENCE,
                        strength=strength,
                        description="Price made higher high, RSI made lower high"
                    )

            return DivergenceData(detected=False)

        except Exception:
            return DivergenceData(detected=False)

    def get_previous_rsi_extreme(self, rsi: pd.Series) -> Optional[str]:
        """
        Get description of previous RSI extreme
        """
        try:
            if len(rsi) < 10:
                return None

            recent_rsi = rsi.iloc[-10:].tolist()

            for i, value in enumerate(reversed(recent_rsi[:-1]), 1):
                if value <= 30:
                    return f"OVERSOLD_{i}_BARS_AGO"
                elif value >= 70:
                    return f"OVERBOUGHT_{i}_BARS_AGO"

            return None

        except Exception:
            return None

    def analyze_momentum_oscillator(self, df: pd.DataFrame) -> MomentumOscillator:
        """
        Analyze momentum oscillator (using MACD)
        """
        try:
            # Calculate MACD using pandas_ta
            macd_data = ta.macd(df['close'])

            if macd_data is None or macd_data.empty:
                return MomentumOscillator(
                    status=MomentumStatus.NEUTRAL,
                    strength=SignalStrength.WEAK,
                    acceleration=MomentumAcceleration.STEADY,
                    cross_signal=CrossSignal.NONE
                )

            # Get MACD components
            macd_line = macd_data.iloc[:, 0]  # MACD line
            signal_line = macd_data.iloc[:, 1]  # Signal line
            histogram = macd_data.iloc[:, 2]  # Histogram

            # Current values
            current_macd = macd_line.iloc[-1] if not macd_line.empty else 0
            current_signal = signal_line.iloc[-1] if not signal_line.empty else 0
            current_histogram = histogram.iloc[-1] if not histogram.empty else 0

            # Determine status
            if current_macd > current_signal and current_macd > 0:
                status = MomentumStatus.BULLISH
            elif current_macd < current_signal and current_macd < 0:
                status = MomentumStatus.BEARISH
            else:
                status = MomentumStatus.NEUTRAL

            # Determine strength
            macd_strength = abs(current_macd - current_signal)
            if macd_strength > 0.005:  # Adjust threshold as needed
                strength = SignalStrength.STRONG
            elif macd_strength > 0.002:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK

            # Determine acceleration
            if len(histogram) >= 3:
                recent_hist = histogram.iloc[-3:].tolist()
                if recent_hist[-1] > recent_hist[-2] > recent_hist[-3]:
                    acceleration = MomentumAcceleration.INCREASING
                elif recent_hist[-1] < recent_hist[-2] < recent_hist[-3]:
                    acceleration = MomentumAcceleration.DECREASING
                else:
                    acceleration = MomentumAcceleration.STEADY
            else:
                acceleration = MomentumAcceleration.STEADY

            # Detect cross signals
            cross_signal = CrossSignal.NONE
            if len(macd_line) >= 2 and len(signal_line) >= 2:
                prev_macd = macd_line.iloc[-2]
                prev_signal = signal_line.iloc[-2]

                # Bullish cross: MACD crosses above signal
                if current_macd > current_signal and prev_macd <= prev_signal:
                    cross_signal = CrossSignal.BULLISH_CROSS
                # Bearish cross: MACD crosses below signal
                elif current_macd < current_signal and prev_macd >= prev_signal:
                    cross_signal = CrossSignal.BEARISH_CROSS

            return MomentumOscillator(
                status=status,
                strength=strength,
                acceleration=acceleration,
                cross_signal=cross_signal
            )

        except Exception as e:
            print(f"Error analyzing momentum oscillator: {e}")
            return MomentumOscillator(
                status=MomentumStatus.NEUTRAL,
                strength=SignalStrength.WEAK,
                acceleration=MomentumAcceleration.STEADY,
                cross_signal=CrossSignal.NONE
            )

    def analyze_volume(self, df: pd.DataFrame) -> VolumeAnalysis:
        """
        Analyze volume patterns and smart money flow
        """
        try:
            volume = df['volume'].copy()
            close = df['close'].copy()

            # Calculate volume profile
            volume_profile = self.determine_volume_profile(volume)

            # Determine volume trend
            volume_trend = self.determine_volume_trend(volume)

            # Check volume confirmation
            volume_confirmation = self.check_volume_confirmation(close, volume)

            # Detect unusual activity
            unusual_activity = self.detect_unusual_volume_activity(volume)

            # Analyze volume patterns
            volume_patterns = self.analyze_volume_patterns(close, volume)

            # Analyze smart money flow
            smart_money_flow = self.analyze_smart_money_flow(df)

            return VolumeAnalysis(
                current_volume_profile=volume_profile,
                volume_trend=volume_trend,
                volume_confirmation=volume_confirmation,
                unusual_activity=unusual_activity,
                volume_patterns=volume_patterns,
                smart_money_flow=smart_money_flow
            )

        except Exception as e:
            print(f"Error analyzing volume: {e}")
            return VolumeAnalysis(
                current_volume_profile=VolumeProfile.AVERAGE,
                volume_trend=TrendDirection.SIDEWAYS,
                volume_confirmation=False,
                unusual_activity=UnusualActivity(detected=False),
                volume_patterns=VolumePatterns(
                    accumulation=False,
                    distribution=False,
                    pattern=VolumePattern.RISING_PRICE_RISING_VOLUME
                ),
                smart_money_flow=SmartMoneyFlow(
                    direction=SmartMoneyDirection.NEUTRAL,
                    strength=SmartMoneyStrength.WEAK,
                    persistence="0_BARS"
                )
            )

    def determine_volume_profile(self, volume: pd.Series) -> VolumeProfile:
        """
        Determine current volume profile relative to average
        """
        try:
            if len(volume) < 20:
                return VolumeProfile.AVERAGE

            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]

            ratio = current_volume / avg_volume if avg_volume > 0 else 1

            if ratio >= 3.0:
                return VolumeProfile.VERY_HIGH
            elif ratio >= 1.5:
                return VolumeProfile.ABOVE_AVERAGE
            elif ratio >= 0.7:
                return VolumeProfile.AVERAGE
            elif ratio >= 0.3:
                return VolumeProfile.BELOW_AVERAGE
            else:
                return VolumeProfile.VERY_LOW

        except Exception:
            return VolumeProfile.AVERAGE

    def determine_volume_trend(self, volume: pd.Series) -> TrendDirection:
        """
        Determine volume trend direction
        """
        try:
            if len(volume) < 10:
                return TrendDirection.SIDEWAYS

            recent_volume = volume.iloc[-10:]

            # Use linear regression to determine trend
            x = np.arange(len(recent_volume))
            y = recent_volume.values

            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]

            # Normalize slope by average volume
            avg_volume = recent_volume.mean()
            normalized_slope = slope / avg_volume if avg_volume > 0 else 0

            if normalized_slope > 0.05:
                return TrendDirection.RISING
            elif normalized_slope < -0.05:
                return TrendDirection.FALLING
            else:
                return TrendDirection.SIDEWAYS

        except Exception:
            return TrendDirection.SIDEWAYS

    def check_volume_confirmation(self, close: pd.Series, volume: pd.Series) -> bool:
        """
        Check if volume confirms price action
        """
        try:
            if len(close) < 5 or len(volume) < 5:
                return False

            # Check last 5 periods
            recent_price_change = close.iloc[-1] - close.iloc[-5]
            recent_volume_avg = volume.iloc[-5:].mean()
            overall_volume_avg = volume.iloc[-20:].mean() if len(volume) >= 20 else volume.mean()

            # Volume confirmation if:
            # 1. Strong price move (>1%) with above-average volume
            price_move_pct = abs(recent_price_change) / close.iloc[-5] if close.iloc[-5] > 0 else 0
            volume_above_avg = recent_volume_avg > overall_volume_avg * 1.2

            return price_move_pct > 0.01 and volume_above_avg

        except Exception:
            return False

    def detect_unusual_volume_activity(self, volume: pd.Series) -> UnusualActivity:
        """
        Detect unusual volume activity
        """
        try:
            if len(volume) < 20:
                return UnusualActivity(detected=False)

            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]

            ratio = current_volume / avg_volume if avg_volume > 0 else 1

            if ratio >= 2.5:
                magnitude = f"{ratio:.1f}X_AVERAGE"
                return UnusualActivity(
                    detected=True,
                    type=UnusualActivityType.VOLUME_SPIKE,
                    magnitude=magnitude,
                    interpretation="Strong buying/selling interest"
                )
            elif ratio <= 0.3:
                magnitude = f"{ratio:.1f}X_AVERAGE"
                return UnusualActivity(
                    detected=True,
                    type=UnusualActivityType.VOLUME_DRYUP,
                    magnitude=magnitude,
                    interpretation="Low market participation"
                )
            else:
                return UnusualActivity(detected=False)

        except Exception:
            return UnusualActivity(detected=False)

    def analyze_volume_patterns(self, close: pd.Series, volume: pd.Series) -> VolumePatterns:
        """
        Analyze volume patterns for accumulation/distribution
        """
        try:
            if len(close) < 10 or len(volume) < 10:
                return VolumePatterns(
                    accumulation=False,
                    distribution=False,
                    pattern=VolumePattern.RISING_PRICE_RISING_VOLUME
                )

            # Check last 10 periods
            price_change = close.iloc[-1] - close.iloc[-10]
            volume_change = volume.iloc[-5:].mean() - volume.iloc[-10:-5].mean()

            # Determine pattern
            price_rising = price_change > 0
            volume_rising = volume_change > 0

            if price_rising and volume_rising:
                pattern = VolumePattern.RISING_PRICE_RISING_VOLUME
                accumulation = True
                distribution = False
            elif price_rising and not volume_rising:
                pattern = VolumePattern.RISING_PRICE_FALLING_VOLUME
                accumulation = False
                distribution = True
            elif not price_rising and volume_rising:
                pattern = VolumePattern.FALLING_PRICE_RISING_VOLUME
                accumulation = True
                distribution = False
            else:
                pattern = VolumePattern.FALLING_PRICE_FALLING_VOLUME
                accumulation = False
                distribution = True

            return VolumePatterns(
                accumulation=accumulation,
                distribution=distribution,
                pattern=pattern
            )

        except Exception:
            return VolumePatterns(
                accumulation=False,
                distribution=False,
                pattern=VolumePattern.RISING_PRICE_RISING_VOLUME
            )

    def analyze_smart_money_flow(self, df: pd.DataFrame) -> SmartMoneyFlow:
        """
        Analyze smart money flow using volume and price action
        """
        try:
            if len(df) < 10:
                return SmartMoneyFlow(
                    direction=SmartMoneyDirection.NEUTRAL,
                    strength=SmartMoneyStrength.WEAK,
                    persistence="0_BARS"
                )

            # Calculate Money Flow Index using pandas_ta
            mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)

            if mfi is None or mfi.empty:
                return SmartMoneyFlow(
                    direction=SmartMoneyDirection.NEUTRAL,
                    strength=SmartMoneyStrength.WEAK,
                    persistence="0_BARS"
                )

            current_mfi = mfi.iloc[-1]

            # Determine direction
            if current_mfi > 60:
                direction = SmartMoneyDirection.INFLOW
            elif current_mfi < 40:
                direction = SmartMoneyDirection.OUTFLOW
            else:
                direction = SmartMoneyDirection.NEUTRAL

            # Determine strength
            if current_mfi > 80 or current_mfi < 20:
                strength = SmartMoneyStrength.STRONG
            elif current_mfi > 70 or current_mfi < 30:
                strength = SmartMoneyStrength.MODERATE
            else:
                strength = SmartMoneyStrength.WEAK

            # Calculate persistence
            persistence_count = 0
            if len(mfi) >= 5:
                recent_mfi = mfi.iloc[-5:].tolist()
                if direction == SmartMoneyDirection.INFLOW:
                    persistence_count = sum(1 for x in recent_mfi if x > 50)
                elif direction == SmartMoneyDirection.OUTFLOW:
                    persistence_count = sum(1 for x in recent_mfi if x < 50)

            persistence = f"{persistence_count}_CONSECUTIVE_BARS"

            return SmartMoneyFlow(
                direction=direction,
                strength=strength,
                persistence=persistence
            )

        except Exception:
            return SmartMoneyFlow(
                direction=SmartMoneyDirection.NEUTRAL,
                strength=SmartMoneyStrength.WEAK,
                persistence="0_BARS"
            )

    def analyze_volatility(self, df: pd.DataFrame) -> VolatilityMetrics:
        """
        Analyze volatility metrics using ATR
        """
        try:
            # Calculate ATR using pandas_ta
            atr = ta.atr(df['high'], df['low'], df['close'], length=14)

            if atr is None or atr.empty:
                return VolatilityMetrics(
                    atr_status=ATRStatus.NORMAL,
                    volatility_trend=TrendDirection.SIDEWAYS,
                    current_range=CurrentRange.MEDIUM,
                    range_expansion=False,
                    suggested_stop_distance="2.0%",
                    volatility_regime=VolatilityRegime.CONSOLIDATION
                )

            current_atr = atr.iloc[-1]
            current_price = df['close'].iloc[-1]

            # Determine ATR status relative to its historical values
            atr_status = self.determine_atr_status(atr)

            # Determine volatility trend
            volatility_trend = self.determine_volatility_trend(atr)

            # Determine current range
            current_range = self.determine_current_range(atr, current_atr)

            # Check for range expansion
            range_expansion = self.check_range_expansion(atr)

            # Calculate suggested stop distance
            stop_distance_pct = (current_atr / current_price) * 100 * 1.5  # 1.5x ATR
            suggested_stop_distance = f"{stop_distance_pct:.1f}%"

            # Determine volatility regime
            volatility_regime = self.determine_volatility_regime(atr, df)

            return VolatilityMetrics(
                atr_status=atr_status,
                volatility_trend=volatility_trend,
                current_range=current_range,
                range_expansion=range_expansion,
                suggested_stop_distance=suggested_stop_distance,
                volatility_regime=volatility_regime
            )

        except Exception as e:
            print(f"Error analyzing volatility: {e}")
            return VolatilityMetrics(
                atr_status=ATRStatus.NORMAL,
                volatility_trend=TrendDirection.SIDEWAYS,
                current_range=CurrentRange.MEDIUM,
                range_expansion=False,
                suggested_stop_distance="2.0%",
                volatility_regime=VolatilityRegime.CONSOLIDATION
            )

    def determine_atr_status(self, atr: pd.Series) -> ATRStatus:
        """
        Determine ATR status relative to historical values
        """
        try:
            if len(atr) < 50:
                return ATRStatus.NORMAL

            current_atr = atr.iloc[-1]
            atr_percentile = (atr <= current_atr).sum() / len(atr)

            if atr_percentile >= 0.9:
                return ATRStatus.EXTREME
            elif atr_percentile >= 0.7:
                return ATRStatus.HIGH
            elif atr_percentile >= 0.3:
                return ATRStatus.NORMAL
            else:
                return ATRStatus.LOW

        except Exception:
            return ATRStatus.NORMAL

    def determine_volatility_trend(self, atr: pd.Series) -> TrendDirection:
        """
        Determine volatility trend direction
        """
        try:
            if len(atr) < 10:
                return TrendDirection.SIDEWAYS

            recent_atr = atr.iloc[-10:]

            # Use linear regression to determine trend
            x = np.arange(len(recent_atr))
            y = recent_atr.values

            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]

            # Normalize slope by average ATR
            avg_atr = recent_atr.mean()
            normalized_slope = slope / avg_atr if avg_atr > 0 else 0

            if normalized_slope > 0.05:
                return TrendDirection.RISING
            elif normalized_slope < -0.05:
                return TrendDirection.FALLING
            else:
                return TrendDirection.SIDEWAYS

        except Exception:
            return TrendDirection.SIDEWAYS

    def determine_current_range(self, atr: pd.Series, current_atr: float) -> CurrentRange:
        """
        Determine current trading range based on ATR
        """
        try:
            if len(atr) < 20:
                return CurrentRange.MEDIUM

            atr_avg = atr.rolling(20).mean().iloc[-1]
            ratio = current_atr / atr_avg if atr_avg > 0 else 1

            if ratio >= 2.0:
                return CurrentRange.VERY_HIGH
            elif ratio >= 1.5:
                return CurrentRange.HIGH
            elif ratio >= 0.8:
                return CurrentRange.MEDIUM
            elif ratio >= 0.5:
                return CurrentRange.LOW
            else:
                return CurrentRange.VERY_LOW

        except Exception:
            return CurrentRange.MEDIUM

    def check_range_expansion(self, atr: pd.Series) -> bool:
        """
        Check if range is expanding (volatility increasing)
        """
        try:
            if len(atr) < 5:
                return False

            recent_atr = atr.iloc[-5:].mean()
            previous_atr = atr.iloc[-10:-5].mean() if len(atr) >= 10 else atr.mean()

            return recent_atr > previous_atr * 1.2

        except Exception:
            return False

    def determine_volatility_regime(self, atr: pd.Series, df: pd.DataFrame) -> VolatilityRegime:
        """
        Determine current volatility regime
        """
        try:
            if len(atr) < 20 or len(df) < 20:
                return VolatilityRegime.CONSOLIDATION

            # Calculate Bollinger Bands to help determine regime
            bb = ta.bbands(df['close'], length=20)

            if bb is not None and not bb.empty:
                bb_width = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]  # (Upper - Lower) / Middle
                current_bb_width = bb_width.iloc[-1]
                avg_bb_width = bb_width.rolling(50).mean().iloc[-1] if len(bb_width) >= 50 else bb_width.mean()

                # Check for squeeze (low volatility)
                if current_bb_width < avg_bb_width * 0.7:
                    return VolatilityRegime.SQUEEZE

                # Check for expansion (high volatility)
                elif current_bb_width > avg_bb_width * 1.3:
                    return VolatilityRegime.EXPANSION

                # Check for breakout (recent expansion)
                recent_bb_width = bb_width.iloc[-5:].mean()
                previous_bb_width = bb_width.iloc[-15:-5].mean() if len(bb_width) >= 15 else bb_width.mean()

                if recent_bb_width > previous_bb_width * 1.5:
                    return VolatilityRegime.BREAKOUT

            return VolatilityRegime.CONSOLIDATION

        except Exception:
            return VolatilityRegime.CONSOLIDATION

    def calculate_market_strength(self, df: pd.DataFrame, rsi_data: RSIData,
                                  volume_analysis: VolumeAnalysis,
                                  volatility_metrics: VolatilityMetrics) -> MarketStrength:
        """
        Calculate comprehensive market strength
        """
        try:
            # Calculate individual component scores
            price_action_score = self.calculate_price_action_score(df)
            volume_score = self.calculate_volume_score(volume_analysis)
            momentum_score = self.calculate_momentum_score(rsi_data)
            volatility_score = self.calculate_volatility_score(volatility_metrics)

            # Calculate overall score (weighted average)
            overall_score = (
                    price_action_score * 0.3 +
                    volume_score * 0.25 +
                    momentum_score * 0.25 +
                    volatility_score * 0.2
            )

            # Determine trend strength
            trend_strength = self.determine_trend_strength(overall_score)

            # Determine buy/sell pressure
            buy_pressure = self.determine_buy_pressure(volume_analysis, rsi_data)
            sell_pressure = self.determine_sell_pressure(volume_analysis, rsi_data)

            # Determine market phase
            market_phase = self.determine_market_phase(overall_score, volatility_metrics)

            return MarketStrength(
                overall_score=overall_score,
                trend_strength=trend_strength,
                buy_pressure=buy_pressure,
                sell_pressure=sell_pressure,
                strength_components=StrengthComponents(
                    price_action=price_action_score,
                    volume=volume_score,
                    momentum=momentum_score,
                    volatility=volatility_score
                ),
                market_phase=market_phase
            )

        except Exception as e:
            print(f"Error calculating market strength: {e}")
            return MarketStrength(
                overall_score=5.0,
                trend_strength=MarketStrengthTrend.NEUTRAL,
                buy_pressure=BuySellPressure.MODERATE,
                sell_pressure=BuySellPressure.MODERATE,
                strength_components=StrengthComponents(
                    price_action=5.0,
                    volume=5.0,
                    momentum=5.0,
                    volatility=5.0
                ),
                market_phase=MarketPhase.CONSOLIDATION
            )

    def calculate_price_action_score(self, df: pd.DataFrame) -> float:
        """
        Calculate price action score based on recent price movements
        """
        try:
            if len(df) < 20:
                return 5.0

            # Calculate recent price change
            recent_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]

            # Calculate trend consistency
            closes = df['close'].iloc[-10:].tolist()
            upward_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
            trend_consistency = upward_moves / (len(closes) - 1)

            # Base score on price change magnitude and trend consistency
            change_score = min(10, max(0, (abs(recent_change) * 100) + 3))  # 0-10 scale
            consistency_score = trend_consistency * 10  # 0-10 scale

            # Combine scores
            score = (change_score * 0.6 + consistency_score * 0.4)

            # Adjust for direction
            if recent_change > 0:
                return min(10, score)
            else:
                return max(0, 10 - score)

        except Exception:
            return 5.0

    def calculate_volume_score(self, volume_analysis: VolumeAnalysis) -> float:
        """
        Calculate volume score
        """
        try:
            score = 5.0  # Base score

            # Volume profile scoring
            if volume_analysis.current_volume_profile == VolumeProfile.VERY_HIGH:
                score += 2
            elif volume_analysis.current_volume_profile == VolumeProfile.ABOVE_AVERAGE:
                score += 1
            elif volume_analysis.current_volume_profile == VolumeProfile.BELOW_AVERAGE:
                score -= 1
            elif volume_analysis.current_volume_profile == VolumeProfile.VERY_LOW:
                score -= 2

            # Volume confirmation
            if volume_analysis.volume_confirmation:
                score += 1

            # Smart money flow
            if volume_analysis.smart_money_flow.direction == SmartMoneyDirection.INFLOW:
                if volume_analysis.smart_money_flow.strength == SmartMoneyStrength.STRONG:
                    score += 2
                else:
                    score += 1
            elif volume_analysis.smart_money_flow.direction == SmartMoneyDirection.OUTFLOW:
                if volume_analysis.smart_money_flow.strength == SmartMoneyStrength.STRONG:
                    score -= 2
                else:
                    score -= 1

            return min(10, max(0, score))

        except Exception:
            return 5.0

    def calculate_momentum_score(self, rsi_data: RSIData) -> float:
        """
        Calculate momentum score based on RSI
        """
        try:
            score = 5.0  # Base score

            # RSI status scoring
            if rsi_data.status == RSIStatus.OVERBOUGHT:
                score += 1  # Positive momentum but potentially overextended
            elif rsi_data.status == RSIStatus.OVERSOLD:
                score -= 1  # Negative momentum but potentially oversold

            # RSI trend scoring
            if rsi_data.trend == TrendDirection.RISING:
                score += 2
            elif rsi_data.trend == TrendDirection.FALLING:
                score -= 2

            # Divergence scoring
            if rsi_data.divergence.detected:
                if rsi_data.divergence.type == DivergenceType.BULLISH_DIVERGENCE:
                    score += 1.5
                elif rsi_data.divergence.type == DivergenceType.BEARISH_DIVERGENCE:
                    score -= 1.5

            return min(10, max(0, score))

        except Exception:
            return 5.0

    def calculate_volatility_score(self, volatility_metrics: VolatilityMetrics) -> float:
        """
        Calculate volatility score
        """
        try:
            score = 5.0  # Base score

            # ATR status scoring (moderate volatility is preferred)
            if volatility_metrics.atr_status == ATRStatus.NORMAL:
                score += 2
            elif volatility_metrics.atr_status == ATRStatus.HIGH:
                score += 1
            elif volatility_metrics.atr_status == ATRStatus.LOW:
                score -= 1
            elif volatility_metrics.atr_status == ATRStatus.EXTREME:
                score -= 2

            # Volatility regime scoring
            if volatility_metrics.volatility_regime == VolatilityRegime.EXPANSION:
                score += 1
            elif volatility_metrics.volatility_regime == VolatilityRegime.BREAKOUT:
                score += 2
            elif volatility_metrics.volatility_regime == VolatilityRegime.SQUEEZE:
                score -= 1

            # Range expansion
            if volatility_metrics.range_expansion:
                score += 1

            return min(10, max(0, score))

        except Exception:
            return 5.0

    def determine_trend_strength(self, overall_score: float) -> MarketStrengthTrend:
        """
        Determine trend strength based on overall score
        """
        if overall_score >= 8.5:
            return MarketStrengthTrend.STRONG_BULLISH
        elif overall_score >= 7:
            return MarketStrengthTrend.MODERATE_BULLISH
        elif overall_score >= 5.5:
            return MarketStrengthTrend.WEAK_BULLISH
        elif overall_score >= 4.5:
            return MarketStrengthTrend.NEUTRAL
        elif overall_score >= 3:
            return MarketStrengthTrend.WEAK_BEARISH
        elif overall_score >= 1.5:
            return MarketStrengthTrend.MODERATE_BEARISH
        else:
            return MarketStrengthTrend.STRONG_BEARISH

    def determine_buy_pressure(self, volume_analysis: VolumeAnalysis, rsi_data: RSIData) -> BuySellPressure:
        """
        Determine buy pressure level
        """
        score = 0

        # Volume factors
        if volume_analysis.smart_money_flow.direction == SmartMoneyDirection.INFLOW:
            score += 2 if volume_analysis.smart_money_flow.strength == SmartMoneyStrength.STRONG else 1

        if volume_analysis.volume_patterns.accumulation:
            score += 1

        # RSI factors
        if rsi_data.trend == TrendDirection.RISING:
            score += 1

        if rsi_data.divergence.detected and rsi_data.divergence.type == DivergenceType.BULLISH_DIVERGENCE:
            score += 1

        if score >= 4:
            return BuySellPressure.EXTREME
        elif score >= 3:
            return BuySellPressure.HIGH
        elif score >= 1:
            return BuySellPressure.MODERATE
        elif score == 0:
            return BuySellPressure.LOW
        else:
            return BuySellPressure.VERY_LOW

    def determine_sell_pressure(self, volume_analysis: VolumeAnalysis, rsi_data: RSIData) -> BuySellPressure:
        """
        Determine sell pressure level
        """
        score = 0

        # Volume factors
        if volume_analysis.smart_money_flow.direction == SmartMoneyDirection.OUTFLOW:
            score += 2 if volume_analysis.smart_money_flow.strength == SmartMoneyStrength.STRONG else 1

        if volume_analysis.volume_patterns.distribution:
            score += 1

        # RSI factors
        if rsi_data.trend == TrendDirection.FALLING:
            score += 1

        if rsi_data.divergence.detected and rsi_data.divergence.type == DivergenceType.BEARISH_DIVERGENCE:
            score += 1

        if score >= 4:
            return BuySellPressure.EXTREME
        elif score >= 3:
            return BuySellPressure.HIGH
        elif score >= 1:
            return BuySellPressure.MODERATE
        elif score == 0:
            return BuySellPressure.LOW
        else:
            return BuySellPressure.VERY_LOW

    def determine_market_phase(self, overall_score: float, volatility_metrics: VolatilityMetrics) -> MarketPhase:
        """
        Determine current market phase
        """
        if volatility_metrics.volatility_regime == VolatilityRegime.BREAKOUT:
            return MarketPhase.EARLY_TREND
        elif volatility_metrics.volatility_regime == VolatilityRegime.EXPANSION:
            if overall_score > 7:
                return MarketPhase.EARLY_TREND
            else:
                return MarketPhase.MATURE_TREND
        elif volatility_metrics.volatility_regime == VolatilityRegime.SQUEEZE:
            return MarketPhase.CONSOLIDATION
        else:
            if overall_score > 8:
                return MarketPhase.LATE_TREND
            elif overall_score < 3:
                return MarketPhase.REVERSAL
            else:
                return MarketPhase.CONSOLIDATION

    def analyze_confluences(self, rsi_data: RSIData, volume_analysis: VolumeAnalysis,
                            volatility_metrics: VolatilityMetrics, market_strength: MarketStrength) -> Confluences:
        """
        Analyze signal confluences and alignment
        """
        try:
            bullish_signals = []
            bearish_signals = []
            neutral_factors = []

            # RSI signals
            if rsi_data.divergence.detected:
                if rsi_data.divergence.type == DivergenceType.BULLISH_DIVERGENCE:
                    bullish_signals.append("RSI bullish divergence")
                elif rsi_data.divergence.type == DivergenceType.BEARISH_DIVERGENCE:
                    bearish_signals.append("RSI bearish divergence")

            if rsi_data.trend == TrendDirection.RISING:
                bullish_signals.append("RSI trending upward")
            elif rsi_data.trend == TrendDirection.FALLING:
                bearish_signals.append("RSI trending downward")
            else:
                neutral_factors.append("RSI in sideways trend")

            if rsi_data.status == RSIStatus.OVERSOLD:
                bullish_signals.append("RSI oversold, potential bounce")
            elif rsi_data.status == RSIStatus.OVERBOUGHT:
                bearish_signals.append("RSI overbought, potential reversal")
            else:
                neutral_factors.append("RSI in neutral zone")

            # Volume signals
            if volume_analysis.unusual_activity.detected:
                if volume_analysis.unusual_activity.type == UnusualActivityType.VOLUME_SPIKE:
                    if volume_analysis.smart_money_flow.direction == SmartMoneyDirection.INFLOW:
                        bullish_signals.append("Volume spike with smart money inflow")
                    else:
                        neutral_factors.append("Volume spike detected")

            if volume_analysis.smart_money_flow.direction == SmartMoneyDirection.INFLOW:
                bullish_signals.append("Smart money flowing in")
            elif volume_analysis.smart_money_flow.direction == SmartMoneyDirection.OUTFLOW:
                bearish_signals.append("Smart money flowing out")

            if volume_analysis.volume_patterns.accumulation:
                bullish_signals.append("Accumulation pattern detected")
            elif volume_analysis.volume_patterns.distribution:
                bearish_signals.append("Distribution pattern detected")

            if volume_analysis.volume_confirmation:
                bullish_signals.append("Volume confirms price action")

            # Volatility signals
            if volatility_metrics.volatility_regime == VolatilityRegime.BREAKOUT:
                bullish_signals.append("Volatility breakout pattern")
            elif volatility_metrics.volatility_regime == VolatilityRegime.SQUEEZE:
                neutral_factors.append("Volatility squeeze - awaiting direction")

            if volatility_metrics.range_expansion:
                bullish_signals.append("Range expansion indicating momentum")
            else:
                neutral_factors.append("Volatility contracting")

            # Market strength signals
            if market_strength.buy_pressure in [BuySellPressure.HIGH, BuySellPressure.EXTREME]:
                bullish_signals.append("High buy pressure detected")
            elif market_strength.sell_pressure in [BuySellPressure.HIGH, BuySellPressure.EXTREME]:
                bearish_signals.append("High sell pressure detected")

            if market_strength.trend_strength in [MarketStrengthTrend.STRONG_BULLISH,
                                                  MarketStrengthTrend.MODERATE_BULLISH]:
                bullish_signals.append("Strong bullish market strength")
            elif market_strength.trend_strength in [MarketStrengthTrend.STRONG_BEARISH,
                                                    MarketStrengthTrend.MODERATE_BEARISH]:
                bearish_signals.append("Strong bearish market strength")

            # Determine overall signal alignment
            bullish_count = len(bullish_signals)
            bearish_count = len(bearish_signals)
            total_signals = bullish_count + bearish_count

            if total_signals == 0:
                signal_alignment = SignalAlignment.NEUTRAL
                confidence_score = 5.0
            else:
                bullish_ratio = bullish_count / total_signals

                if bullish_ratio >= 0.8:
                    signal_alignment = SignalAlignment.STRONG_BULLISH_BIAS
                    confidence_score = min(10, 7 + bullish_count * 0.5)
                elif bullish_ratio >= 0.65:
                    signal_alignment = SignalAlignment.BULLISH_BIAS
                    confidence_score = min(10, 6 + bullish_count * 0.3)
                elif bullish_ratio >= 0.55:
                    signal_alignment = SignalAlignment.NEUTRAL_BULLISH_BIAS
                    confidence_score = 5.5
                elif bullish_ratio >= 0.45:
                    signal_alignment = SignalAlignment.NEUTRAL
                    confidence_score = 5.0
                elif bullish_ratio >= 0.35:
                    signal_alignment = SignalAlignment.NEUTRAL_BEARISH_BIAS
                    confidence_score = 4.5
                elif bullish_ratio >= 0.2:
                    signal_alignment = SignalAlignment.BEARISH_BIAS
                    confidence_score = min(10, 6 + bearish_count * 0.3)
                else:
                    signal_alignment = SignalAlignment.STRONG_BEARISH_BIAS
                    confidence_score = min(10, 7 + bearish_count * 0.5)

            return Confluences(
                bullish_signals=bullish_signals,
                bearish_signals=bearish_signals,
                neutral_factors=neutral_factors,
                signal_alignment=signal_alignment,
                confidence_score=confidence_score
            )

        except Exception as e:
            print(f"Error analyzing confluences: {e}")
            return Confluences(
                bullish_signals=[],
                bearish_signals=[],
                neutral_factors=["Analysis pending"],
                signal_alignment=SignalAlignment.NEUTRAL,
                confidence_score=5.0
            )

    def create_indicator_summary(self, confluences: Confluences, market_strength: MarketStrength,
                                 rsi_data: RSIData) -> IndicatorSummary:
        """
        Create indicator summary with key observations and recommendations
        """
        try:
            # Determine primary signal
            if confluences.signal_alignment in [SignalAlignment.STRONG_BULLISH_BIAS, SignalAlignment.BULLISH_BIAS]:
                if market_strength.overall_score >= 8:
                    primary_signal = PrimarySignal.STRONG_BULLISH
                else:
                    primary_signal = PrimarySignal.BULLISH
            elif confluences.signal_alignment == SignalAlignment.NEUTRAL_BULLISH_BIAS:
                primary_signal = PrimarySignal.NEUTRAL_BULLISH
            elif confluences.signal_alignment == SignalAlignment.NEUTRAL:
                primary_signal = PrimarySignal.NEUTRAL
            elif confluences.signal_alignment == SignalAlignment.NEUTRAL_BEARISH_BIAS:
                primary_signal = PrimarySignal.NEUTRAL_BEARISH
            elif confluences.signal_alignment in [SignalAlignment.BEARISH_BIAS, SignalAlignment.STRONG_BEARISH_BIAS]:
                if market_strength.overall_score <= 3:
                    primary_signal = PrimarySignal.STRONG_BEARISH
                else:
                    primary_signal = PrimarySignal.BEARISH
            else:
                primary_signal = PrimarySignal.NEUTRAL

            # Determine signal strength
            if confluences.confidence_score >= 8:
                signal_strength = SignalStrength.VERY_STRONG
            elif confluences.confidence_score >= 7:
                signal_strength = SignalStrength.STRONG
            elif confluences.confidence_score >= 5.5:
                signal_strength = SignalStrength.MODERATE
            elif confluences.confidence_score >= 4:
                signal_strength = SignalStrength.WEAK
            else:
                signal_strength = SignalStrength.VERY_WEAK

            # Create key observation
            key_bullish = confluences.bullish_signals[:2]  # Top 2 bullish signals
            key_bearish = confluences.bearish_signals[:2]  # Top 2 bearish signals

            if key_bullish and not key_bearish:
                key_observation = f"{', '.join(key_bullish)} suggests potential upward move"
            elif key_bearish and not key_bullish:
                key_observation = f"{', '.join(key_bearish)} suggests potential downward move"
            elif key_bullish and key_bearish:
                key_observation = f"Mixed signals: {key_bullish[0]} vs {key_bearish[0]}"
            else:
                key_observation = "Market showing neutral technical conditions"

            # Create caution notes
            caution_notes = []
            if market_strength.market_phase == MarketPhase.LATE_TREND:
                caution_notes.append("Late trend phase - watch for reversal signs")
            if rsi_data.status == RSIStatus.OVERBOUGHT:
                caution_notes.append("RSI overbought - potential pullback risk")
            elif rsi_data.status == RSIStatus.OVERSOLD:
                caution_notes.append("RSI oversold - potential bounce opportunity")

            caution_text = "; ".join(caution_notes) if caution_notes else "Monitor for volatility changes"

            # Determine recommended action
            if primary_signal == PrimarySignal.STRONG_BULLISH:
                recommended_action = RecommendedAction.STRONG_BUY
            elif primary_signal == PrimarySignal.BULLISH:
                recommended_action = RecommendedAction.LOOK_FOR_LONG_ENTRIES
            elif primary_signal == PrimarySignal.NEUTRAL_BULLISH:
                recommended_action = RecommendedAction.HOLD
            elif primary_signal == PrimarySignal.NEUTRAL:
                recommended_action = RecommendedAction.NEUTRAL
            elif primary_signal == PrimarySignal.NEUTRAL_BEARISH:
                recommended_action = RecommendedAction.HOLD
            elif primary_signal == PrimarySignal.BEARISH:
                recommended_action = RecommendedAction.LOOK_FOR_SHORT_ENTRIES
            else:
                recommended_action = RecommendedAction.STRONG_SELL

            # Create invalidation scenario
            if primary_signal in [PrimarySignal.BULLISH, PrimarySignal.STRONG_BULLISH]:
                invalidation_scenario = "Loss of volume support or RSI breakdown below 40"
            elif primary_signal in [PrimarySignal.BEARISH, PrimarySignal.STRONG_BEARISH]:
                invalidation_scenario = "Volume surge with RSI recovery above 60"
            else:
                invalidation_scenario = "Significant change in volume patterns or momentum"

            return IndicatorSummary(
                primary_signal=primary_signal,
                signal_strength=signal_strength,
                key_observation=key_observation,
                caution_notes=caution_text,
                recommended_action=recommended_action,
                invalidation_scenario=invalidation_scenario
            )

        except Exception as e:
            print(f"Error creating indicator summary: {e}")
            return IndicatorSummary(
                primary_signal=PrimarySignal.NEUTRAL,
                signal_strength=SignalStrength.WEAK,
                key_observation="Technical analysis in progress",
                caution_notes="Monitor market conditions",
                recommended_action=RecommendedAction.NEUTRAL,
                invalidation_scenario="Significant market structure change"
            )

    def create_trading_conditions(self, market_strength: MarketStrength, volatility_metrics: VolatilityMetrics,
                                  confluences: Confluences) -> TradingConditions:
        """
        Create trading conditions assessment
        """
        try:
            # Determine market state
            if volatility_metrics.volatility_regime == VolatilityRegime.EXPANSION:
                market_state = MarketState.VOLATILE
            elif volatility_metrics.volatility_regime == VolatilityRegime.BREAKOUT:
                market_state = MarketState.TRENDING
            elif volatility_metrics.volatility_regime == VolatilityRegime.SQUEEZE:
                market_state = MarketState.RANGING
            else:
                if market_strength.overall_score > 6:
                    market_state = MarketState.TRENDING
                else:
                    market_state = MarketState.RANGING

            # Determine trend maturity
            if market_strength.market_phase == MarketPhase.EARLY_TREND:
                trend_maturity = TrendMaturity.EARLY
            elif market_strength.market_phase == MarketPhase.MATURE_TREND:
                trend_maturity = TrendMaturity.MIDDLE
            elif market_strength.market_phase == MarketPhase.LATE_TREND:
                trend_maturity = TrendMaturity.LATE
            elif market_strength.market_phase == MarketPhase.REVERSAL:
                trend_maturity = TrendMaturity.EXHAUSTED
            else:
                trend_maturity = TrendMaturity.EARLY

            # Determine optimal strategy
            if market_state == MarketState.TRENDING and confluences.confidence_score >= 7:
                optimal_strategy = OptimalStrategy.TREND_FOLLOWING
            elif market_state == MarketState.RANGING:
                optimal_strategy = OptimalStrategy.MEAN_REVERSION
            elif volatility_metrics.volatility_regime == VolatilityRegime.BREAKOUT:
                optimal_strategy = OptimalStrategy.BREAKOUT
            elif volatility_metrics.volatility_regime == VolatilityRegime.SQUEEZE:
                optimal_strategy = OptimalStrategy.WAIT_AND_SEE
            else:
                optimal_strategy = OptimalStrategy.SCALPING

            # Determine risk environment
            if volatility_metrics.atr_status == ATRStatus.EXTREME:
                risk_environment = RiskEnvironment.HIGH
            elif volatility_metrics.atr_status == ATRStatus.HIGH:
                risk_environment = RiskEnvironment.ELEVATED
            elif volatility_metrics.atr_status == ATRStatus.LOW:
                risk_environment = RiskEnvironment.LOW
            else:
                risk_environment = RiskEnvironment.NORMAL

            # Session alignment (simplified - assume always aligned for now)
            session_alignment = confluences.confidence_score >= 6

            return TradingConditions(
                market_state=market_state,
                trend_maturity=trend_maturity,
                optimal_strategy=optimal_strategy,
                risk_environment=risk_environment,
                session_alignment=session_alignment
            )

        except Exception as e:
            print(f"Error creating trading conditions: {e}")
            return TradingConditions(
                market_state=MarketState.RANGING,
                trend_maturity=TrendMaturity.EARLY,
                optimal_strategy=OptimalStrategy.WAIT_AND_SEE,
                risk_environment=RiskEnvironment.NORMAL,
                session_alignment=False
            )

    def analyze_technical_indicators(self, symbol: str, timeframe: str,
                                     as_of_datetime: Optional[str] = None) -> Dict[str, Any]:
        """
        Main function to analyze technical indicators
        """
        current_price = 95000  # Will be updated with real data

        # Fetch data
        binance_fetcher = BinanceDataFetcher()

        try:
            # Parse end time if provided
            end_time_ms = None
            if as_of_datetime:
                as_of_dt = datetime.fromisoformat(as_of_datetime.replace('Z', '+00:00'))
                end_time_ms = int(as_of_dt.timestamp() * 1000)

            # Fetch klines data
            import asyncio
            klines_data = asyncio.run(binance_fetcher.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=200,
                end_time=end_time_ms
            ))
            formatted_data = binance_fetcher.format_klines_data(klines_data)
            df = self.prepare_dataframe(formatted_data)
            current_price = float(formatted_data['close'][-1])

            # Perform technical analysis
            rsi_data = self.analyze_rsi(df)
            momentum_oscillator = self.analyze_momentum_oscillator(df)
            volume_analysis = self.analyze_volume(df)
            volatility_metrics = self.analyze_volatility(df)
            market_strength = self.calculate_market_strength(df, rsi_data, volume_analysis, volatility_metrics)
            confluences = self.analyze_confluences(rsi_data, volume_analysis, volatility_metrics, market_strength)
            indicator_summary = self.create_indicator_summary(confluences, market_strength, rsi_data)
            trading_conditions = self.create_trading_conditions(market_strength, volatility_metrics, confluences)

            return {
                "current_price": current_price,
                "momentum_indicators": MomentumIndicators(
                    rsi=rsi_data,
                    momentum_oscillator=momentum_oscillator
                ),
                "volume_analysis": volume_analysis,
                "volatility_metrics": volatility_metrics,
                "market_strength": market_strength,
                "confluences": confluences,
                "indicator_summary": indicator_summary,
                "trading_conditions": trading_conditions
            }

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            # Return default analysis
            return {
                "current_price": current_price,
                "momentum_indicators": MomentumIndicators(
                    rsi=RSIData(
                        status=RSIStatus.NEUTRAL,
                        value_range="40-60",
                        trend=TrendDirection.SIDEWAYS,
                        divergence=DivergenceData(detected=False),
                        key_levels=RSIKeyLevels(oversold=False, overbought=False)
                    ),
                    momentum_oscillator=MomentumOscillator(
                        status=MomentumStatus.NEUTRAL,
                        strength=SignalStrength.WEAK,
                        acceleration=MomentumAcceleration.STEADY,
                        cross_signal=CrossSignal.NONE
                    )
                ),
                "volume_analysis": VolumeAnalysis(
                    current_volume_profile=VolumeProfile.AVERAGE,
                    volume_trend=TrendDirection.SIDEWAYS,
                    volume_confirmation=False,
                    unusual_activity=UnusualActivity(detected=False),
                    volume_patterns=VolumePatterns(
                        accumulation=False,
                        distribution=False,
                        pattern=VolumePattern.RISING_PRICE_RISING_VOLUME
                    ),
                    smart_money_flow=SmartMoneyFlow(
                        direction=SmartMoneyDirection.NEUTRAL,
                        strength=SmartMoneyStrength.WEAK,
                        persistence="0_BARS"
                    )
                ),
                "volatility_metrics": VolatilityMetrics(
                    atr_status=ATRStatus.NORMAL,
                    volatility_trend=TrendDirection.SIDEWAYS,
                    current_range=CurrentRange.MEDIUM,
                    range_expansion=False,
                    suggested_stop_distance="2.0%",
                    volatility_regime=VolatilityRegime.CONSOLIDATION
                ),
                "market_strength": MarketStrength(
                    overall_score=5.0,
                    trend_strength=MarketStrengthTrend.NEUTRAL,
                    buy_pressure=BuySellPressure.MODERATE,
                    sell_pressure=BuySellPressure.MODERATE,
                    strength_components=StrengthComponents(
                        price_action=5.0,
                        volume=5.0,
                        momentum=5.0,
                        volatility=5.0
                    ),
                    market_phase=MarketPhase.CONSOLIDATION
                ),
                "confluences": Confluences(
                    bullish_signals=[],
                    bearish_signals=[],
                    neutral_factors=["Analysis pending"],
                    signal_alignment=SignalAlignment.NEUTRAL,
                    confidence_score=5.0
                ),
                "indicator_summary": IndicatorSummary(
                    primary_signal=PrimarySignal.NEUTRAL,
                    signal_strength=SignalStrength.WEAK,
                    key_observation="Technical analysis in progress",
                    caution_notes="Monitor market conditions",
                    recommended_action=RecommendedAction.NEUTRAL,
                    invalidation_scenario="Significant market structure change"
                ),
                "trading_conditions": TradingConditions(
                    market_state=MarketState.RANGING,
                    trend_maturity=TrendMaturity.EARLY,
                    optimal_strategy=OptimalStrategy.WAIT_AND_SEE,
                    risk_environment=RiskEnvironment.NORMAL,
                    session_alignment=False
                )
            }


@router.get("/api/technical-indicators/{symbol}/{timeframe}",
            response_model=TechnicalIndicatorsResponse,
            summary="Get Technical Indicators Analysis",
            description="Analyze cryptocurrency using traditional technical indicators with GPT-optimized output")
async def get_technical_indicators(
        symbol: str = Path(..., description="Trading pair symbol (e.g., BTCUSDT, BTC)"),
        timeframe: str = Path(..., description="Timeframe for analysis (e.g., 1h, 4h, 1d)"),
        as_of_datetime: Optional[str] = Query(None,
                                              description="ISO 8601 datetime string for historical analysis (e.g., 2024-01-01T00:00:00Z). Defaults to current time."),
        _: bool = Depends(verify_api_key)
):
    """
    Analyze cryptocurrency using traditional technical indicators optimized for GPT processing.
    
    This endpoint:
    1. Focuses on indicators that complement SMC/ICT strategies
    2. Converts numerical values to descriptive states for easier GPT processing
    3. Emphasizes divergences, extremes, and confluence signals
    4. Provides volume confirmation and smart money flow analysis
    5. Includes comprehensive market strength scoring
    6. Delivers clear trading recommendations and risk assessment
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., 'BTC', 'ETH', 'BTCUSDT')
    - **timeframe**: Any valid Binance timeframe (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
    - **as_of_datetime**: Optional ISO 8601 datetime string for historical analysis
    
    **Core Indicators Analyzed:**
    - **RSI (14)**: Momentum and divergence detection with state-based output
    - **Volume Analysis**: Smart money flow, accumulation/distribution patterns
    - **ATR (14)**: Volatility regime identification and stop-loss suggestions
    - **Market Strength Index**: Composite scoring across multiple factors
    - **MACD**: Momentum oscillator for trend confirmation
    
    **Key Features:**
    - Status-based output (OVERSOLD/NEUTRAL/OVERBOUGHT) instead of raw numbers
    - Automatic divergence detection with strength assessment
    - Volume spike detection and smart money flow analysis
    - Volatility regime classification (EXPANSION/CONSOLIDATION/BREAKOUT/SQUEEZE)
    - Comprehensive signal confluence analysis
    - Clear trading recommendations and invalidation scenarios
    
    **Returns comprehensive analysis including:**
    - Momentum indicators with divergence detection
    - Volume analysis with smart money flow assessment
    - Volatility metrics with suggested stop distances
    - Market strength scoring with component breakdown
    - Signal confluences with confidence scoring
    - Trading recommendations and risk environment assessment
    
    **Simplified Output Strategy:**
    - No raw numerical values that burden GPT processing
    - Descriptive labels for easy interpretation
    - Integrated signals for reduced complexity
    - Focus on actionable insights rather than data points
    
    **Note:** If the symbol doesn't end with 'USDT', it will be automatically appended.
    """
    try:
        # Initialize analyzer
        analyzer = TechnicalIndicatorAnalyzer()

        # Process symbol
        processed_symbol = symbol.upper()
        if not processed_symbol.endswith('USDT'):
            processed_symbol = f"{processed_symbol}USDT"

        # Perform technical indicators analysis
        analysis_result = analyzer.analyze_technical_indicators(processed_symbol, timeframe, as_of_datetime)

        # Create response
        return TechnicalIndicatorsResponse(
            symbol=processed_symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc).isoformat() if as_of_datetime is None else as_of_datetime,
            current_price=analysis_result["current_price"],
            momentum_indicators=analysis_result["momentum_indicators"],
            volume_analysis=analysis_result["volume_analysis"],
            volatility_metrics=analysis_result["volatility_metrics"],
            market_strength=analysis_result["market_strength"],
            confluences=analysis_result["confluences"],
            indicator_summary=analysis_result["indicator_summary"],
            trading_conditions=analysis_result["trading_conditions"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Technical indicators analysis failed: {str(e)}")
