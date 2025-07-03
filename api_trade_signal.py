from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field
from smartmoneyconcepts import smc

from binance_data import BinanceDataFetcher
from smc_analysis import SMCAnalyzer
from utils import verify_api_key


class SignalDirection(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalStrength(str, Enum):
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    VERY_WEAK = "VERY_WEAK"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class TradingAction(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    HOLD_LONG = "HOLD_LONG"
    HOLD_SHORT = "HOLD_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    WAIT = "WAIT"


class MarketCondition(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"


class ConfidenceLevel(str, Enum):
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class SignalComponent(BaseModel):
    name: str = Field(..., description="Component name")
    value: float = Field(..., description="Component value/score")
    weight: float = Field(..., description="Component weight in final signal")
    contribution: float = Field(..., description="Contribution to final signal")
    status: str = Field(..., description="Component status description")


class TechnicalSignal(BaseModel):
    rsi_signal: float = Field(..., description="RSI signal contribution")
    macd_signal: float = Field(..., description="MACD signal contribution")
    ma_signal: float = Field(..., description="Moving average signal contribution")
    volume_signal: float = Field(..., description="Volume signal contribution")
    volatility_signal: float = Field(..., description="Volatility signal contribution")
    combined_score: float = Field(..., description="Combined technical score")


class SMCSignal(BaseModel):
    market_structure_signal: float = Field(..., description="Market structure signal")
    liquidity_signal: float = Field(..., description="Liquidity signal")
    supply_demand_signal: float = Field(..., description="Supply/demand signal")
    order_block_signal: float = Field(..., description="Order block signal")
    fair_value_gap_signal: float = Field(..., description="Fair value gap signal")
    combined_score: float = Field(..., description="Combined SMC score")


class RiskMetrics(BaseModel):
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    volatility_risk: float = Field(..., description="Volatility risk score")
    liquidity_risk: float = Field(..., description="Liquidity risk score")
    market_risk: float = Field(..., description="Market condition risk score")
    suggested_position_size: float = Field(..., description="Suggested position size percentage")
    max_drawdown_estimate: float = Field(..., description="Estimated max drawdown percentage")


class EntryExitLevels(BaseModel):
    entry_price: float = Field(..., description="Suggested entry price")
    stop_loss: float = Field(..., description="Stop loss level")
    take_profit_1: float = Field(..., description="First take profit level")
    take_profit_2: float = Field(..., description="Second take profit level")
    take_profit_3: Optional[float] = Field(None, description="Third take profit level")
    risk_reward_ratio: float = Field(..., description="Risk to reward ratio")


class MarketContext(BaseModel):
    current_condition: MarketCondition = Field(..., description="Current market condition")
    trend_direction: str = Field(..., description="Primary trend direction")
    trend_strength: float = Field(..., description="Trend strength score")
    support_level: float = Field(..., description="Key support level")
    resistance_level: float = Field(..., description="Key resistance level")
    consolidation_range: Optional[Dict[str, float]] = Field(None, description="Consolidation range if applicable")


class SignalValidation(BaseModel):
    signal_confirmed: bool = Field(..., description="Whether signal is confirmed")
    confirmation_factors: List[str] = Field(..., description="Factors confirming the signal")
    warning_factors: List[str] = Field(..., description="Warning factors against the signal")
    time_horizon: str = Field(..., description="Recommended time horizon for signal")
    invalidation_level: float = Field(..., description="Level that would invalidate signal")


class TradeSignalResponse(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Analysis timeframe")
    timestamp: str = Field(..., description="Signal generation timestamp")
    current_price: float = Field(..., description="Current price")

    # Core Signal Data
    signal_direction: SignalDirection = Field(..., description="Primary signal direction")
    signal_strength: SignalStrength = Field(..., description="Signal strength assessment")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence in signal")
    recommended_action: TradingAction = Field(..., description="Recommended trading action")

    # Signal Components
    technical_signal: TechnicalSignal = Field(..., description="Technical analysis signals")
    smc_signal: SMCSignal = Field(..., description="Smart Money Concepts signals")
    signal_components: List[SignalComponent] = Field(..., description="Individual signal components")

    # Risk and Levels
    risk_metrics: RiskMetrics = Field(..., description="Risk assessment metrics")
    entry_exit_levels: EntryExitLevels = Field(..., description="Entry and exit levels")

    # Market Context
    market_context: MarketContext = Field(..., description="Current market context")
    signal_validation: SignalValidation = Field(..., description="Signal validation data")

    # Summary
    key_message: str = Field(..., description="Key takeaway message")
    execution_notes: str = Field(..., description="Execution notes and considerations")


# Create router
router = APIRouter(tags=["Trade Signals"])


class TradeSignalAnalyzer:
    """
    Comprehensive trade signal analyzer combining technical indicators and SMC analysis
    """

    def __init__(self):
        self.smc_analyzer = SMCAnalyzer()

    def prepare_dataframe(self, ohlcv_data: Dict[str, List[float]]) -> pd.DataFrame:
        """Convert OHLCV data to pandas DataFrame"""
        df = pd.DataFrame(ohlcv_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        return df

    def calculate_technical_signals(self, df: pd.DataFrame) -> TechnicalSignal:
        """Calculate technical indicator signals"""
        try:
            # RSI Signal
            rsi = ta.rsi(df['close'], length=14)
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50

            if current_rsi <= 30:
                rsi_signal = 2.0  # Strong buy
            elif current_rsi <= 40:
                rsi_signal = 1.0  # Buy
            elif current_rsi <= 60:
                rsi_signal = 0.0  # Neutral
            elif current_rsi <= 70:
                rsi_signal = -1.0  # Sell
            else:
                rsi_signal = -2.0  # Strong sell

            # MACD Signal
            macd_data = ta.macd(df['close'])
            if macd_data is not None and not macd_data.empty:
                macd_line = macd_data.iloc[-1, 0]
                signal_line = macd_data.iloc[-1, 1]
                histogram = macd_data.iloc[-1, 2]

                if macd_line > signal_line and histogram > 0:
                    macd_signal = 1.5
                elif macd_line > signal_line:
                    macd_signal = 1.0
                elif macd_line < signal_line and histogram < 0:
                    macd_signal = -1.5
                else:
                    macd_signal = -1.0
            else:
                macd_signal = 0.0

            # Moving Average Signal
            ma_short = ta.sma(df['close'], length=20)
            ma_long = ta.sma(df['close'], length=50)

            if not ma_short.empty and not ma_long.empty:
                current_price = df['close'].iloc[-1]
                ma_short_val = ma_short.iloc[-1]
                ma_long_val = ma_long.iloc[-1]

                if current_price > ma_short_val > ma_long_val:
                    ma_signal = 2.0
                elif current_price > ma_short_val:
                    ma_signal = 1.0
                elif current_price < ma_short_val < ma_long_val:
                    ma_signal = -2.0
                else:
                    ma_signal = -1.0
            else:
                ma_signal = 0.0

            # Volume Signal
            volume = df['volume']
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]

            if current_volume > avg_volume * 1.5:
                volume_signal = 1.0
            elif current_volume < avg_volume * 0.5:
                volume_signal = -1.0
            else:
                volume_signal = 0.0

            # Volatility Signal (using ATR)
            atr = ta.atr(df['high'], df['low'], df['close'], length=14)
            if not atr.empty:
                current_atr = atr.iloc[-1]
                avg_atr = atr.rolling(20).mean().iloc[-1]

                if current_atr > avg_atr * 1.3:
                    volatility_signal = -0.5  # High volatility reduces signal
                elif current_atr < avg_atr * 0.7:
                    volatility_signal = 0.5  # Low volatility increases signal
                else:
                    volatility_signal = 0.0
            else:
                volatility_signal = 0.0

            # Combined technical score
            weights = {'rsi': 0.25, 'macd': 0.25, 'ma': 0.25, 'volume': 0.15, 'volatility': 0.10}
            combined_score = (
                    rsi_signal * weights['rsi'] +
                    macd_signal * weights['macd'] +
                    ma_signal * weights['ma'] +
                    volume_signal * weights['volume'] +
                    volatility_signal * weights['volatility']
            )

            return TechnicalSignal(
                rsi_signal=rsi_signal,
                macd_signal=macd_signal,
                ma_signal=ma_signal,
                volume_signal=volume_signal,
                volatility_signal=volatility_signal,
                combined_score=combined_score
            )

        except Exception as e:
            print(f"Error calculating technical signals: {e}")
            return TechnicalSignal(
                rsi_signal=0.0,
                macd_signal=0.0,
                ma_signal=0.0,
                volume_signal=0.0,
                volatility_signal=0.0,
                combined_score=0.0
            )

    def calculate_smc_signals(self, df: pd.DataFrame) -> SMCSignal:
        """Calculate Smart Money Concepts signals"""
        try:
            # Market Structure Signal
            structure_score = self.analyze_market_structure(df)

            # Liquidity Signal
            liquidity_score = self.analyze_liquidity_zones(df)

            # Supply/Demand Signal
            supply_demand_score = self.analyze_supply_demand(df)

            # Order Block Signal
            order_block_score = self.analyze_order_blocks(df)

            # Fair Value Gap Signal
            fvg_score = self.analyze_fair_value_gaps(df)

            # Combined SMC score
            weights = {'structure': 0.25, 'liquidity': 0.20, 'supply_demand': 0.25, 'order_block': 0.15, 'fvg': 0.15}
            combined_score = (
                    structure_score * weights['structure'] +
                    liquidity_score * weights['liquidity'] +
                    supply_demand_score * weights['supply_demand'] +
                    order_block_score * weights['order_block'] +
                    fvg_score * weights['fvg']
            )

            return SMCSignal(
                market_structure_signal=structure_score,
                liquidity_signal=liquidity_score,
                supply_demand_signal=supply_demand_score,
                order_block_signal=order_block_score,
                fair_value_gap_signal=fvg_score,
                combined_score=combined_score
            )

        except Exception as e:
            print(f"Error calculating SMC signals: {e}")
            return SMCSignal(
                market_structure_signal=0.0,
                liquidity_signal=0.0,
                supply_demand_signal=0.0,
                order_block_signal=0.0,
                fair_value_gap_signal=0.0,
                combined_score=0.0
            )

    def analyze_market_structure(self, df: pd.DataFrame) -> float:
        """Analyze market structure for signal generation"""
        try:
            # Use SMC library for swing highs/lows
            swing_highs = smc.swing_highs_lows(df, swing_length=10)

            if swing_highs is not None and not swing_highs.empty:
                # Analyze recent swing pattern
                recent_highs = swing_highs[swing_highs['HighLow'] == 1].tail(3)
                recent_lows = swing_highs[swing_highs['HighLow'] == -1].tail(3)

                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    # Check for higher highs and higher lows (bullish)
                    if (recent_highs.iloc[-1]['Level'] > recent_highs.iloc[-2]['Level'] and
                            recent_lows.iloc[-1]['Level'] > recent_lows.iloc[-2]['Level']):
                        return 1.5  # Strong bullish structure
                    # Check for lower highs and lower lows (bearish)
                    elif (recent_highs.iloc[-1]['Level'] < recent_highs.iloc[-2]['Level'] and
                          recent_lows.iloc[-1]['Level'] < recent_lows.iloc[-2]['Level']):
                        return -1.5  # Strong bearish structure

            return 0.0  # Neutral structure

        except Exception:
            return 0.0

    def analyze_liquidity_zones(self, df: pd.DataFrame) -> float:
        """Analyze liquidity zones for signal generation"""
        try:
            # Simple liquidity analysis based on volume and price action
            volume = df['volume']
            high_volume_bars = volume > volume.quantile(0.8)

            if high_volume_bars.tail(5).sum() >= 3:
                return 1.0  # High liquidity environment
            elif high_volume_bars.tail(10).sum() <= 2:
                return -0.5  # Low liquidity environment

            return 0.0

        except Exception:
            return 0.0

    def analyze_supply_demand(self, df: pd.DataFrame) -> float:
        """Analyze supply and demand zones"""
        try:
            # Calculate buying/selling pressure
            close = df['close']
            volume = df['volume']

            # Price-volume relationship
            price_change = close.pct_change()
            volume_change = volume.pct_change()

            # Correlation between price and volume changes
            correlation = price_change.corr(volume_change)

            if correlation > 0.3:
                return 1.0  # Strong demand
            elif correlation < -0.3:
                return -1.0  # Strong supply

            return 0.0

        except Exception:
            return 0.0

    def analyze_order_blocks(self, df: pd.DataFrame) -> float:
        """Analyze order blocks for signal generation"""
        try:
            # Simplified order block analysis
            close = df['close']
            volume = df['volume']

            # Look for high volume candles with small bodies
            body_size = abs(close - df['open']) / df['open']
            high_volume_small_body = (volume > volume.quantile(0.8)) & (body_size < 0.01)

            if high_volume_small_body.tail(5).sum() >= 1:
                return 0.5  # Potential order block

            return 0.0

        except Exception:
            return 0.0

    def analyze_fair_value_gaps(self, df: pd.DataFrame) -> float:
        """Analyze fair value gaps"""
        try:
            # Simple FVG detection
            high = df['high']
            low = df['low']

            # Check for gaps
            gaps = []
            for i in range(2, len(df)):
                if low.iloc[i] > high.iloc[i - 2]:  # Bullish FVG
                    gaps.append(1)
                elif high.iloc[i] < low.iloc[i - 2]:  # Bearish FVG
                    gaps.append(-1)

            if gaps:
                recent_gaps = gaps[-5:]  # Last 5 gaps
                gap_signal = sum(recent_gaps) / len(recent_gaps)
                return gap_signal

            return 0.0

        except Exception:
            return 0.0

    def calculate_risk_metrics(self, df: pd.DataFrame, signal_strength: float) -> RiskMetrics:
        """Calculate risk metrics for the trade signal"""
        try:
            # Volatility risk
            returns = df['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility

            if volatility > 0.5:
                volatility_risk = 0.8
                risk_level = RiskLevel.VERY_HIGH
            elif volatility > 0.3:
                volatility_risk = 0.6
                risk_level = RiskLevel.HIGH
            elif volatility > 0.2:
                volatility_risk = 0.4
                risk_level = RiskLevel.MEDIUM
            else:
                volatility_risk = 0.2
                risk_level = RiskLevel.LOW

            # Liquidity risk (simplified)
            volume = df['volume']
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]

            if current_volume < avg_volume * 0.5:
                liquidity_risk = 0.7
            elif current_volume < avg_volume * 0.8:
                liquidity_risk = 0.4
            else:
                liquidity_risk = 0.2

            # Market risk
            market_risk = min(0.8, abs(signal_strength) * 0.2)

            # Position sizing
            base_position = 0.02  # 2% base position
            risk_adjustment = 1 - (volatility_risk + liquidity_risk + market_risk) / 3
            suggested_position_size = base_position * risk_adjustment

            # Max drawdown estimate
            max_drawdown_estimate = volatility * 0.5  # Simplified estimate

            return RiskMetrics(
                risk_level=risk_level,
                volatility_risk=volatility_risk,
                liquidity_risk=liquidity_risk,
                market_risk=market_risk,
                suggested_position_size=max(0.005, suggested_position_size),  # Minimum 0.5%
                max_drawdown_estimate=max_drawdown_estimate
            )

        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                risk_level=RiskLevel.MEDIUM,
                volatility_risk=0.4,
                liquidity_risk=0.3,
                market_risk=0.3,
                suggested_position_size=0.01,
                max_drawdown_estimate=0.1
            )

    def calculate_entry_exit_levels(self, df: pd.DataFrame, signal_direction: float) -> EntryExitLevels:
        """Calculate entry and exit levels"""
        try:
            current_price = df['close'].iloc[-1]
            atr = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1]

            if signal_direction > 0:  # Long signal
                entry_price = current_price
                stop_loss = current_price - (atr * 2)
                take_profit_1 = current_price + (atr * 1.5)
                take_profit_2 = current_price + (atr * 3)
                take_profit_3 = current_price + (atr * 4.5)
            else:  # Short signal
                entry_price = current_price
                stop_loss = current_price + (atr * 2)
                take_profit_1 = current_price - (atr * 1.5)
                take_profit_2 = current_price - (atr * 3)
                take_profit_3 = current_price - (atr * 4.5)

            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit_1 - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 1.0

            return EntryExitLevels(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                take_profit_3=take_profit_3,
                risk_reward_ratio=risk_reward_ratio
            )

        except Exception as e:
            print(f"Error calculating entry/exit levels: {e}")
            current_price = df['close'].iloc[-1]
            return EntryExitLevels(
                entry_price=current_price,
                stop_loss=current_price * 0.98,
                take_profit_1=current_price * 1.02,
                take_profit_2=current_price * 1.04,
                take_profit_3=current_price * 1.06,
                risk_reward_ratio=1.0
            )

    def analyze_market_context(self, df: pd.DataFrame) -> MarketContext:
        """Analyze current market context"""
        try:
            # Trend analysis
            ma_short = ta.sma(df['close'], length=20).iloc[-1]
            ma_long = ta.sma(df['close'], length=50).iloc[-1]
            current_price = df['close'].iloc[-1]

            if current_price > ma_short > ma_long:
                current_condition = MarketCondition.TRENDING_UP
                trend_direction = "BULLISH"
                trend_strength = 0.8
            elif current_price < ma_short < ma_long:
                current_condition = MarketCondition.TRENDING_DOWN
                trend_direction = "BEARISH"
                trend_strength = 0.8
            else:
                current_condition = MarketCondition.RANGING
                trend_direction = "NEUTRAL"
                trend_strength = 0.3

            # Support and resistance levels
            recent_highs = df['high'].rolling(20).max().iloc[-1]
            recent_lows = df['low'].rolling(20).min().iloc[-1]

            return MarketContext(
                current_condition=current_condition,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                support_level=recent_lows,
                resistance_level=recent_highs
            )

        except Exception as e:
            print(f"Error analyzing market context: {e}")
            current_price = df['close'].iloc[-1]
            return MarketContext(
                current_condition=MarketCondition.RANGING,
                trend_direction="NEUTRAL",
                trend_strength=0.5,
                support_level=current_price * 0.95,
                resistance_level=current_price * 1.05
            )

    def validate_signal(self, technical_signal: TechnicalSignal, smc_signal: SMCSignal,
                        market_context: MarketContext) -> SignalValidation:
        """Validate the generated signal"""
        try:
            confirmation_factors = []
            warning_factors = []

            # Check technical and SMC alignment
            if (technical_signal.combined_score > 0.5 and smc_signal.combined_score > 0.5) or \
                    (technical_signal.combined_score < -0.5 and smc_signal.combined_score < -0.5):
                confirmation_factors.append("Technical and SMC signals aligned")
                signal_confirmed = True
            else:
                warning_factors.append("Technical and SMC signals diverging")
                signal_confirmed = False

            # Check market context alignment
            if market_context.trend_strength > 0.6:
                confirmation_factors.append("Strong trending market")
            else:
                warning_factors.append("Weak or ranging market")

            # Time horizon based on signal strength
            combined_strength = abs(technical_signal.combined_score + smc_signal.combined_score) / 2
            if combined_strength > 1.5:
                time_horizon = "1-3 days"
            elif combined_strength > 1.0:
                time_horizon = "3-7 days"
            else:
                time_horizon = "1-2 weeks"

            # Invalidation level
            current_price = market_context.support_level if technical_signal.combined_score > 0 else market_context.resistance_level
            invalidation_level = current_price * 0.98 if technical_signal.combined_score > 0 else current_price * 1.02

            return SignalValidation(
                signal_confirmed=signal_confirmed,
                confirmation_factors=confirmation_factors,
                warning_factors=warning_factors,
                time_horizon=time_horizon,
                invalidation_level=invalidation_level
            )

        except Exception as e:
            print(f"Error validating signal: {e}")
            return SignalValidation(
                signal_confirmed=False,
                confirmation_factors=["Signal analysis pending"],
                warning_factors=["Unable to validate signal"],
                time_horizon="Unknown",
                invalidation_level=0.0
            )

    async def generate_trade_signal(self, symbol: str, timeframe: str,
                              as_of_datetime: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive trade signal"""
        try:
            # Fetch data
            binance_fetcher = BinanceDataFetcher()

            # Parse end time if provided
            end_time_ms = None
            if as_of_datetime:
                as_of_dt = datetime.fromisoformat(as_of_datetime.replace('Z', '+00:00'))
                end_time_ms = int(as_of_dt.timestamp() * 1000)

            # Get klines data
            klines_data = await binance_fetcher.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=200,
                end_time=end_time_ms
            )

            formatted_data = binance_fetcher.format_klines_data(klines_data)
            df = self.prepare_dataframe(formatted_data)
            current_price = float(formatted_data['close'][-1])

            # Calculate signals
            technical_signal = self.calculate_technical_signals(df)
            smc_signal = self.calculate_smc_signals(df)

            # Combined signal score
            combined_score = (technical_signal.combined_score + smc_signal.combined_score) / 2

            # Determine signal direction and strength
            if combined_score >= 1.5:
                signal_direction = SignalDirection.STRONG_BUY
                signal_strength = SignalStrength.VERY_STRONG
                confidence_level = ConfidenceLevel.VERY_HIGH
                recommended_action = TradingAction.ENTER_LONG
            elif combined_score >= 1.0:
                signal_direction = SignalDirection.BUY
                signal_strength = SignalStrength.STRONG
                confidence_level = ConfidenceLevel.HIGH
                recommended_action = TradingAction.ENTER_LONG
            elif combined_score >= 0.5:
                signal_direction = SignalDirection.WEAK_BUY
                signal_strength = SignalStrength.MODERATE
                confidence_level = ConfidenceLevel.MEDIUM
                recommended_action = TradingAction.HOLD_LONG
            elif combined_score <= -1.5:
                signal_direction = SignalDirection.STRONG_SELL
                signal_strength = SignalStrength.VERY_STRONG
                confidence_level = ConfidenceLevel.VERY_HIGH
                recommended_action = TradingAction.ENTER_SHORT
            elif combined_score <= -1.0:
                signal_direction = SignalDirection.SELL
                signal_strength = SignalStrength.STRONG
                confidence_level = ConfidenceLevel.HIGH
                recommended_action = TradingAction.ENTER_SHORT
            elif combined_score <= -0.5:
                signal_direction = SignalDirection.WEAK_SELL
                signal_strength = SignalStrength.MODERATE
                confidence_level = ConfidenceLevel.MEDIUM
                recommended_action = TradingAction.HOLD_SHORT
            else:
                signal_direction = SignalDirection.NEUTRAL
                signal_strength = SignalStrength.WEAK
                confidence_level = ConfidenceLevel.LOW
                recommended_action = TradingAction.WAIT

            # Calculate other components
            risk_metrics = self.calculate_risk_metrics(df, combined_score)
            entry_exit_levels = self.calculate_entry_exit_levels(df, combined_score)
            market_context = self.analyze_market_context(df)
            signal_validation = self.validate_signal(technical_signal, smc_signal, market_context)

            # Signal components
            signal_components = [
                SignalComponent(
                    name="RSI",
                    value=technical_signal.rsi_signal,
                    weight=0.25,
                    contribution=technical_signal.rsi_signal * 0.25,
                    status="RSI analysis component"
                ),
                SignalComponent(
                    name="MACD",
                    value=technical_signal.macd_signal,
                    weight=0.25,
                    contribution=technical_signal.macd_signal * 0.25,
                    status="MACD analysis component"
                ),
                SignalComponent(
                    name="Market Structure",
                    value=smc_signal.market_structure_signal,
                    weight=0.25,
                    contribution=smc_signal.market_structure_signal * 0.25,
                    status="SMC market structure component"
                ),
                SignalComponent(
                    name="Supply/Demand",
                    value=smc_signal.supply_demand_signal,
                    weight=0.25,
                    contribution=smc_signal.supply_demand_signal * 0.25,
                    status="SMC supply/demand component"
                )
            ]

            # Generate key message and execution notes
            if signal_direction in [SignalDirection.STRONG_BUY, SignalDirection.BUY]:
                key_message = f"Strong bullish signal detected with {confidence_level.value} confidence"
                execution_notes = "Consider entering long position with proper risk management"
            elif signal_direction in [SignalDirection.STRONG_SELL, SignalDirection.SELL]:
                key_message = f"Strong bearish signal detected with {confidence_level.value} confidence"
                execution_notes = "Consider entering short position with proper risk management"
            else:
                key_message = "Neutral market conditions, wait for clearer signals"
                execution_notes = "Monitor for better entry opportunities"

            return {
                "current_price": current_price,
                "signal_direction": signal_direction,
                "signal_strength": signal_strength,
                "confidence_level": confidence_level,
                "recommended_action": recommended_action,
                "technical_signal": technical_signal,
                "smc_signal": smc_signal,
                "signal_components": signal_components,
                "risk_metrics": risk_metrics,
                "entry_exit_levels": entry_exit_levels,
                "market_context": market_context,
                "signal_validation": signal_validation,
                "key_message": key_message,
                "execution_notes": execution_notes
            }

        except Exception as e:
            print(f"Error generating trade signal: {e}")
            # Re-raise the exception to be handled by the API endpoint
            raise Exception(f"Failed to generate trade signal for {symbol}: {str(e)}")


@router.get("/api/trade-signal/{symbol}/{timeframe}",
            response_model=TradeSignalResponse,
            summary="Generate Comprehensive Trade Signal",
            description="Generate comprehensive trade signals combining technical analysis and Smart Money Concepts")
async def get_trade_signal(
        symbol: str = Path(..., description="Trading pair symbol (e.g., BTCUSDT, BTC)"),
        timeframe: str = Path(..., description="Timeframe for analysis (e.g., 1h, 4h, 1d)"),
        as_of_datetime: Optional[str] = Query(None,
                                              description="ISO 8601 datetime string for historical analysis (e.g., 2024-01-01T00:00:00Z). Defaults to current time."),
        _: bool = Depends(verify_api_key)
):
    """
    Generate comprehensive trade signals by combining technical indicators with Smart Money Concepts analysis.
    
    This endpoint provides:
    - **Technical Analysis**: RSI, MACD, Moving Averages, Volume, Volatility
    - **Smart Money Concepts**: Market Structure, Liquidity, Supply/Demand, Order Blocks, Fair Value Gaps
    - **Risk Assessment**: Volatility, Liquidity, and Market Risk metrics
    - **Entry/Exit Levels**: Precise entry points with multiple take profit levels
    - **Signal Validation**: Confirmation factors and warning signals
    - **Market Context**: Current market conditions and trend analysis
    
    **Signal Directions:**
    - STRONG_BUY/STRONG_SELL: Very high confidence signals
    - BUY/SELL: High confidence signals  
    - WEAK_BUY/WEAK_SELL: Moderate confidence signals
    - NEUTRAL: No clear directional bias
    
    **Signal Strength Levels:**
    - VERY_STRONG: Exceptional signal quality
    - STRONG: High quality signal
    - MODERATE: Decent signal quality
    - WEAK: Low quality signal
    - VERY_WEAK: Very poor signal quality
    
    **Risk Levels:**
    - LOW: Conservative trade setup
    - MEDIUM: Moderate risk trade
    - HIGH: High risk trade
    - VERY_HIGH: Extremely risky trade
    
    **Trading Actions:**
    - ENTER_LONG/ENTER_SHORT: Initiate new position
    - HOLD_LONG/HOLD_SHORT: Maintain existing position
    - EXIT_LONG/EXIT_SHORT: Close existing position
    - WAIT: No action recommended
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., 'BTC', 'ETH', 'BTCUSDT')
    - **timeframe**: Any valid Binance timeframe (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
    - **as_of_datetime**: Optional ISO 8601 datetime string for historical analysis
    
    **Note:** If the symbol doesn't end with 'USDT', it will be automatically appended.
    """
    try:
        # Initialize analyzer
        analyzer = TradeSignalAnalyzer()

        # Process symbol
        processed_symbol = symbol.upper()
        if not processed_symbol.endswith('USDT'):
            processed_symbol = f"{processed_symbol}USDT"

        # Generate trade signal
        signal_result = await analyzer.generate_trade_signal(processed_symbol, timeframe, as_of_datetime)

        # Create response
        return TradeSignalResponse(
            symbol=processed_symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc).isoformat() if as_of_datetime is None else as_of_datetime,
            current_price=signal_result["current_price"],
            signal_direction=signal_result["signal_direction"],
            signal_strength=signal_result["signal_strength"],
            confidence_level=signal_result["confidence_level"],
            recommended_action=signal_result["recommended_action"],
            technical_signal=signal_result["technical_signal"],
            smc_signal=signal_result["smc_signal"],
            signal_components=signal_result["signal_components"],
            risk_metrics=signal_result["risk_metrics"],
            entry_exit_levels=signal_result["entry_exit_levels"],
            market_context=signal_result["market_context"],
            signal_validation=signal_result["signal_validation"],
            key_message=signal_result["key_message"],
            execution_notes=signal_result["execution_notes"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trade signal generation failed: {str(e)}")


@router.get("/api/trade-signal/doc",
            summary="Trade Signal API Documentation",
            description="Get comprehensive documentation for the Trade Signal API response format")
async def get_trade_signal_documentation():
    """
    Get comprehensive documentation for the Trade Signal API response format.
    
    This endpoint provides detailed explanations of all response fields, enums, and trading concepts
    used in the comprehensive trade signal generation API.
    """
    return {
        "api_endpoint": "/api/trade-signal/{symbol}/{timeframe}",
        "description": "Comprehensive Trade Signal Generation combining Technical Analysis and Smart Money Concepts",
        "response_format": {
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol (e.g., BTCUSDT)",
                "example": "BTCUSDT"
            },
            "timeframe": {
                "type": "string",
                "description": "Analysis timeframe",
                "possible_values": ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w",
                                    "1M"],
                "example": "4h"
            },
            "timestamp": {
                "type": "string",
                "description": "Signal generation timestamp",
                "example": "2024-01-01T12:00:00Z"
            },
            "current_price": {
                "type": "float",
                "description": "Current market price at time of analysis",
                "example": 95000.0
            },
            "signal_direction": {
                "type": "SignalDirection",
                "description": "Primary trading signal direction",
                "possible_values": ["STRONG_BUY", "BUY", "WEAK_BUY", "NEUTRAL", "WEAK_SELL", "SELL", "STRONG_SELL"],
                "interpretation": {
                    "STRONG_BUY": "Very high confidence bullish signal, strong buy recommendation",
                    "BUY": "High confidence bullish signal, buy recommendation",
                    "WEAK_BUY": "Moderate confidence bullish signal, weak buy recommendation",
                    "NEUTRAL": "No clear directional bias, wait for better signals",
                    "WEAK_SELL": "Moderate confidence bearish signal, weak sell recommendation",
                    "SELL": "High confidence bearish signal, sell recommendation",
                    "STRONG_SELL": "Very high confidence bearish signal, strong sell recommendation"
                }
            },
            "signal_strength": {
                "type": "SignalStrength",
                "description": "Overall signal quality assessment",
                "possible_values": ["VERY_STRONG", "STRONG", "MODERATE", "WEAK", "VERY_WEAK"],
                "interpretation": {
                    "VERY_STRONG": "Exceptional signal quality, act with high confidence",
                    "STRONG": "High quality signal, good trading opportunity",
                    "MODERATE": "Decent signal quality, trade with caution",
                    "WEAK": "Low quality signal, avoid large positions",
                    "VERY_WEAK": "Very poor signal quality, avoid trading"
                }
            },
            "confidence_level": {
                "type": "ConfidenceLevel",
                "description": "Confidence in the generated signal",
                "possible_values": ["VERY_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW"],
                "interpretation": {
                    "VERY_HIGH": "90-100% confidence, execute with full conviction",
                    "HIGH": "75-89% confidence, good execution opportunity",
                    "MEDIUM": "50-74% confidence, proceed with caution",
                    "LOW": "25-49% confidence, avoid or use small position",
                    "VERY_LOW": "0-24% confidence, do not trade"
                }
            },
            "recommended_action": {
                "type": "TradingAction",
                "description": "Specific trading action recommendation",
                "possible_values": ["ENTER_LONG", "ENTER_SHORT", "HOLD_LONG", "HOLD_SHORT", "EXIT_LONG", "EXIT_SHORT",
                                    "WAIT"],
                "interpretation": {
                    "ENTER_LONG": "Initiate new long position",
                    "ENTER_SHORT": "Initiate new short position",
                    "HOLD_LONG": "Maintain existing long position",
                    "HOLD_SHORT": "Maintain existing short position",
                    "EXIT_LONG": "Close long position",
                    "EXIT_SHORT": "Close short position",
                    "WAIT": "No action recommended, wait for better opportunity"
                }
            },
            "technical_signal": {
                "type": "TechnicalSignal",
                "description": "Traditional technical analysis signals breakdown",
                "fields": {
                    "rsi_signal": {
                        "type": "float",
                        "description": "RSI signal contribution",
                        "range": "-2.0 to +2.0",
                        "interpretation": {
                            "+2.0": "Strong oversold condition, buy signal",
                            "+1.0": "Oversold condition, weak buy signal",
                            "0.0": "Neutral RSI condition",
                            "-1.0": "Overbought condition, weak sell signal",
                            "-2.0": "Strong overbought condition, sell signal"
                        }
                    },
                    "macd_signal": {
                        "type": "float",
                        "description": "MACD signal contribution",
                        "range": "-1.5 to +1.5",
                        "interpretation": {
                            "+1.5": "Strong bullish MACD signal",
                            "+1.0": "Bullish MACD signal",
                            "0.0": "Neutral MACD",
                            "-1.0": "Bearish MACD signal",
                            "-1.5": "Strong bearish MACD signal"
                        }
                    },
                    "ma_signal": {
                        "type": "float",
                        "description": "Moving average signal contribution",
                        "range": "-2.0 to +2.0",
                        "interpretation": {
                            "+2.0": "Price above both MAs, strong uptrend",
                            "+1.0": "Price above short MA, uptrend",
                            "0.0": "Neutral MA alignment",
                            "-1.0": "Price below short MA, downtrend",
                            "-2.0": "Price below both MAs, strong downtrend"
                        }
                    },
                    "volume_signal": {
                        "type": "float",
                        "description": "Volume signal contribution",
                        "range": "-1.0 to +1.0",
                        "interpretation": {
                            "+1.0": "High volume confirming move",
                            "0.0": "Normal volume",
                            "-1.0": "Low volume, weak conviction"
                        }
                    },
                    "volatility_signal": {
                        "type": "float",
                        "description": "Volatility signal contribution",
                        "range": "-0.5 to +0.5",
                        "interpretation": {
                            "+0.5": "Low volatility, favorable conditions",
                            "0.0": "Normal volatility",
                            "-0.5": "High volatility, unfavorable conditions"
                        }
                    },
                    "combined_score": {
                        "type": "float",
                        "description": "Weighted combination of all technical signals",
                        "calculation": "rsi_signal*0.25 + macd_signal*0.25 + ma_signal*0.25 + volume_signal*0.15 + volatility_signal*0.10"
                    }
                }
            },
            "smc_signal": {
                "type": "SMCSignal",
                "description": "Smart Money Concepts analysis signals breakdown",
                "fields": {
                    "market_structure_signal": {
                        "type": "float",
                        "description": "Market structure analysis signal",
                        "range": "-1.5 to +1.5",
                        "interpretation": {
                            "+1.5": "Strong bullish structure (HH, HL pattern)",
                            "0.0": "Neutral structure",
                            "-1.5": "Strong bearish structure (LH, LL pattern)"
                        }
                    },
                    "liquidity_signal": {
                        "type": "float",
                        "description": "Liquidity environment signal",
                        "range": "-0.5 to +1.0",
                        "interpretation": {
                            "+1.0": "High liquidity environment, favorable for large moves",
                            "0.0": "Normal liquidity",
                            "-0.5": "Low liquidity, unfavorable conditions"
                        }
                    },
                    "supply_demand_signal": {
                        "type": "float",
                        "description": "Supply and demand balance signal",
                        "range": "-1.0 to +1.0",
                        "interpretation": {
                            "+1.0": "Strong demand, bullish pressure",
                            "0.0": "Balanced supply/demand",
                            "-1.0": "Strong supply, bearish pressure"
                        }
                    },
                    "order_block_signal": {
                        "type": "float",
                        "description": "Order block presence signal",
                        "range": "0.0 to +0.5",
                        "interpretation": {
                            "+0.5": "Strong order block detected",
                            "0.0": "No significant order blocks"
                        }
                    },
                    "fair_value_gap_signal": {
                        "type": "float",
                        "description": "Fair value gap analysis signal",
                        "range": "-1.0 to +1.0",
                        "interpretation": {
                            "+1.0": "Bullish fair value gaps dominating",
                            "0.0": "No significant FVGs",
                            "-1.0": "Bearish fair value gaps dominating"
                        }
                    },
                    "combined_score": {
                        "type": "float",
                        "description": "Weighted combination of all SMC signals",
                        "calculation": "structure*0.25 + liquidity*0.20 + supply_demand*0.25 + order_block*0.15 + fvg*0.15"
                    }
                }
            },
            "signal_components": {
                "type": "array",
                "description": "Individual signal components with their contributions",
                "item_structure": {
                    "name": "Component name (RSI, MACD, Market Structure, etc.)",
                    "value": "Raw signal value",
                    "weight": "Weight in final signal calculation",
                    "contribution": "Actual contribution to final signal",
                    "status": "Component status description"
                }
            },
            "risk_metrics": {
                "type": "RiskMetrics",
                "description": "Comprehensive risk assessment for the trade",
                "fields": {
                    "risk_level": {
                        "type": "RiskLevel",
                        "possible_values": ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
                        "interpretation": {
                            "LOW": "Conservative trade setup, minimal risk",
                            "MEDIUM": "Moderate risk trade, standard management",
                            "HIGH": "High risk trade, careful management required",
                            "VERY_HIGH": "Extremely risky, avoid or use tiny position"
                        }
                    },
                    "volatility_risk": {
                        "type": "float",
                        "description": "Risk from price volatility",
                        "range": "0.0 to 1.0",
                        "interpretation": {
                            "0.0-0.3": "Low volatility risk",
                            "0.3-0.6": "Moderate volatility risk",
                            "0.6-0.8": "High volatility risk",
                            "0.8-1.0": "Very high volatility risk"
                        }
                    },
                    "liquidity_risk": {
                        "type": "float",
                        "description": "Risk from liquidity conditions",
                        "range": "0.0 to 1.0",
                        "calculation": "Based on volume relative to average"
                    },
                    "market_risk": {
                        "type": "float",
                        "description": "Risk from market conditions",
                        "range": "0.0 to 1.0",
                        "calculation": "Based on signal strength and market volatility"
                    },
                    "suggested_position_size": {
                        "type": "float",
                        "description": "Recommended position size as percentage of portfolio",
                        "range": "0.005 to 0.02",
                        "calculation": "Base 2% adjusted by risk factors"
                    },
                    "max_drawdown_estimate": {
                        "type": "float",
                        "description": "Estimated maximum drawdown percentage",
                        "calculation": "Based on historical volatility"
                    }
                }
            },
            "entry_exit_levels": {
                "type": "EntryExitLevels",
                "description": "Precise entry and exit price levels",
                "fields": {
                    "entry_price": {
                        "type": "float",
                        "description": "Recommended entry price level",
                        "note": "Usually current market price"
                    },
                    "stop_loss": {
                        "type": "float",
                        "description": "Stop loss level for risk management",
                        "calculation": "Entry ± (2 * ATR)"
                    },
                    "take_profit_1": {
                        "type": "float",
                        "description": "First take profit level",
                        "calculation": "Entry ± (1.5 * ATR)"
                    },
                    "take_profit_2": {
                        "type": "float",
                        "description": "Second take profit level",
                        "calculation": "Entry ± (3 * ATR)"
                    },
                    "take_profit_3": {
                        "type": "float",
                        "description": "Third take profit level (optional)",
                        "calculation": "Entry ± (4.5 * ATR)"
                    },
                    "risk_reward_ratio": {
                        "type": "float",
                        "description": "Risk to reward ratio for the trade",
                        "calculation": "(take_profit_1 - entry) / (entry - stop_loss)",
                        "interpretation": {
                            "< 1.0": "Poor risk/reward, avoid trade",
                            "1.0-2.0": "Acceptable risk/reward",
                            "2.0-3.0": "Good risk/reward",
                            "> 3.0": "Excellent risk/reward"
                        }
                    }
                }
            },
            "market_context": {
                "type": "MarketContext",
                "description": "Current market environment analysis",
                "fields": {
                    "current_condition": {
                        "type": "MarketCondition",
                        "possible_values": ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "BREAKOUT",
                                            "REVERSAL"],
                        "interpretation": {
                            "TRENDING_UP": "Clear upward trend, follow momentum",
                            "TRENDING_DOWN": "Clear downward trend, follow momentum",
                            "RANGING": "Sideways movement, use mean reversion",
                            "VOLATILE": "High volatility, adjust position sizes",
                            "BREAKOUT": "Breaking out of range, momentum opportunity",
                            "REVERSAL": "Trend reversal in progress"
                        }
                    },
                    "trend_direction": {
                        "type": "string",
                        "possible_values": ["BULLISH", "BEARISH", "NEUTRAL"],
                        "description": "Primary trend direction"
                    },
                    "trend_strength": {
                        "type": "float",
                        "description": "Strength of current trend",
                        "range": "0.0 to 1.0",
                        "interpretation": {
                            "0.8-1.0": "Very strong trend",
                            "0.6-0.8": "Strong trend",
                            "0.4-0.6": "Moderate trend",
                            "0.2-0.4": "Weak trend",
                            "0.0-0.2": "No clear trend"
                        }
                    },
                    "support_level": {
                        "type": "float",
                        "description": "Key support price level",
                        "calculation": "20-period rolling minimum"
                    },
                    "resistance_level": {
                        "type": "float",
                        "description": "Key resistance price level",
                        "calculation": "20-period rolling maximum"
                    },
                    "consolidation_range": {
                        "type": "object",
                        "description": "Price range if market is consolidating",
                        "fields": {
                            "low": "Lower bound of consolidation",
                            "high": "Upper bound of consolidation"
                        }
                    }
                }
            },
            "signal_validation": {
                "type": "SignalValidation",
                "description": "Signal confirmation and validation data",
                "fields": {
                    "signal_confirmed": {
                        "type": "boolean",
                        "description": "Whether the signal is confirmed by multiple factors",
                        "interpretation": {
                            "true": "Signal has multiple confirmations, higher reliability",
                            "false": "Signal lacks confirmation, lower reliability"
                        }
                    },
                    "confirmation_factors": {
                        "type": "array",
                        "description": "List of factors supporting the signal",
                        "examples": ["Technical and SMC signals aligned", "Strong trending market",
                                     "High volume confirmation"]
                    },
                    "warning_factors": {
                        "type": "array",
                        "description": "List of warning factors against the signal",
                        "examples": ["Technical and SMC signals diverging", "Weak or ranging market",
                                     "Low volume environment"]
                    },
                    "time_horizon": {
                        "type": "string",
                        "description": "Recommended time horizon for the signal",
                        "possible_values": ["1-3 days", "3-7 days", "1-2 weeks"],
                        "calculation": "Based on signal strength and market volatility"
                    },
                    "invalidation_level": {
                        "type": "float",
                        "description": "Price level that would invalidate the signal",
                        "usage": "If price reaches this level, exit the trade"
                    }
                }
            },
            "key_message": {
                "type": "string",
                "description": "Main takeaway message summarizing the signal",
                "examples": [
                    "Strong bullish signal detected with VERY_HIGH confidence",
                    "Strong bearish signal detected with HIGH confidence",
                    "Neutral market conditions, wait for clearer signals"
                ]
            },
            "execution_notes": {
                "type": "string",
                "description": "Detailed execution guidance and considerations",
                "examples": [
                    "Consider entering long position with proper risk management",
                    "Consider entering short position with proper risk management",
                    "Monitor for better entry opportunities"
                ]
            }
        },
        "signal_generation_methodology": {
            "description": "How the comprehensive trade signal is calculated",
            "steps": [
                "1. Fetch OHLCV data from Binance API",
                "2. Calculate technical indicators (RSI, MACD, MA, Volume, Volatility)",
                "3. Calculate Smart Money Concepts signals (Structure, Liquidity, Supply/Demand, Order Blocks, FVGs)",
                "4. Combine technical and SMC scores with equal weighting",
                "5. Determine signal direction and strength based on combined score",
                "6. Calculate risk metrics and position sizing",
                "7. Generate entry/exit levels using ATR-based calculations",
                "8. Analyze market context and validate signal",
                "9. Provide actionable recommendations"
            ],
            "weighting_scheme": {
                "technical_indicators": {
                    "rsi": "25%",
                    "macd": "25%",
                    "moving_averages": "25%",
                    "volume": "15%",
                    "volatility": "10%"
                },
                "smc_indicators": {
                    "market_structure": "25%",
                    "supply_demand": "25%",
                    "liquidity": "20%",
                    "order_blocks": "15%",
                    "fair_value_gaps": "15%"
                },
                "final_signal": {
                    "technical_score": "50%",
                    "smc_score": "50%"
                }
            }
        },
        "trading_concepts": {
            "signal_strength_mapping": {
                "description": "How combined scores map to signal directions",
                "score_ranges": {
                    "≥ 1.5": "STRONG_BUY (VERY_STRONG confidence)",
                    "1.0 to 1.5": "BUY (STRONG confidence)",
                    "0.5 to 1.0": "WEAK_BUY (MODERATE confidence)",
                    "-0.5 to 0.5": "NEUTRAL (WEAK confidence)",
                    "-1.0 to -0.5": "WEAK_SELL (MODERATE confidence)",
                    "-1.5 to -1.0": "SELL (STRONG confidence)",
                    "≤ -1.5": "STRONG_SELL (VERY_STRONG confidence)"
                }
            },
            "risk_management": {
                "description": "Built-in risk management principles",
                "position_sizing": "Base 2% of portfolio adjusted by risk factors",
                "stop_loss": "2x ATR from entry price",
                "take_profits": "1.5x, 3x, and 4.5x ATR from entry",
                "risk_factors": ["Volatility risk", "Liquidity risk", "Market risk"],
                "max_position": "2% of portfolio even for strongest signals"
            },
            "signal_validation": {
                "description": "Multi-factor signal validation process",
                "confirmation_criteria": [
                    "Technical and SMC alignment",
                    "Strong market trend",
                    "High volume confirmation",
                    "Multiple timeframe agreement"
                ],
                "warning_flags": [
                    "Diverging signals",
                    "Weak market conditions",
                    "Low volume environment",
                    "High volatility periods"
                ]
            },
            "smart_money_concepts": {
                "market_structure": "Analysis of swing highs/lows for trend determination",
                "liquidity_zones": "Areas where institutional orders are likely placed",
                "supply_demand": "Balance between buying and selling pressure",
                "order_blocks": "Price levels where large institutional orders were placed",
                "fair_value_gaps": "Price gaps that indicate institutional activity"
            }
        },
        "usage_examples": {
            "strong_bullish_signal": {
                "scenario": "STRONG_BUY signal with VERY_HIGH confidence",
                "signal_components": {
                    "technical_score": "+1.2 (RSI oversold, MACD bullish cross, price above MAs)",
                    "smc_score": "+1.3 (Bullish structure break, demand zone test, liquidity sweep)"
                },
                "combined_score": "+1.25",
                "recommended_action": "ENTER_LONG",
                "risk_management": "2% position size, 2x ATR stop loss",
                "execution": "Enter immediately with full position size"
            },
            "moderate_bearish_signal": {
                "scenario": "WEAK_SELL signal with MEDIUM confidence",
                "signal_components": {
                    "technical_score": "-0.6 (RSI bearish, MACD declining, volume weak)",
                    "smc_score": "-0.8 (Bearish structure, supply zone, order block resistance)"
                },
                "combined_score": "-0.7",
                "recommended_action": "HOLD_SHORT or small short position",
                "risk_management": "1% position size, wider stops",
                "execution": "Wait for better confirmation or use smaller size"
            },
            "neutral_market": {
                "scenario": "NEUTRAL signal with LOW confidence",
                "signal_components": {
                    "technical_score": "+0.1 (Mixed technical signals)",
                    "smc_score": "-0.2 (Ranging market structure)"
                },
                "combined_score": "-0.05",
                "recommended_action": "WAIT",
                "risk_management": "No position recommended",
                "execution": "Monitor for clearer directional signals"
            }
        },
        "integration_benefits": {
            "comprehensive_analysis": "Combines traditional TA with modern SMC concepts",
            "risk_awareness": "Built-in risk assessment and position sizing",
            "actionable_signals": "Clear buy/sell/wait recommendations",
            "level_precision": "Exact entry, stop, and target levels",
            "market_context": "Understanding of current market environment",
            "signal_validation": "Multiple confirmation factors",
            "time_horizon": "Appropriate holding period guidance"
        },
        "limitations_and_warnings": {
            "market_conditions": "Signals may be less reliable in highly volatile or news-driven markets",
            "timeframe_dependency": "Signal quality varies by timeframe - higher timeframes generally more reliable",
            "confirmation_importance": "Always wait for signal confirmation before executing trades",
            "risk_management": "Never risk more than suggested position size regardless of signal strength",
            "stop_loss_discipline": "Always use stop losses and honor them",
            "continuous_monitoring": "Monitor positions and adjust based on new signals"
        }
    }
