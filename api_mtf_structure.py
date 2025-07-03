from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field

from binance_data import BinanceDataFetcher
from smc_analysis import SMCAnalyzer
from utils import verify_api_key


class TrendDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"
    NEUTRAL = "NEUTRAL"


class StructureStatus(str, Enum):
    INTACT = "INTACT"
    QUESTIONING = "QUESTIONING"
    BROKEN = "BROKEN"


class MarketPhase(str, Enum):
    EXPANSION = "EXPANSION"
    PULLBACK = "PULLBACK"
    CONSOLIDATION = "CONSOLIDATION"
    ACCUMULATION = "ACCUMULATION"
    RETRACEMENT = "RETRACEMENT"


class BiasStrength(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL_BULLISH = "NEUTRAL_BULLISH"
    NEUTRAL = "NEUTRAL"
    NEUTRAL_BEARISH = "NEUTRAL_BEARISH"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class CurrentPosition(str, Enum):
    ABOVE_RANGE = "ABOVE_RANGE"
    UPPER_RANGE = "UPPER_RANGE"
    MIDDLE_RANGE = "MIDDLE_RANGE"
    AT_PIVOT = "AT_PIVOT"
    NEAR_PIVOT = "NEAR_PIVOT"
    NEAR_RESISTANCE = "NEAR_RESISTANCE"
    NEAR_SUPPORT = "NEAR_SUPPORT"
    LOWER_RANGE = "LOWER_RANGE"
    BELOW_RANGE = "BELOW_RANGE"
    MID_RANGE = "MID_RANGE"


class AlignmentStatus(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    CONFLICTED = "CONFLICTED"


class TradingApproach(str, Enum):
    AGGRESSIVE_TREND = "AGGRESSIVE_TREND"
    PATIENT_ACCUMULATION = "PATIENT_ACCUMULATION"
    COUNTER_TREND = "COUNTER_TREND"
    WAIT_FOR_SETUP = "WAIT_FOR_SETUP"
    NO_TRADE = "NO_TRADE"


class StructuralIntegrity(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class EntryQuality(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


class AlignmentType(str, Enum):
    CONFIRMED = "CONFIRMED"
    DIVERGING = "DIVERGING"
    TRANSITIONING = "TRANSITIONING"


class KeyLevels(BaseModel):
    major_resistance: Optional[float] = Field(None, description="Major resistance level")
    major_support: Optional[float] = Field(None, description="Major support level")
    resistance: Optional[float] = Field(None, description="Resistance level")
    support: Optional[float] = Field(None, description="Support level")
    pivot: Optional[float] = Field(None, description="Pivot level")
    range_high: Optional[float] = Field(None, description="Range high")
    range_low: Optional[float] = Field(None, description="Range low")
    current_position: CurrentPosition = Field(..., description="Current position relative to levels")


class TimeframeAnalysis(BaseModel):
    timeframe: str = Field(..., description="Timeframe identifier")
    trend: TrendDirection = Field(..., description="Current trend direction")
    structure_status: StructureStatus = Field(..., description="Structure status")
    last_major_move: str = Field(..., description="Last major structural move")
    key_levels: KeyLevels = Field(..., description="Key levels for this timeframe")
    strength_score: int = Field(..., description="Trend strength score 1-10", ge=1, le=10)
    bias: BiasStrength = Field(..., description="Trading bias")
    phase: MarketPhase = Field(..., description="Current market phase")
    confluence_with_monthly: Optional[bool] = Field(None, description="Confluence with monthly timeframe")
    notes: Optional[str] = Field(None, description="Additional notes")
    entry_quality: Optional[EntryQuality] = Field(None, description="Entry quality assessment")


class MTFAlignment(BaseModel):
    alignment_score: float = Field(..., description="Alignment score 0-10", ge=0, le=10)
    alignment_status: AlignmentStatus = Field(..., description="Overall alignment status")
    aligned_timeframes: List[str] = Field(..., description="Timeframes that are aligned")
    conflicting_timeframes: List[str] = Field(..., description="Timeframes showing conflict")
    dominant_bias: TrendDirection = Field(..., description="Dominant bias across timeframes")
    conflict_resolution: str = Field(..., description="How to resolve conflicts")
    trading_recommendation: str = Field(..., description="Overall trading recommendation")


class ConfluenceZone(BaseModel):
    zone: List[float] = Field(..., description="Price zone [low, high]")
    timeframes_present: List[str] = Field(..., description="Timeframes where level is present")
    type: str = Field(..., description="Type of level (SUPPORT/RESISTANCE)")
    strength: int = Field(..., description="Strength score 1-10", ge=1, le=10)
    description: str = Field(..., description="Description of confluence")


class SingleTFLevel(BaseModel):
    level: float = Field(..., description="Price level")
    timeframe: str = Field(..., description="Timeframe")
    importance: str = Field(..., description="Importance level")


class KeyLevelConfluence(BaseModel):
    major_confluence_zones: List[ConfluenceZone] = Field(..., description="Major confluence zones")
    single_tf_levels: List[SingleTFLevel] = Field(..., description="Single timeframe levels")


class StructuralIntegrityData(BaseModel):
    htf: StructuralIntegrity = Field(..., description="High timeframe integrity")
    mtf: StructuralIntegrity = Field(..., description="Medium timeframe integrity")
    ltf: StructuralIntegrity = Field(..., description="Low timeframe integrity")


class MTFStructureSummary(BaseModel):
    primary_trend: TrendDirection = Field(..., description="Primary trend direction")
    trading_timeframe_trend: TrendDirection = Field(..., description="Trading timeframe trend")
    entry_timeframe_trend: TrendDirection = Field(..., description="Entry timeframe trend")
    structural_integrity: StructuralIntegrityData = Field(..., description="Structural integrity assessment")
    best_trading_approach: TradingApproach = Field(..., description="Best trading approach")
    ideal_entry_scenario: str = Field(..., description="Ideal entry scenario description")


class CascadeAnalysisItem(BaseModel):
    alignment: AlignmentType = Field(..., description="Alignment status between timeframes")
    weekly_respecting_monthly: Optional[bool] = Field(None, description="Weekly respecting monthly")
    daily_respecting_weekly: Optional[bool] = Field(None, description="Daily respecting weekly")
    potential_shift: Optional[bool] = Field(None, description="Potential structure shift")
    watch_level: Optional[float] = Field(None, description="Level to watch for shift")
    accumulation_phase: Optional[bool] = Field(None, description="In accumulation phase")


class CascadeAnalysis(BaseModel):
    monthly_to_weekly: CascadeAnalysisItem = Field(..., description="Monthly to weekly cascade")
    weekly_to_daily: CascadeAnalysisItem = Field(..., description="Weekly to daily cascade")
    daily_to_4h: CascadeAnalysisItem = Field(..., description="Daily to 4H cascade")
    h4_to_1h: CascadeAnalysisItem = Field(..., description="4H to 1H cascade")


class TradingZone(BaseModel):
    range: List[float] = Field(..., description="Trading zone range [low, high]")
    timeframe_support: Optional[List[str]] = Field(None, description="Supporting timeframes")
    timeframe_resistance: Optional[List[str]] = Field(None, description="Resistance timeframes")
    risk_reward: str = Field(..., description="Risk/reward assessment")
    entry_criteria: str = Field(..., description="Entry criteria")
    reason: Optional[str] = Field(None, description="Reason for no-trade zone")


class TradingZones(BaseModel):
    optimal_buy_zone: Optional[TradingZone] = Field(None, description="Optimal buying zone")
    optimal_sell_zone: Optional[TradingZone] = Field(None, description="Optimal selling zone")
    no_trade_zone: Optional[TradingZone] = Field(None, description="No trade zone")


class MTFBiasMatrix(BaseModel):
    current_bias: str = Field(..., description="Current overall bias")
    confidence: int = Field(..., description="Confidence level 1-10", ge=1, le=10)
    key_message: str = Field(..., description="Key message summary")
    invalidation_scenarios: List[str] = Field(..., description="Scenarios that would invalidate bias")
    confirmation_scenarios: List[str] = Field(..., description="Scenarios that would confirm bias")


class TimeframeTransition(BaseModel):
    timeframe: str = Field(..., description="Timeframe identifier")
    time_to_close: str = Field(..., description="Time remaining to close")
    critical_level: float = Field(..., description="Critical level to watch")
    next_candle: Optional[str] = Field(None, description="Time to next candle")
    watch_for: Optional[str] = Field(None, description="What to watch for")


class TimeframeTransitions(BaseModel):
    next_htf_decision: TimeframeTransition = Field(..., description="Next high timeframe decision point")
    next_mtf_decision: TimeframeTransition = Field(..., description="Next medium timeframe decision point")
    immediate_focus: TimeframeTransition = Field(..., description="Immediate focus timeframe")


class MTFStructureResponse(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol")
    timestamp: str = Field(..., description="Analysis timestamp")
    current_price: float = Field(..., description="Current price")
    timeframe_analysis: Dict[str, TimeframeAnalysis] = Field(..., description="Individual timeframe analysis")
    mtf_alignment: MTFAlignment = Field(..., description="Multi-timeframe alignment analysis")
    key_level_confluence: KeyLevelConfluence = Field(..., description="Key level confluence analysis")
    mtf_structure_summary: MTFStructureSummary = Field(..., description="MTF structure summary")
    cascade_analysis: CascadeAnalysis = Field(..., description="Cascade analysis between timeframes")
    trading_zones: TradingZones = Field(..., description="Trading zone recommendations")
    mtf_bias_matrix: MTFBiasMatrix = Field(..., description="MTF bias matrix")
    timeframe_transitions: TimeframeTransitions = Field(..., description="Timeframe transition points")


# Create router
router = APIRouter(tags=["MTF Structure"])


class MTFStructureAnalyzer(SMCAnalyzer):
    """
    Multi-Timeframe Structure Analysis using smartmoneyconcepts library
    Inherits from SMCAnalyzer to reuse common functionality
    """

    def __init__(self):
        super().__init__()

    def calculate_distance_percentage(self, level: float, current_price: float) -> str:
        """
        Calculate percentage distance from current price
        """
        if current_price == 0:
            return "0.00%"

        percentage = ((level - current_price) / current_price) * 100
        sign = "+" if percentage >= 0 else ""
        return f"{sign}{percentage:.2f}%"

    def determine_current_position(self, current_price: float, key_levels: Dict) -> CurrentPosition:
        """
        Determine current position relative to key levels
        """
        if "major_resistance" in key_levels and "major_support" in key_levels:
            resistance = key_levels["major_resistance"]
            support = key_levels["major_support"]

            if resistance and support:
                range_size = resistance - support
                if range_size > 0:
                    position_ratio = (current_price - support) / range_size

                    if current_price > resistance:
                        return CurrentPosition.ABOVE_RANGE
                    elif current_price < support:
                        return CurrentPosition.BELOW_RANGE
                    elif position_ratio > 0.7:
                        return CurrentPosition.UPPER_RANGE
                    elif position_ratio < 0.3:
                        return CurrentPosition.LOWER_RANGE
                    else:
                        return CurrentPosition.MIDDLE_RANGE

        # Check for pivot levels
        if "pivot" in key_levels and key_levels["pivot"]:
            pivot = key_levels["pivot"]
            tolerance = current_price * 0.005  # 0.5% tolerance

            if abs(current_price - pivot) <= tolerance:
                return CurrentPosition.AT_PIVOT
            elif abs(current_price - pivot) <= tolerance * 2:
                return CurrentPosition.NEAR_PIVOT

        # Check for resistance/support proximity
        if "resistance" in key_levels and key_levels["resistance"]:
            resistance = key_levels["resistance"]
            tolerance = current_price * 0.01  # 1% tolerance
            if abs(current_price - resistance) <= tolerance:
                return CurrentPosition.NEAR_RESISTANCE

        if "support" in key_levels and key_levels["support"]:
            support = key_levels["support"]
            tolerance = current_price * 0.01  # 1% tolerance
            if abs(current_price - support) <= tolerance:
                return CurrentPosition.NEAR_SUPPORT

        return CurrentPosition.MID_RANGE

    def determine_trend_direction(self, df: pd.DataFrame, swing_data: Dict) -> TrendDirection:
        """
        Determine trend direction based on swing analysis
        """
        try:
            if not swing_data or not swing_data.get("swing_highs") or not swing_data.get("swing_lows"):
                return TrendDirection.RANGING

            highs = [point["price"] for point in swing_data["swing_highs"][-3:]]
            lows = [point["price"] for point in swing_data["swing_lows"][-3:]]

            if len(highs) >= 2 and len(lows) >= 2:
                # Check for higher highs and higher lows (bullish)
                higher_highs = all(highs[i] > highs[i - 1] for i in range(1, len(highs)))
                higher_lows = all(lows[i] > lows[i - 1] for i in range(1, len(lows)))

                # Check for lower highs and lower lows (bearish)
                lower_highs = all(highs[i] < highs[i - 1] for i in range(1, len(highs)))
                lower_lows = all(lows[i] < lows[i - 1] for i in range(1, len(lows)))

                if higher_highs and higher_lows:
                    return TrendDirection.BULLISH
                elif lower_highs and lower_lows:
                    return TrendDirection.BEARISH
                else:
                    return TrendDirection.RANGING

            return TrendDirection.RANGING

        except Exception:
            return TrendDirection.RANGING

    def determine_structure_status(self, trend: TrendDirection, bos_choch_data: Dict) -> StructureStatus:
        """
        Determine structure status based on BOS/CHoCH events
        """
        try:
            recent_events = []
            if bos_choch_data.get("bos"):
                recent_events.extend(bos_choch_data["bos"][-2:])
            if bos_choch_data.get("choch"):
                recent_events.extend(bos_choch_data["choch"][-2:])

            if not recent_events:
                return StructureStatus.INTACT

            # Check for recent CHoCH events (structure breaks)
            recent_choch = [event for event in recent_events if "choch" in str(event).lower()]
            if recent_choch:
                return StructureStatus.QUESTIONING

            # Check for BOS confirming trend
            recent_bos = [event for event in recent_events if "bos" in str(event).lower()]
            if recent_bos and trend != TrendDirection.RANGING:
                return StructureStatus.INTACT

            return StructureStatus.INTACT

        except Exception:
            return StructureStatus.INTACT

    def calculate_strength_score(self, trend: TrendDirection, structure_status: StructureStatus,
                                 volume_data: List[float], price_data: List[float]) -> int:
        """
        Calculate strength score based on various factors
        """
        try:
            score = 5  # Base score

            # Trend factor
            if trend == TrendDirection.BULLISH or trend == TrendDirection.BEARISH:
                score += 2
            elif trend == TrendDirection.RANGING:
                score -= 1

            # Structure factor
            if structure_status == StructureStatus.INTACT:
                score += 2
            elif structure_status == StructureStatus.QUESTIONING:
                score -= 1
            elif structure_status == StructureStatus.BROKEN:
                score -= 2

            # Volume factor
            if len(volume_data) > 10:
                recent_volume = np.mean(volume_data[-5:])
                avg_volume = np.mean(volume_data)
                if recent_volume > avg_volume * 1.2:
                    score += 1

            # Price momentum factor
            if len(price_data) > 5:
                recent_change = (price_data[-1] - price_data[-5]) / price_data[-5]
                if abs(recent_change) > 0.02:  # 2% move
                    score += 1

            return min(10, max(1, score))

        except Exception:
            return 5

    def determine_bias_strength(self, trend: TrendDirection, strength_score: int) -> BiasStrength:
        """
        Determine bias strength based on trend and strength score
        """
        if trend == TrendDirection.BULLISH:
            if strength_score >= 8:
                return BiasStrength.STRONG_BULLISH
            elif strength_score >= 6:
                return BiasStrength.BULLISH
            else:
                return BiasStrength.NEUTRAL_BULLISH
        elif trend == TrendDirection.BEARISH:
            if strength_score >= 8:
                return BiasStrength.STRONG_BEARISH
            elif strength_score >= 6:
                return BiasStrength.BEARISH
            else:
                return BiasStrength.NEUTRAL_BEARISH
        else:
            return BiasStrength.NEUTRAL

    def determine_market_phase(self, trend: TrendDirection, structure_status: StructureStatus,
                               strength_score: int) -> MarketPhase:
        """
        Determine current market phase
        """
        if trend == TrendDirection.RANGING:
            if structure_status == StructureStatus.BROKEN:
                return MarketPhase.ACCUMULATION
            else:
                return MarketPhase.CONSOLIDATION
        elif trend in [TrendDirection.BULLISH, TrendDirection.BEARISH]:
            if strength_score >= 7:
                return MarketPhase.EXPANSION
            elif structure_status == StructureStatus.QUESTIONING:
                return MarketPhase.PULLBACK
            else:
                return MarketPhase.RETRACEMENT
        else:
            return MarketPhase.CONSOLIDATION

    def get_last_major_move(self, bos_choch_data: Dict) -> str:
        """
        Get description of last major structural move
        """
        try:
            all_events = []
            if bos_choch_data.get("bos"):
                all_events.extend([("BOS", event) for event in bos_choch_data["bos"]])
            if bos_choch_data.get("choch"):
                all_events.extend([("CHoCH", event) for event in bos_choch_data["choch"]])

            if not all_events:
                return "No major moves detected"

            # Get the most recent event
            last_event_type, last_event = all_events[-1]

            # Determine direction (simplified)
            if "bullish" in str(last_event).lower() or "up" in str(last_event).lower():
                return f"BULLISH_{last_event_type}"
            elif "bearish" in str(last_event).lower() or "down" in str(last_event).lower():
                return f"BEARISH_{last_event_type}"
            else:
                return f"{last_event_type}_DETECTED"

        except Exception:
            return "Structure analysis pending"

    def analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str,
                                 current_price: float) -> TimeframeAnalysis:
        """
        Analyze a single timeframe
        """
        try:
            # Get swing data
            swing_data = self.calculate_swing_highs_lows(df, swing_length=50)
            swing_df = swing_data["swing_dataframe"]

            # Get BOS/CHoCH data
            bos_choch_data = self.calculate_bos_choch(df, swing_df, close_break=True)

            # Determine trend
            trend = self.determine_trend_direction(df, swing_data)

            # Determine structure status
            structure_status = self.determine_structure_status(trend, bos_choch_data)

            # Calculate strength score
            volume_data = df['volume'].tolist() if 'volume' in df.columns else [1] * len(df)
            price_data = df['close'].tolist()
            strength_score = self.calculate_strength_score(trend, structure_status, volume_data, price_data)

            # Determine bias and phase
            bias = self.determine_bias_strength(trend, strength_score)
            phase = self.determine_market_phase(trend, structure_status, strength_score)

            # Get last major move
            last_major_move = self.get_last_major_move(bos_choch_data)

            # Calculate key levels
            high = df['high'].max()
            low = df['low'].min()

            # Get recent support/resistance levels
            recent_highs = [point["price"] for point in swing_data["swing_highs"][-3:]] if swing_data[
                "swing_highs"] else []
            recent_lows = [point["price"] for point in swing_data["swing_lows"][-3:]] if swing_data[
                "swing_lows"] else []

            resistance = max(recent_highs) if recent_highs else high
            support = min(recent_lows) if recent_lows else low
            pivot = (resistance + support) / 2

            # Determine key levels structure based on timeframe
            if timeframe in ["1M"]:
                key_levels = KeyLevels(
                    major_resistance=resistance,
                    major_support=support,
                    current_position=self.determine_current_position(current_price, {
                        "major_resistance": resistance, "major_support": support
                    })
                )
            elif timeframe in ["1W"]:
                key_levels = KeyLevels(
                    resistance=resistance,
                    support=support,
                    pivot=pivot,
                    current_position=self.determine_current_position(current_price, {
                        "resistance": resistance, "support": support, "pivot": pivot
                    })
                )
            elif timeframe in ["1D"]:
                key_levels = KeyLevels(
                    resistance=resistance,
                    support=support,
                    pivot=pivot,
                    current_position=self.determine_current_position(current_price, {
                        "resistance": resistance, "support": support, "pivot": pivot
                    })
                )
            elif timeframe in ["4H"]:
                key_levels = KeyLevels(
                    range_high=high,
                    range_low=low,
                    current_position=self.determine_current_position(current_price, {
                        "range_high": high, "range_low": low
                    })
                )
            else:  # 1H and others
                key_levels = KeyLevels(
                    resistance=resistance,
                    support=support,
                    current_position=self.determine_current_position(current_price, {
                        "resistance": resistance, "support": support
                    })
                )

            # Additional fields based on timeframe
            confluence_with_monthly = None
            notes = None
            entry_quality = None

            if timeframe == "1W":
                confluence_with_monthly = True
            elif timeframe == "1D":
                if structure_status == StructureStatus.QUESTIONING:
                    notes = "Potential CHoCH, watching for confirmation"
            elif timeframe == "4H":
                if structure_status == StructureStatus.BROKEN:
                    notes = "Building cause after CHoCH"
            elif timeframe == "1H":
                entry_quality = EntryQuality.POOR if strength_score < 5 else EntryQuality.GOOD

            return TimeframeAnalysis(
                timeframe=timeframe,
                trend=trend,
                structure_status=structure_status,
                last_major_move=last_major_move,
                key_levels=key_levels,
                strength_score=strength_score,
                bias=bias,
                phase=phase,
                confluence_with_monthly=confluence_with_monthly,
                notes=notes,
                entry_quality=entry_quality
            )

        except Exception as e:
            print(f"Error analyzing {timeframe}: {e}")
            # Return default analysis
            return TimeframeAnalysis(
                timeframe=timeframe,
                trend=TrendDirection.RANGING,
                structure_status=StructureStatus.INTACT,
                last_major_move="Analysis pending",
                key_levels=KeyLevels(current_position=CurrentPosition.MID_RANGE),
                strength_score=5,
                bias=BiasStrength.NEUTRAL,
                phase=MarketPhase.CONSOLIDATION
            )

    def calculate_mtf_alignment(self, timeframe_analysis: Dict[str, TimeframeAnalysis]) -> MTFAlignment:
        """
        Calculate multi-timeframe alignment
        """
        try:
            # Get trend directions
            trends = {tf: analysis.trend for tf, analysis in timeframe_analysis.items()}

            # Count aligned timeframes
            bullish_tfs = [tf for tf, trend in trends.items() if trend == TrendDirection.BULLISH]
            bearish_tfs = [tf for tf, trend in trends.items() if trend == TrendDirection.BEARISH]
            ranging_tfs = [tf for tf, trend in trends.items() if trend == TrendDirection.RANGING]

            # Determine dominant bias
            if len(bullish_tfs) > len(bearish_tfs):
                dominant_bias = TrendDirection.BULLISH
                aligned_timeframes = bullish_tfs
                conflicting_timeframes = bearish_tfs + ranging_tfs
            elif len(bearish_tfs) > len(bullish_tfs):
                dominant_bias = TrendDirection.BEARISH
                aligned_timeframes = bearish_tfs
                conflicting_timeframes = bullish_tfs + ranging_tfs
            else:
                dominant_bias = TrendDirection.RANGING
                aligned_timeframes = ranging_tfs
                conflicting_timeframes = bullish_tfs + bearish_tfs

            # Calculate alignment score
            total_tfs = len(timeframe_analysis)
            aligned_count = len(aligned_timeframes)
            alignment_score = (aligned_count / total_tfs) * 10 if total_tfs > 0 else 5

            # Determine alignment status
            if alignment_score >= 8:
                alignment_status = AlignmentStatus.FULL
            elif alignment_score >= 6:
                alignment_status = AlignmentStatus.PARTIAL
            else:
                alignment_status = AlignmentStatus.CONFLICTED

            # Conflict resolution and trading recommendation
            if alignment_status == AlignmentStatus.FULL:
                conflict_resolution = "ALL_TIMEFRAMES_ALIGNED"
                trading_recommendation = "FOLLOW_TREND"
            elif alignment_status == AlignmentStatus.PARTIAL:
                conflict_resolution = "HIGHER_TF_PRIORITY"
                trading_recommendation = "WAIT_FOR_LTF_ALIGNMENT"
            else:
                conflict_resolution = "WAIT_FOR_CLARITY"
                trading_recommendation = "NO_TRADE"

            return MTFAlignment(
                alignment_score=alignment_score,
                alignment_status=alignment_status,
                aligned_timeframes=aligned_timeframes,
                conflicting_timeframes=conflicting_timeframes,
                dominant_bias=dominant_bias,
                conflict_resolution=conflict_resolution,
                trading_recommendation=trading_recommendation
            )

        except Exception as e:
            print(f"Error calculating MTF alignment: {e}")
            return MTFAlignment(
                alignment_score=5.0,
                alignment_status=AlignmentStatus.PARTIAL,
                aligned_timeframes=[],
                conflicting_timeframes=[],
                dominant_bias=TrendDirection.RANGING,
                conflict_resolution="ANALYSIS_PENDING",
                trading_recommendation="WAIT_FOR_SETUP"
            )

    def find_confluence_zones(self, timeframe_analysis: Dict[str, TimeframeAnalysis]) -> KeyLevelConfluence:
        """
        Find confluence zones across timeframes
        """
        try:
            confluence_zones = []
            single_tf_levels = []

            # Collect all levels from all timeframes
            all_levels = {}
            for tf, analysis in timeframe_analysis.items():
                levels = []
                kl = analysis.key_levels

                if kl.major_resistance:
                    levels.append(("RESISTANCE", kl.major_resistance))
                if kl.major_support:
                    levels.append(("SUPPORT", kl.major_support))
                if kl.resistance:
                    levels.append(("RESISTANCE", kl.resistance))
                if kl.support:
                    levels.append(("SUPPORT", kl.support))
                if kl.pivot:
                    levels.append(("PIVOT", kl.pivot))
                if kl.range_high:
                    levels.append(("RESISTANCE", kl.range_high))
                if kl.range_low:
                    levels.append(("SUPPORT", kl.range_low))

                all_levels[tf] = levels

            # Find confluences
            tolerance = 0.01  # 1% tolerance for confluence
            processed_levels = set()

            for tf1, levels1 in all_levels.items():
                for level_type1, level1 in levels1:
                    if level1 in processed_levels:
                        continue

                    confluence_tfs = [tf1]
                    confluence_type = level_type1

                    # Look for similar levels in other timeframes
                    for tf2, levels2 in all_levels.items():
                        if tf2 == tf1:
                            continue

                        for level_type2, level2 in levels2:
                            if abs(level1 - level2) / level1 <= tolerance:
                                confluence_tfs.append(tf2)
                                if level_type2 == "RESISTANCE" and confluence_type != "RESISTANCE":
                                    confluence_type = "RESISTANCE"

                    processed_levels.add(level1)

                    if len(confluence_tfs) >= 2:
                        # Calculate zone range
                        zone_range = [level1 * 0.999, level1 * 1.001]

                        confluence_zones.append(ConfluenceZone(
                            zone=zone_range,
                            timeframes_present=confluence_tfs,
                            type=confluence_type,
                            strength=min(10, len(confluence_tfs) * 3),
                            description=f"Multi-timeframe {confluence_type.lower()} confluence"
                        ))
                    else:
                        # Single timeframe level
                        importance = "HIGH" if tf1 in ["1M", "1W"] else "MEDIUM"
                        single_tf_levels.append(SingleTFLevel(
                            level=level1,
                            timeframe=tf1,
                            importance=importance
                        ))

            # Sort confluence zones by strength
            confluence_zones.sort(key=lambda x: x.strength, reverse=True)

            return KeyLevelConfluence(
                major_confluence_zones=confluence_zones[:5],  # Top 5
                single_tf_levels=single_tf_levels[:10]  # Top 10
            )

        except Exception as e:
            print(f"Error finding confluence zones: {e}")
            return KeyLevelConfluence(
                major_confluence_zones=[],
                single_tf_levels=[]
            )

    def create_mtf_structure_summary(self, timeframe_analysis: Dict[str, TimeframeAnalysis],
                                     mtf_alignment: MTFAlignment) -> MTFStructureSummary:
        """
        Create MTF structure summary
        """
        try:
            # Get trends from different timeframe categories
            htf_trends = []  # Monthly, Weekly
            mtf_trends = []  # Daily, 4H
            ltf_trends = []  # 1H, 15M

            for tf, analysis in timeframe_analysis.items():
                if tf in ["1M", "1W"]:
                    htf_trends.append(analysis.trend)
                elif tf in ["1D", "4H"]:
                    mtf_trends.append(analysis.trend)
                else:
                    ltf_trends.append(analysis.trend)

            # Determine primary trend (HTF dominates)
            primary_trend = mtf_alignment.dominant_bias
            if htf_trends:
                primary_trend = htf_trends[0]  # Use highest timeframe

            # Trading timeframe trend (4H typically)
            trading_timeframe_trend = TrendDirection.RANGING
            if "4H" in timeframe_analysis:
                trading_timeframe_trend = timeframe_analysis["4H"].trend
            elif mtf_trends:
                trading_timeframe_trend = mtf_trends[0]

            # Entry timeframe trend (1H typically)
            entry_timeframe_trend = TrendDirection.RANGING
            if "1H" in timeframe_analysis:
                entry_timeframe_trend = timeframe_analysis["1H"].trend
            elif ltf_trends:
                entry_timeframe_trend = ltf_trends[0]

            # Assess structural integrity
            htf_integrity = StructuralIntegrity.MODERATE
            mtf_integrity = StructuralIntegrity.MODERATE
            ltf_integrity = StructuralIntegrity.MODERATE

            # HTF integrity
            htf_scores = [analysis.strength_score for tf, analysis in timeframe_analysis.items() if tf in ["1M", "1W"]]
            if htf_scores:
                avg_htf_score = sum(htf_scores) / len(htf_scores)
                if avg_htf_score >= 7:
                    htf_integrity = StructuralIntegrity.STRONG
                elif avg_htf_score <= 4:
                    htf_integrity = StructuralIntegrity.WEAK

            # MTF integrity
            mtf_scores = [analysis.strength_score for tf, analysis in timeframe_analysis.items() if tf in ["1D", "4H"]]
            if mtf_scores:
                avg_mtf_score = sum(mtf_scores) / len(mtf_scores)
                if avg_mtf_score >= 7:
                    mtf_integrity = StructuralIntegrity.STRONG
                elif avg_mtf_score <= 4:
                    mtf_integrity = StructuralIntegrity.WEAK

            # LTF integrity
            ltf_scores = [analysis.strength_score for tf, analysis in timeframe_analysis.items() if tf in ["1H"]]
            if ltf_scores:
                avg_ltf_score = sum(ltf_scores) / len(ltf_scores)
                if avg_ltf_score >= 7:
                    ltf_integrity = StructuralIntegrity.STRONG
                elif avg_ltf_score <= 4:
                    ltf_integrity = StructuralIntegrity.WEAK

            # Determine best trading approach
            if mtf_alignment.alignment_status == AlignmentStatus.FULL:
                best_trading_approach = TradingApproach.AGGRESSIVE_TREND
            elif mtf_alignment.alignment_status == AlignmentStatus.PARTIAL:
                if htf_integrity == StructuralIntegrity.STRONG:
                    best_trading_approach = TradingApproach.PATIENT_ACCUMULATION
                else:
                    best_trading_approach = TradingApproach.WAIT_FOR_SETUP
            else:
                best_trading_approach = TradingApproach.NO_TRADE

            # Ideal entry scenario
            if trading_timeframe_trend == TrendDirection.RANGING:
                ideal_entry_scenario = "Wait for 4H bullish structure break"
            elif primary_trend == TrendDirection.BULLISH and entry_timeframe_trend == TrendDirection.BEARISH:
                ideal_entry_scenario = "Wait for 1H bullish CHoCH"
            else:
                ideal_entry_scenario = "Follow primary trend with pullback entry"

            return MTFStructureSummary(
                primary_trend=primary_trend,
                trading_timeframe_trend=trading_timeframe_trend,
                entry_timeframe_trend=entry_timeframe_trend,
                structural_integrity=StructuralIntegrityData(
                    htf=htf_integrity,
                    mtf=mtf_integrity,
                    ltf=ltf_integrity
                ),
                best_trading_approach=best_trading_approach,
                ideal_entry_scenario=ideal_entry_scenario
            )

        except Exception as e:
            print(f"Error creating MTF structure summary: {e}")
            return MTFStructureSummary(
                primary_trend=TrendDirection.RANGING,
                trading_timeframe_trend=TrendDirection.RANGING,
                entry_timeframe_trend=TrendDirection.RANGING,
                structural_integrity=StructuralIntegrityData(
                    htf=StructuralIntegrity.MODERATE,
                    mtf=StructuralIntegrity.MODERATE,
                    ltf=StructuralIntegrity.MODERATE
                ),
                best_trading_approach=TradingApproach.WAIT_FOR_SETUP,
                ideal_entry_scenario="Wait for clearer structure"
            )

    def analyze_cascade_effect(self, timeframe_analysis: Dict[str, TimeframeAnalysis]) -> CascadeAnalysis:
        """
        Analyze cascade effects between timeframes
        """
        try:
            # Monthly to Weekly
            monthly_to_weekly = CascadeAnalysisItem(alignment=AlignmentType.CONFIRMED)
            if "1M" in timeframe_analysis and "1W" in timeframe_analysis:
                monthly = timeframe_analysis["1M"]
                weekly = timeframe_analysis["1W"]

                if monthly.trend == weekly.trend:
                    monthly_to_weekly = CascadeAnalysisItem(
                        alignment=AlignmentType.CONFIRMED,
                        weekly_respecting_monthly=True
                    )
                else:
                    monthly_to_weekly = CascadeAnalysisItem(
                        alignment=AlignmentType.DIVERGING,
                        weekly_respecting_monthly=False
                    )

            # Weekly to Daily
            weekly_to_daily = CascadeAnalysisItem(alignment=AlignmentType.CONFIRMED)
            if "1W" in timeframe_analysis and "1D" in timeframe_analysis:
                weekly = timeframe_analysis["1W"]
                daily = timeframe_analysis["1D"]

                if weekly.trend == daily.trend:
                    weekly_to_daily = CascadeAnalysisItem(
                        alignment=AlignmentType.CONFIRMED,
                        daily_respecting_weekly=True
                    )
                else:
                    weekly_to_daily = CascadeAnalysisItem(
                        alignment=AlignmentType.DIVERGING,
                        daily_respecting_weekly=False
                    )

            # Daily to 4H
            daily_to_4h = CascadeAnalysisItem(alignment=AlignmentType.CONFIRMED)
            if "1D" in timeframe_analysis and "4H" in timeframe_analysis:
                daily = timeframe_analysis["1D"]
                h4 = timeframe_analysis["4H"]

                if daily.trend == h4.trend:
                    daily_to_4h = CascadeAnalysisItem(
                        alignment=AlignmentType.CONFIRMED,
                        potential_shift=False
                    )
                else:
                    daily_to_4h = CascadeAnalysisItem(
                        alignment=AlignmentType.DIVERGING,
                        potential_shift=True,
                        watch_level=h4.key_levels.range_high or h4.key_levels.resistance or 0
                    )

            # 4H to 1H
            h4_to_1h = CascadeAnalysisItem(alignment=AlignmentType.TRANSITIONING)
            if "4H" in timeframe_analysis and "1H" in timeframe_analysis:
                h4 = timeframe_analysis["4H"]
                h1 = timeframe_analysis["1H"]

                if h4.trend == TrendDirection.RANGING:
                    h4_to_1h = CascadeAnalysisItem(
                        alignment=AlignmentType.TRANSITIONING,
                        accumulation_phase=True
                    )
                elif h4.trend == h1.trend:
                    h4_to_1h = CascadeAnalysisItem(
                        alignment=AlignmentType.CONFIRMED,
                        accumulation_phase=False
                    )
                else:
                    h4_to_1h = CascadeAnalysisItem(
                        alignment=AlignmentType.DIVERGING,
                        accumulation_phase=False
                    )

            return CascadeAnalysis(
                monthly_to_weekly=monthly_to_weekly,
                weekly_to_daily=weekly_to_daily,
                daily_to_4h=daily_to_4h,
                h4_to_1h=h4_to_1h
            )

        except Exception as e:
            print(f"Error analyzing cascade effect: {e}")
            return CascadeAnalysis(
                monthly_to_weekly=CascadeAnalysisItem(alignment=AlignmentType.CONFIRMED),
                weekly_to_daily=CascadeAnalysisItem(alignment=AlignmentType.CONFIRMED),
                daily_to_4h=CascadeAnalysisItem(alignment=AlignmentType.CONFIRMED),
                h4_to_1h=CascadeAnalysisItem(alignment=AlignmentType.TRANSITIONING)
            )

    def create_trading_zones(self, timeframe_analysis: Dict[str, TimeframeAnalysis],
                             confluence: KeyLevelConfluence,
                             current_price: float) -> TradingZones:
        """
        Create trading zone recommendations
        """
        try:
            optimal_buy_zone = None
            optimal_sell_zone = None
            no_trade_zone = None

            # Find support and resistance confluence zones
            support_zones = [zone for zone in confluence.major_confluence_zones if zone.type == "SUPPORT"]
            resistance_zones = [zone for zone in confluence.major_confluence_zones if zone.type == "RESISTANCE"]

            # Create buy zone from strongest support below current price
            for zone in support_zones:
                zone_mid = sum(zone.zone) / 2
                if zone_mid < current_price:
                    optimal_buy_zone = TradingZone(
                        range=zone.zone,
                        timeframe_support=zone.timeframes_present,
                        risk_reward="EXCELLENT" if zone.strength >= 8 else "GOOD",
                        entry_criteria="4H bullish structure shift"
                    )
                    break

            # Create sell zone from strongest resistance above current price
            for zone in resistance_zones:
                zone_mid = sum(zone.zone) / 2
                if zone_mid > current_price:
                    optimal_sell_zone = TradingZone(
                        range=zone.zone,
                        timeframe_resistance=zone.timeframes_present,
                        risk_reward="GOOD" if zone.strength >= 7 else "FAIR",
                        entry_criteria="1H bearish structure at resistance"
                    )
                    break

            # Create no-trade zone around current price
            tolerance = current_price * 0.01  # 1%
            no_trade_zone = TradingZone(
                range=[current_price - tolerance, current_price + tolerance],
                risk_reward="POOR",
                entry_criteria="No clear edge",
                reason="Mid-range, no edge"
            )

            return TradingZones(
                optimal_buy_zone=optimal_buy_zone,
                optimal_sell_zone=optimal_sell_zone,
                no_trade_zone=no_trade_zone
            )

        except Exception as e:
            print(f"Error creating trading zones: {e}")
            return TradingZones()

    def create_mtf_bias_matrix(self, timeframe_analysis: Dict[str, TimeframeAnalysis],
                               mtf_alignment: MTFAlignment,
                               current_price: float) -> MTFBiasMatrix:
        """
        Create MTF bias matrix
        """
        try:
            # Determine current bias
            if mtf_alignment.alignment_status == AlignmentStatus.FULL:
                if mtf_alignment.dominant_bias == TrendDirection.BULLISH:
                    current_bias = "STRONG_BULLISH"
                elif mtf_alignment.dominant_bias == TrendDirection.BEARISH:
                    current_bias = "STRONG_BEARISH"
                else:
                    current_bias = "NEUTRAL"
            else:
                current_bias = "BULLISH_WITH_CAUTION"

            # Calculate confidence
            confidence = int(mtf_alignment.alignment_score)

            # Create key message
            htf_trend = "NEUTRAL"
            ltf_trend = "NEUTRAL"

            if "1W" in timeframe_analysis:
                htf_trend = timeframe_analysis["1W"].trend.value
            if "1H" in timeframe_analysis:
                ltf_trend = timeframe_analysis["1H"].trend.value

            key_message = f"HTF {htf_trend.lower()} but LTF showing weakness. Wait for 4H structure repair before long entries."

            # Create invalidation scenarios
            invalidation_scenarios = []
            if "1D" in timeframe_analysis and timeframe_analysis["1D"].key_levels.support:
                support = timeframe_analysis["1D"].key_levels.support
                invalidation_scenarios.append(f"Daily close below {support:.0f}")

            if "1W" in timeframe_analysis and timeframe_analysis["1W"].key_levels.support:
                support = timeframe_analysis["1W"].key_levels.support
                invalidation_scenarios.append(f"Weekly close below {support:.0f}")

            # Create confirmation scenarios
            confirmation_scenarios = []
            if "4H" in timeframe_analysis and timeframe_analysis["4H"].key_levels.range_high:
                resistance = timeframe_analysis["4H"].key_levels.range_high
                confirmation_scenarios.append(f"4H reclaims {resistance:.0f}")

            if "1H" in timeframe_analysis and timeframe_analysis["1H"].key_levels.support:
                support = timeframe_analysis["1H"].key_levels.support
                confirmation_scenarios.append(f"1H forms higher low above {support:.0f}")

            return MTFBiasMatrix(
                current_bias=current_bias,
                confidence=confidence,
                key_message=key_message,
                invalidation_scenarios=invalidation_scenarios,
                confirmation_scenarios=confirmation_scenarios
            )

        except Exception as e:
            print(f"Error creating MTF bias matrix: {e}")
            return MTFBiasMatrix(
                current_bias="NEUTRAL",
                confidence=5,
                key_message="Analysis in progress",
                invalidation_scenarios=[],
                confirmation_scenarios=[]
            )

    def create_timeframe_transitions(self, timeframe_analysis: Dict[str, TimeframeAnalysis]) -> TimeframeTransitions:
        """
        Create timeframe transition analysis
        """
        try:
            # Next HTF decision (Weekly)
            next_htf_decision = TimeframeTransition(
                timeframe="1W",
                time_to_close="3 days",
                critical_level=92000  # Default
            )

            if "1W" in timeframe_analysis:
                weekly = timeframe_analysis["1W"]
                critical_level = weekly.key_levels.support or weekly.key_levels.major_support or 92000
                next_htf_decision.critical_level = critical_level

            # Next MTF decision (Daily)
            next_mtf_decision = TimeframeTransition(
                timeframe="1D",
                time_to_close="10 hours",
                critical_level=94500  # Default
            )

            if "1D" in timeframe_analysis:
                daily = timeframe_analysis["1D"]
                critical_level = daily.key_levels.support or daily.key_levels.pivot or 94500
                next_mtf_decision.critical_level = critical_level

            # Immediate focus (4H)
            immediate_focus = TimeframeTransition(
                timeframe="4H",
                next_candle="2 hours",
                critical_level=95500,  # Default
                watch_for="Break above 95500"
            )

            if "4H" in timeframe_analysis:
                h4 = timeframe_analysis["4H"]
                critical_level = h4.key_levels.range_high or h4.key_levels.resistance or 95500
                immediate_focus.critical_level = critical_level
                immediate_focus.watch_for = f"Break above {critical_level:.0f}"

            return TimeframeTransitions(
                next_htf_decision=next_htf_decision,
                next_mtf_decision=next_mtf_decision,
                immediate_focus=immediate_focus
            )

        except Exception as e:
            print(f"Error creating timeframe transitions: {e}")
            return TimeframeTransitions(
                next_htf_decision=TimeframeTransition(
                    timeframe="1W",
                    time_to_close="3 days",
                    critical_level=92000
                ),
                next_mtf_decision=TimeframeTransition(
                    timeframe="1D",
                    time_to_close="10 hours",
                    critical_level=94500
                ),
                immediate_focus=TimeframeTransition(
                    timeframe="4H",
                    next_candle="2 hours",
                    critical_level=95500,
                    watch_for="Break above 95500"
                )
            )

    def analyze_mtf_structure(self, symbol: str, as_of_datetime: Optional[str] = None) -> Dict[str, Any]:
        """
        Main function to analyze multi-timeframe structure
        """
        current_price = 95000  # Will be updated with real data

        # Define timeframes to analyze
        timeframes = {
            "monthly": "1M",
            "weekly": "1W",
            "daily": "1D",
            "h4": "4H",
            "h1": "1H"
        }

        # Fetch data for each timeframe
        binance_fetcher = BinanceDataFetcher()
        timeframe_analysis = {}

        for tf_name, tf_code in timeframes.items():
            try:
                # Parse end time if provided
                end_time_ms = None
                if as_of_datetime:
                    as_of_dt = datetime.fromisoformat(as_of_datetime.replace('Z', '+00:00'))
                    end_time_ms = int(as_of_dt.timestamp() * 1000)

                # Fetch data
                import asyncio
                klines_data = asyncio.run(binance_fetcher.get_klines(
                    symbol=symbol,
                    interval=tf_code,
                    limit=200,
                    end_time=end_time_ms
                ))
                formatted_data = binance_fetcher.format_klines_data(klines_data)
                df = self.prepare_dataframe(formatted_data)
                current_price = float(formatted_data['close'][-1])

                # Analyze this timeframe
                timeframe_analysis[tf_name] = self.analyze_single_timeframe(df, tf_code, current_price)

            except Exception as e:
                print(f"Error fetching data for {tf_name}: {e}")
                # Create default analysis
                timeframe_analysis[tf_name] = TimeframeAnalysis(
                    timeframe=tf_code,
                    trend=TrendDirection.RANGING,
                    structure_status=StructureStatus.INTACT,
                    last_major_move="Analysis pending",
                    key_levels=KeyLevels(current_position=CurrentPosition.MID_RANGE),
                    strength_score=5,
                    bias=BiasStrength.NEUTRAL,
                    phase=MarketPhase.CONSOLIDATION
                )

        # Calculate MTF alignment
        mtf_alignment = self.calculate_mtf_alignment(timeframe_analysis)

        # Find confluence zones
        key_level_confluence = self.find_confluence_zones(timeframe_analysis)

        # Create MTF structure summary
        mtf_structure_summary = self.create_mtf_structure_summary(timeframe_analysis, mtf_alignment)

        # Analyze cascade effects
        cascade_analysis = self.analyze_cascade_effect(timeframe_analysis)

        # Create trading zones
        trading_zones = self.create_trading_zones(timeframe_analysis, key_level_confluence, current_price)

        # Create MTF bias matrix
        mtf_bias_matrix = self.create_mtf_bias_matrix(timeframe_analysis, mtf_alignment, current_price)

        # Create timeframe transitions
        timeframe_transitions = self.create_timeframe_transitions(timeframe_analysis)

        return {
            "current_price": current_price,
            "timeframe_analysis": timeframe_analysis,
            "mtf_alignment": mtf_alignment,
            "key_level_confluence": key_level_confluence,
            "mtf_structure_summary": mtf_structure_summary,
            "cascade_analysis": cascade_analysis,
            "trading_zones": trading_zones,
            "mtf_bias_matrix": mtf_bias_matrix,
            "timeframe_transitions": timeframe_transitions
        }


@router.get("/api/mtf-structure/{symbol}",
            response_model=MTFStructureResponse,
            summary="Get Multi-Timeframe Structure Analysis",
            description="Analyze cryptocurrency structure across multiple timeframes using Smart Money Concepts")
async def get_mtf_structure(
        symbol: str = Path(..., description="Trading pair symbol (e.g., BTCUSDT, BTC)"),
        as_of_datetime: Optional[str] = Query(None,
                                              description="ISO 8601 datetime string for historical analysis (e.g., 2024-01-01T00:00:00Z). Defaults to current time."),
        _: bool = Depends(verify_api_key)
):
    """
    Analyze cryptocurrency structure across multiple timeframes using Smart Money Concepts.
    
    This endpoint:
    1. Analyzes structure across 5 key timeframes: Monthly, Weekly, Daily, 4H, 1H
    2. Calculates multi-timeframe alignment and confluence zones
    3. Provides cascade analysis showing how trends flow between timeframes
    4. Identifies optimal trading zones and no-trade areas
    5. Creates comprehensive bias matrix with invalidation/confirmation levels
    6. Tracks timeframe transitions and key decision points
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., 'BTC', 'ETH', 'BTCUSDT')
    - **as_of_datetime**: Optional ISO 8601 datetime string for historical analysis
    
    **Timeframe Hierarchy:**
    - **High Timeframe (HTF)**: Monthly (1M), Weekly (1W) - Primary trend direction
    - **Medium Timeframe (MTF)**: Daily (1D), 4-Hour (4H) - Trading bias
    - **Low Timeframe (LTF)**: 1-Hour (1H) - Entry timing
    
    **Returns comprehensive MTF analysis including:**
    - Individual timeframe analysis with trend, structure status, and key levels
    - Multi-timeframe alignment scoring and conflict resolution
    - Key level confluence zones across timeframes
    - Cascade analysis showing trend flow from HTF to LTF
    - Trading zone recommendations with risk/reward assessment
    - MTF bias matrix with clear invalidation and confirmation scenarios
    - Timeframe transition tracking for upcoming decision points
    
    **Key Features:**
    - Automatic confluence detection across timeframes
    - Structural integrity assessment (Strong/Moderate/Weak)
    - Trading approach recommendations based on alignment
    - Clear conflict resolution when timeframes disagree
    - Next key timeframe decision points with critical levels
    
    **Note:** If the symbol doesn't end with 'USDT', it will be automatically appended.
    """
    try:
        # Initialize analyzer
        analyzer = MTFStructureAnalyzer()

        # Process symbol
        processed_symbol = symbol.upper()
        if not processed_symbol.endswith('USDT'):
            processed_symbol = f"{processed_symbol}USDT"

        # Perform MTF structure analysis
        analysis_result = analyzer.analyze_mtf_structure(processed_symbol, as_of_datetime)

        # Create response
        return MTFStructureResponse(
            symbol=processed_symbol,
            timestamp=datetime.now(timezone.utc).isoformat() if as_of_datetime is None else as_of_datetime,
            current_price=analysis_result["current_price"],
            timeframe_analysis=analysis_result["timeframe_analysis"],
            mtf_alignment=analysis_result["mtf_alignment"],
            key_level_confluence=analysis_result["key_level_confluence"],
            mtf_structure_summary=analysis_result["mtf_structure_summary"],
            cascade_analysis=analysis_result["cascade_analysis"],
            trading_zones=analysis_result["trading_zones"],
            mtf_bias_matrix=analysis_result["mtf_bias_matrix"],
            timeframe_transitions=analysis_result["timeframe_transitions"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MTF structure analysis failed: {str(e)}")


@router.get("/api/mtf-structure/doc",
            summary="MTF Structure API Documentation",
            description="Get comprehensive documentation for the Multi-Timeframe Structure API response format")
async def get_mtf_structure_documentation():
    """
    Get comprehensive documentation for the Multi-Timeframe Structure API response format.
    
    This endpoint provides detailed explanations of all response fields, enums, and trading concepts
    used in the MTF structure analysis API.
    """
    return {
        "api_endpoint": "/api/mtf-structure/{symbol}",
        "description": "Multi-Timeframe Structure Analysis using Smart Money Concepts across 5 key timeframes",
        "response_format": {
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol (e.g., BTCUSDT)",
                "example": "BTCUSDT"
            },
            "timestamp": {
                "type": "string",
                "description": "ISO 8601 timestamp of analysis",
                "example": "2024-01-01T12:00:00Z"
            },
            "current_price": {
                "type": "float",
                "description": "Current price at time of analysis",
                "example": 95000.0
            },
            "timeframe_analysis": {
                "type": "object",
                "description": "Individual analysis for each timeframe",
                "structure": {
                    "monthly/weekly/daily/h4/h1": {
                        "timeframe": {
                            "type": "string",
                            "description": "Timeframe identifier",
                            "possible_values": ["1M", "1W", "1D", "4H", "1H"]
                        },
                        "trend": {
                            "type": "TrendDirection",
                            "description": "Current trend direction",
                            "possible_values": ["BULLISH", "BEARISH", "RANGING", "NEUTRAL"],
                            "interpretation": {
                                "BULLISH": "Upward trend with higher highs and higher lows",
                                "BEARISH": "Downward trend with lower highs and lower lows",
                                "RANGING": "Sideways movement within defined range",
                                "NEUTRAL": "No clear directional bias"
                            }
                        },
                        "structure_status": {
                            "type": "StructureStatus",
                            "description": "Current structural integrity",
                            "possible_values": ["INTACT", "QUESTIONING", "BROKEN"],
                            "interpretation": {
                                "INTACT": "Structure remains valid, trend continuation likely",
                                "QUESTIONING": "Recent CHoCH events, structure under pressure",
                                "BROKEN": "Structure compromised, potential trend reversal"
                            }
                        },
                        "last_major_move": {
                            "type": "string",
                            "description": "Description of last significant structural move",
                            "examples": ["BULLISH_BOS", "BEARISH_CHoCH", "Structure analysis pending"]
                        },
                        "key_levels": {
                            "type": "KeyLevels",
                            "description": "Key price levels for this timeframe",
                            "fields": {
                                "major_resistance": "Major resistance level (HTF only)",
                                "major_support": "Major support level (HTF only)",
                                "resistance": "Standard resistance level",
                                "support": "Standard support level",
                                "pivot": "Pivot level between support/resistance",
                                "range_high": "Range high (LTF only)",
                                "range_low": "Range low (LTF only)",
                                "current_position": {
                                    "type": "CurrentPosition",
                                    "possible_values": ["ABOVE_RANGE", "UPPER_RANGE", "MIDDLE_RANGE", "LOWER_RANGE",
                                                        "BELOW_RANGE", "AT_PIVOT", "NEAR_PIVOT", "NEAR_RESISTANCE",
                                                        "NEAR_SUPPORT", "MID_RANGE"]
                                }
                            }
                        },
                        "strength_score": {
                            "type": "integer",
                            "description": "Trend strength score from 1-10",
                            "interpretation": {
                                "1-3": "Weak trend, high probability of reversal",
                                "4-6": "Moderate trend, neutral outlook",
                                "7-8": "Strong trend, continuation likely",
                                "9-10": "Very strong trend, follow momentum"
                            }
                        },
                        "bias": {
                            "type": "BiasStrength",
                            "description": "Trading bias strength",
                            "possible_values": ["STRONG_BULLISH", "BULLISH", "NEUTRAL_BULLISH", "NEUTRAL",
                                                "NEUTRAL_BEARISH", "BEARISH", "STRONG_BEARISH"]
                        },
                        "phase": {
                            "type": "MarketPhase",
                            "description": "Current market phase",
                            "possible_values": ["EXPANSION", "PULLBACK", "CONSOLIDATION", "ACCUMULATION",
                                                "RETRACEMENT"],
                            "interpretation": {
                                "EXPANSION": "Strong trending phase, follow momentum",
                                "PULLBACK": "Healthy retracement in trend",
                                "CONSOLIDATION": "Sideways movement, wait for breakout",
                                "ACCUMULATION": "Building phase after structure break",
                                "RETRACEMENT": "Counter-trend movement"
                            }
                        },
                        "confluence_with_monthly": "Boolean indicating confluence with monthly timeframe",
                        "notes": "Additional observations for this timeframe",
                        "entry_quality": {
                            "type": "EntryQuality",
                            "possible_values": ["EXCELLENT", "GOOD", "FAIR", "POOR"],
                            "description": "Quality assessment for entries on this timeframe"
                        }
                    }
                }
            },
            "mtf_alignment": {
                "type": "MTFAlignment",
                "description": "Multi-timeframe alignment analysis",
                "fields": {
                    "alignment_score": {
                        "type": "float",
                        "description": "Alignment score from 0-10",
                        "interpretation": {
                            "8-10": "Excellent alignment, high probability setups",
                            "6-7": "Good alignment, wait for LTF confirmation",
                            "4-5": "Moderate alignment, selective trading",
                            "0-3": "Poor alignment, avoid trading"
                        }
                    },
                    "alignment_status": {
                        "type": "AlignmentStatus",
                        "possible_values": ["FULL", "PARTIAL", "CONFLICTED"],
                        "interpretation": {
                            "FULL": "All timeframes aligned, high confidence trades",
                            "PARTIAL": "Some alignment, trade with caution",
                            "CONFLICTED": "Timeframes disagree, wait for clarity"
                        }
                    },
                    "aligned_timeframes": "List of timeframes showing same direction",
                    "conflicting_timeframes": "List of timeframes showing different direction",
                    "dominant_bias": "Overall bias across all timeframes",
                    "conflict_resolution": "How to resolve timeframe conflicts",
                    "trading_recommendation": "Overall trading recommendation"
                }
            },
            "key_level_confluence": {
                "type": "KeyLevelConfluence",
                "description": "Price levels where multiple timeframes converge",
                "fields": {
                    "major_confluence_zones": {
                        "type": "array",
                        "description": "Major zones where 2+ timeframes have levels",
                        "item_structure": {
                            "zone": "Price range [low, high]",
                            "timeframes_present": "List of timeframes with levels in this zone",
                            "type": "SUPPORT or RESISTANCE",
                            "strength": "Strength score 1-10 based on number of timeframes",
                            "description": "Description of the confluence"
                        }
                    },
                    "single_tf_levels": {
                        "type": "array",
                        "description": "Important levels from single timeframes",
                        "item_structure": {
                            "level": "Exact price level",
                            "timeframe": "Source timeframe",
                            "importance": "HIGH, MEDIUM, or LOW importance"
                        }
                    }
                }
            },
            "mtf_structure_summary": {
                "type": "MTFStructureSummary",
                "description": "High-level summary of MTF structure",
                "fields": {
                    "primary_trend": "Primary trend from highest timeframes",
                    "trading_timeframe_trend": "4H trend for trade bias",
                    "entry_timeframe_trend": "1H trend for entry timing",
                    "structural_integrity": {
                        "htf": "High timeframe structural integrity",
                        "mtf": "Medium timeframe structural integrity",
                        "ltf": "Low timeframe structural integrity",
                        "possible_values": ["STRONG", "MODERATE", "WEAK"]
                    },
                    "best_trading_approach": {
                        "type": "TradingApproach",
                        "possible_values": ["AGGRESSIVE_TREND", "PATIENT_ACCUMULATION", "COUNTER_TREND",
                                            "WAIT_FOR_SETUP", "NO_TRADE"],
                        "interpretation": {
                            "AGGRESSIVE_TREND": "Strong alignment, follow momentum aggressively",
                            "PATIENT_ACCUMULATION": "Build positions gradually",
                            "COUNTER_TREND": "Look for reversal opportunities",
                            "WAIT_FOR_SETUP": "Wait for better alignment",
                            "NO_TRADE": "Avoid trading until clarity emerges"
                        }
                    },
                    "ideal_entry_scenario": "Description of best entry setup"
                }
            },
            "cascade_analysis": {
                "type": "CascadeAnalysis",
                "description": "How trends flow from higher to lower timeframes",
                "fields": {
                    "monthly_to_weekly": "Monthly trend influence on weekly",
                    "weekly_to_daily": "Weekly trend influence on daily",
                    "daily_to_4h": "Daily trend influence on 4H",
                    "h4_to_1h": "4H trend influence on 1H",
                    "cascade_item_structure": {
                        "alignment": {
                            "type": "AlignmentType",
                            "possible_values": ["CONFIRMED", "DIVERGING", "TRANSITIONING"],
                            "interpretation": {
                                "CONFIRMED": "Lower TF respecting higher TF structure",
                                "DIVERGING": "Lower TF moving against higher TF",
                                "TRANSITIONING": "Lower TF in transition phase"
                            }
                        },
                        "weekly_respecting_monthly": "Boolean - weekly following monthly",
                        "daily_respecting_weekly": "Boolean - daily following weekly",
                        "potential_shift": "Boolean - potential structure shift",
                        "watch_level": "Key level to watch for shift",
                        "accumulation_phase": "Boolean - in accumulation phase"
                    }
                }
            },
            "trading_zones": {
                "type": "TradingZones",
                "description": "Recommended trading zones based on confluence",
                "fields": {
                    "optimal_buy_zone": {
                        "range": "Price range for optimal buying [low, high]",
                        "timeframe_support": "Timeframes supporting this zone",
                        "risk_reward": "Risk/reward assessment",
                        "entry_criteria": "Required conditions for entry"
                    },
                    "optimal_sell_zone": {
                        "range": "Price range for optimal selling [low, high]",
                        "timeframe_resistance": "Timeframes resisting at this zone",
                        "risk_reward": "Risk/reward assessment",
                        "entry_criteria": "Required conditions for entry"
                    },
                    "no_trade_zone": {
                        "range": "Price range to avoid trading [low, high]",
                        "reason": "Why this zone should be avoided",
                        "risk_reward": "Risk/reward assessment",
                        "entry_criteria": "Why no clear edge exists"
                    }
                }
            },
            "mtf_bias_matrix": {
                "type": "MTFBiasMatrix",
                "description": "Comprehensive bias matrix with scenarios",
                "fields": {
                    "current_bias": {
                        "type": "string",
                        "description": "Current overall bias",
                        "possible_values": ["STRONG_BULLISH", "BULLISH_WITH_CAUTION", "NEUTRAL", "BEARISH_WITH_CAUTION",
                                            "STRONG_BEARISH"]
                    },
                    "confidence": {
                        "type": "integer",
                        "description": "Confidence level 1-10",
                        "interpretation": {
                            "8-10": "High confidence, act on signals",
                            "6-7": "Medium confidence, wait for confirmation",
                            "4-5": "Low confidence, avoid large positions",
                            "1-3": "Very low confidence, stay flat"
                        }
                    },
                    "key_message": "Summary message of current market condition",
                    "invalidation_scenarios": "List of scenarios that would invalidate current bias",
                    "confirmation_scenarios": "List of scenarios that would confirm current bias"
                }
            },
            "timeframe_transitions": {
                "type": "TimeframeTransitions",
                "description": "Upcoming decision points across timeframes",
                "fields": {
                    "next_htf_decision": {
                        "timeframe": "High timeframe with next decision",
                        "time_to_close": "Time remaining to timeframe close",
                        "critical_level": "Key level to watch",
                        "next_candle": "Time to next candle",
                        "watch_for": "What to watch for"
                    },
                    "next_mtf_decision": "Next medium timeframe decision point",
                    "immediate_focus": "Immediate timeframe to focus on"
                }
            }
        },
        "trading_concepts": {
            "multi_timeframe_analysis": {
                "description": "Analysis across multiple timeframes to identify high-probability setups",
                "hierarchy": {
                    "HTF (High Timeframe)": "Monthly/Weekly - Determines primary trend direction",
                    "MTF (Medium Timeframe)": "Daily/4H - Provides trading bias",
                    "LTF (Low Timeframe)": "1H/15M - Provides entry timing"
                },
                "principle": "HTF gives direction, MTF gives bias, LTF gives entry"
            },
            "confluence_zones": {
                "description": "Price levels where multiple timeframes have support/resistance",
                "significance": "Higher confluence = stronger levels = better risk/reward",
                "usage": "Target these zones for entries and exits"
            },
            "cascade_analysis": {
                "description": "How trends flow from higher to lower timeframes",
                "principle": "Trends should cascade from HTF to LTF for best setups",
                "divergence_meaning": "When LTF goes against HTF, often signals reversal or pullback"
            },
            "structural_integrity": {
                "description": "Assessment of how well structure is holding",
                "strong": "Structure holding well, trend continuation likely",
                "moderate": "Some pressure but structure intact",
                "weak": "Structure under pressure, potential reversal"
            },
            "alignment_scoring": {
                "description": "Measures how well timeframes agree",
                "calculation": "Percentage of timeframes showing same direction",
                "usage": "Higher alignment = higher probability trades"
            }
        },
        "usage_examples": {
            "full_alignment_bullish": {
                "scenario": "All timeframes bullish, high alignment score",
                "approach": "AGGRESSIVE_TREND following",
                "entry_criteria": "Any pullback to support levels",
                "risk_management": "Tight stops, trend continuation expected"
            },
            "htf_bullish_ltf_bearish": {
                "scenario": "HTF bullish but LTF showing weakness",
                "approach": "PATIENT_ACCUMULATION",
                "entry_criteria": "Wait for LTF structure repair",
                "risk_management": "Wider stops, expect consolidation"
            },
            "conflicted_timeframes": {
                "scenario": "Timeframes showing different directions",
                "approach": "NO_TRADE or WAIT_FOR_SETUP",
                "entry_criteria": "Wait for alignment to improve",
                "risk_management": "Avoid trading until clarity emerges"
            }
        },
        "risk_management": {
            "high_alignment": "Smaller stops, larger position sizes",
            "medium_alignment": "Normal stops, normal position sizes",
            "low_alignment": "Wider stops, smaller position sizes",
            "conflicted": "Avoid trading or very small positions"
        }
    }
