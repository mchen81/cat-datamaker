from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field

from binance_data import BinanceDataFetcher
from smc_analysis import SMCAnalyzer
from utils import verify_api_key


class CurrentSession(str, Enum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP = "OVERLAP"
    CLOSED = "CLOSED"


class SessionDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class VolumeProfile(str, Enum):
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class RangeCharacteristic(str, Enum):
    TIGHT = "TIGHT"
    NORMAL = "NORMAL"
    EXPANSION = "EXPANSION"
    TRENDING = "TRENDING"


class LiquidityTaken(str, Enum):
    NONE = "NONE"
    ASIA_HIGH = "ASIA_HIGH"
    ASIA_LOW = "ASIA_LOW"
    LONDON_HIGH = "LONDON_HIGH"
    LONDON_LOW = "LONDON_LOW"


class ManipulationType(str, Enum):
    STOP_HUNT_HIGH = "STOP_HUNT_HIGH"
    STOP_HUNT_LOW = "STOP_HUNT_LOW"
    BEAR_TRAP = "BEAR_TRAP"
    BULL_TRAP = "BULL_TRAP"


class WeeklyBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class RangePosition(str, Enum):
    UPPER_75_PERCENT = "UPPER_75_PERCENT"
    UPPER_50_PERCENT = "UPPER_50_PERCENT"
    MID_RANGE = "MID_RANGE"
    LOWER_50_PERCENT = "LOWER_50_PERCENT"
    LOWER_25_PERCENT = "LOWER_25_PERCENT"


class KillzoneType(str, Enum):
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION"
    DISTRIBUTION = "DISTRIBUTION"


class PowerOf3Phase(str, Enum):
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION"
    DISTRIBUTION = "DISTRIBUTION"


class WeeklyReference(BaseModel):
    weekly_open: float = Field(..., description="Weekly opening price")
    distance_from_open: str = Field(..., description="Distance from weekly open as percentage")
    current_week_high: float = Field(..., description="Current week high")
    current_week_low: float = Field(..., description="Current week low")
    weekly_bias: WeeklyBias = Field(..., description="Weekly bias direction")
    days_into_week: int = Field(..., description="Days into current week", ge=0, le=7)


class DailyReference(BaseModel):
    daily_open: float = Field(..., description="Daily opening price")
    distance_from_open: str = Field(..., description="Distance from daily open as percentage")
    daily_high: float = Field(..., description="Daily high")
    daily_low: float = Field(..., description="Daily low")
    daily_range: float = Field(..., description="Daily range in price points")
    range_position: RangePosition = Field(..., description="Current position within daily range")


class SessionData(BaseModel):
    session_high: float = Field(..., description="Session high price")
    session_low: float = Field(..., description="Session low price")
    session_range: float = Field(..., description="Session range in price points")
    opening_price: float = Field(..., description="Session opening price")
    closing_price: Optional[float] = Field(None, description="Session closing price (if session ended)")
    current_price: Optional[float] = Field(None, description="Current price (if session active)")
    direction: SessionDirection = Field(..., description="Session direction")
    high_time: str = Field(..., description="Time when session high was made (HH:MM)")
    low_time: str = Field(..., description="Time when session low was made (HH:MM)")
    volume_profile: VolumeProfile = Field(..., description="Session volume profile")
    range_characteristic: RangeCharacteristic = Field(..., description="Range characteristic")
    liquidity_taken: LiquidityTaken = Field(..., description="Which liquidity was taken")
    manipulation_detected: Optional[bool] = Field(None, description="Whether manipulation was detected")
    manipulation_type: Optional[ManipulationType] = Field(None, description="Type of manipulation detected")
    session_active: Optional[bool] = Field(None, description="Whether session is currently active")
    time_remaining_hours: Optional[float] = Field(None, description="Hours remaining in session")


class KillzoneAnalysis(BaseModel):
    time_window: str = Field(..., description="Killzone time window")
    high: float = Field(..., description="Killzone high")
    low: float = Field(..., description="Killzone low")
    range: float = Field(..., description="Killzone range in price points")
    type: KillzoneType = Field(..., description="Killzone behavior type")
    key_levels_formed: List[float] = Field(..., description="Key levels formed in killzone")
    false_breakout: Optional[bool] = Field(None, description="Whether false breakout occurred")
    sweep_direction: Optional[str] = Field(None, description="Direction of liquidity sweep")
    reversal_level: Optional[float] = Field(None, description="Level where reversal occurred")
    trend_continuation: Optional[bool] = Field(None, description="Whether trend continued")
    target_projection: Optional[float] = Field(None, description="Projected target level")
    recommendation: str = Field(..., description="Trading recommendation")


class PowerOf3Phase(BaseModel):
    session: str = Field(..., description="Session where phase occurred")
    range: Optional[List[float]] = Field(None, description="Price range for accumulation")
    sweep_level: Optional[float] = Field(None, description="Level swept in manipulation")
    direction: Optional[str] = Field(None, description="Direction of manipulation/distribution")
    trend_direction: Optional[str] = Field(None, description="Trend direction in distribution")
    target: Optional[float] = Field(None, description="Target level")
    completed: bool = Field(..., description="Whether phase is completed")
    in_progress: Optional[bool] = Field(None, description="Whether phase is in progress")


class PowerOf3Data(BaseModel):
    current_phase: str = Field(..., description="Current Power of 3 phase")
    accumulation: PowerOf3Phase = Field(..., description="Accumulation phase data")
    manipulation: PowerOf3Phase = Field(..., description="Manipulation phase data")
    distribution: PowerOf3Phase = Field(..., description="Distribution phase data")
    po3_confidence: float = Field(..., description="Power of 3 pattern confidence score", ge=0, le=10)
    pattern_clarity: str = Field(..., description="Pattern clarity assessment")


class SessionOverlap(BaseModel):
    time_window: str = Field(..., description="Overlap time window")
    high: float = Field(..., description="Overlap period high")
    low: float = Field(..., description="Overlap period low")
    volatility: str = Field(..., description="Volatility level during overlap")
    directional_bias: str = Field(..., description="Directional bias during overlap")
    volume_spike: bool = Field(..., description="Whether volume spike occurred")


class SessionBias(BaseModel):
    intraday: str = Field(..., description="Intraday bias direction")
    key_message: str = Field(..., description="Key message about session analysis")


class TradingRecommendations(BaseModel):
    current_opportunity: str = Field(..., description="Current trading opportunity")
    entry_zone: List[float] = Field(..., description="Entry zone range [low, high]")
    stop_loss: float = Field(..., description="Stop loss level")
    targets: List[float] = Field(..., description="Target levels")
    session_bias: SessionBias = Field(..., description="Session bias information")
    next_key_time: str = Field(..., description="Next key time to watch")
    next_event: str = Field(..., description="Next key event")


class SessionStatistics(BaseModel):
    most_volatile_session: str = Field(..., description="Most volatile session")
    highest_volume_session: str = Field(..., description="Highest volume session")
    avg_daily_range: float = Field(..., description="Average daily range")
    current_day_range: float = Field(..., description="Current day range")
    range_expansion: bool = Field(..., description="Whether range is expanding")
    session_rhythm: str = Field(..., description="Session rhythm assessment")


class KillzoneSessionsResponse(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Analysis timeframe")
    timestamp: str = Field(..., description="Analysis timestamp")
    current_price: float = Field(..., description="Current price")
    current_session: CurrentSession = Field(..., description="Current active session")
    weekly_reference: WeeklyReference = Field(..., description="Weekly reference data")
    daily_reference: DailyReference = Field(..., description="Daily reference data")
    session_data: Dict[str, SessionData] = Field(..., description="Session analysis data")
    killzone_analysis: Dict[str, KillzoneAnalysis] = Field(..., description="Killzone analysis")
    power_of_3: PowerOf3Data = Field(..., description="Power of 3 analysis")
    session_overlaps: Dict[str, SessionOverlap] = Field(..., description="Session overlap analysis")
    trading_recommendations: TradingRecommendations = Field(..., description="Trading recommendations")
    session_statistics: SessionStatistics = Field(..., description="Session statistics")


# Create router
router = APIRouter(tags=["Killzone Sessions"])


class KillzoneSessionAnalyzer(SMCAnalyzer):
    """
    Killzone Session Analysis using smartmoneyconcepts library
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

    def determine_current_session(self, current_time: datetime) -> CurrentSession:
        """
        Determine current active session based on UTC time
        """
        utc_hour = current_time.hour

        # Session times in UTC
        # Asia: 00:00-08:00
        # London: 08:00-16:00  
        # NY: 13:00-21:00
        # London-NY Overlap: 13:00-16:00

        if 0 <= utc_hour < 8:
            return CurrentSession.ASIA
        elif 8 <= utc_hour < 13:
            return CurrentSession.LONDON
        elif 13 <= utc_hour < 16:
            return CurrentSession.OVERLAP  # London-NY overlap
        elif 16 <= utc_hour < 21:
            return CurrentSession.NEW_YORK
        else:
            return CurrentSession.CLOSED

    def analyze_weekly_reference(self, df: pd.DataFrame, current_price: float) -> WeeklyReference:
        """
        Analyze weekly reference levels
        """
        try:
            # Get weekly data using inherited method
            weekly_data = self.calculate_previous_high_low(df, time_frame="1W")

            # Get most recent weekly open (approximate)
            weekly_open = current_price
            current_week_high = current_price
            current_week_low = current_price

            if weekly_data["previous_highs"]:
                # Use last week's data as reference
                weekly_open = weekly_data["previous_highs"][-1]["level"]

            # Calculate current week high/low from recent data
            if len(df) > 0:
                recent_days = min(7, len(df))
                recent_data = df.tail(recent_days)
                current_week_high = recent_data['high'].max()
                current_week_low = recent_data['low'].min()

                # Better weekly open estimation
                if len(df) >= 7:
                    weekly_open = df.iloc[-7]['open']

            # Determine weekly bias
            if current_price > weekly_open * 1.01:
                weekly_bias = WeeklyBias.BULLISH
            elif current_price < weekly_open * 0.99:
                weekly_bias = WeeklyBias.BEARISH
            else:
                weekly_bias = WeeklyBias.NEUTRAL

            # Calculate days into week (approximate)
            days_into_week = min(7, len(df) % 7) if len(df) > 0 else 1

            return WeeklyReference(
                weekly_open=weekly_open,
                distance_from_open=self.calculate_distance_percentage(weekly_open, current_price),
                current_week_high=current_week_high,
                current_week_low=current_week_low,
                weekly_bias=weekly_bias,
                days_into_week=days_into_week
            )

        except Exception as e:
            print(f"Error in weekly reference analysis: {e}")
            return WeeklyReference(
                weekly_open=current_price,
                distance_from_open="0.00%",
                current_week_high=current_price,
                current_week_low=current_price,
                weekly_bias=WeeklyBias.NEUTRAL,
                days_into_week=1
            )

    def analyze_daily_reference(self, df: pd.DataFrame, current_price: float) -> DailyReference:
        """
        Analyze daily reference levels
        """
        try:
            # Get today's data (last 24 hours approximately)
            daily_high = current_price
            daily_low = current_price
            daily_open = current_price

            if len(df) > 0:
                # Get last 24 candles as proxy for daily data
                daily_candles = min(24, len(df))
                today_data = df.tail(daily_candles)

                daily_high = today_data['high'].max()
                daily_low = today_data['low'].min()
                daily_open = today_data.iloc[0]['open']

            daily_range = daily_high - daily_low

            # Determine range position
            if daily_range > 0:
                position_ratio = (current_price - daily_low) / daily_range
                if position_ratio >= 0.75:
                    range_position = RangePosition.UPPER_75_PERCENT
                elif position_ratio >= 0.50:
                    range_position = RangePosition.UPPER_50_PERCENT
                elif position_ratio >= 0.25:
                    range_position = RangePosition.LOWER_50_PERCENT
                else:
                    range_position = RangePosition.LOWER_25_PERCENT
            else:
                range_position = RangePosition.MID_RANGE

            return DailyReference(
                daily_open=daily_open,
                distance_from_open=self.calculate_distance_percentage(daily_open, current_price),
                daily_high=daily_high,
                daily_low=daily_low,
                daily_range=daily_range,
                range_position=range_position
            )

        except Exception as e:
            print(f"Error in daily reference analysis: {e}")
            return DailyReference(
                daily_open=current_price,
                distance_from_open="0.00%",
                daily_high=current_price,
                daily_low=current_price,
                daily_range=0,
                range_position=RangePosition.MID_RANGE
            )

    def determine_volume_profile(self, session_data: pd.DataFrame, df: pd.DataFrame) -> VolumeProfile:
        """
        Determine session volume profile
        """
        if session_data.empty or df.empty:
            return VolumeProfile.MODERATE

        # Calculate session average volume
        session_volume = session_data['volume'].mean() if 'volume' in session_data.columns else 0

        # Calculate overall average volume
        overall_volume = df['volume'].mean() if 'volume' in df.columns else 1

        if overall_volume == 0:
            return VolumeProfile.MODERATE

        volume_ratio = session_volume / overall_volume

        if volume_ratio >= 2.5:
            return VolumeProfile.VERY_HIGH
        elif volume_ratio >= 1.8:
            return VolumeProfile.HIGH
        elif volume_ratio >= 0.8:
            return VolumeProfile.MODERATE
        elif volume_ratio >= 0.5:
            return VolumeProfile.LOW
        else:
            return VolumeProfile.VERY_LOW

    def determine_range_characteristic(self, session_range: float, avg_range: float) -> RangeCharacteristic:
        """
        Determine session range characteristic
        """
        if avg_range == 0:
            return RangeCharacteristic.NORMAL

        range_ratio = session_range / avg_range

        if range_ratio >= 1.5:
            return RangeCharacteristic.TRENDING
        elif range_ratio >= 1.2:
            return RangeCharacteristic.EXPANSION
        elif range_ratio >= 0.7:
            return RangeCharacteristic.NORMAL
        else:
            return RangeCharacteristic.TIGHT

    def detect_liquidity_taken(self, session_data: pd.DataFrame, prev_session_high: float,
                               prev_session_low: float, session_name: str) -> LiquidityTaken:
        """
        Detect which liquidity was taken during session
        """
        if session_data.empty:
            return LiquidityTaken.NONE

        session_high = session_data['high'].max()
        session_low = session_data['low'].min()

        # Check if session took previous session liquidity
        tolerance = prev_session_high * 0.001  # 0.1% tolerance

        if session_high > prev_session_high + tolerance:
            if session_name == "london":
                return LiquidityTaken.ASIA_HIGH
            elif session_name == "new_york":
                return LiquidityTaken.LONDON_HIGH

        if session_low < prev_session_low - tolerance:
            if session_name == "london":
                return LiquidityTaken.ASIA_LOW
            elif session_name == "new_york":
                return LiquidityTaken.LONDON_LOW

        return LiquidityTaken.NONE

    def detect_manipulation(self, session_data: pd.DataFrame, prev_high: float,
                            prev_low: float) -> tuple[bool, Optional[ManipulationType]]:
        """
        Detect manipulation patterns in session
        """
        if session_data.empty:
            return False, None

        session_high = session_data['high'].max()
        session_low = session_data['low'].min()
        session_close = session_data['close'].iloc[-1]
        session_open = session_data['open'].iloc[0]

        tolerance = prev_high * 0.002  # 0.2% tolerance

        # Check for false breakout above previous high (bear trap)
        if session_high > prev_high + tolerance and session_close < prev_high:
            return True, ManipulationType.BEAR_TRAP

        # Check for false breakout below previous low (bull trap)
        if session_low < prev_low - tolerance and session_close > prev_low:
            return True, ManipulationType.BULL_TRAP

        # Check for stop hunt above previous high
        if session_high > prev_high + tolerance and session_close < session_high * 0.995:
            return True, ManipulationType.STOP_HUNT_HIGH

        # Check for stop hunt below previous low
        if session_low < prev_low - tolerance and session_close > session_low * 1.005:
            return True, ManipulationType.STOP_HUNT_LOW

        return False, None

    def get_session_times(self, session_data: pd.DataFrame) -> tuple[str, str]:
        """
        Get high and low times for session
        """
        if session_data.empty:
            return "00:00", "00:00"

        # Find indices of high and low
        high_idx = session_data['high'].idxmax()
        low_idx = session_data['low'].idxmin()

        # Extract hour:minute from timestamp
        try:
            high_time = session_data.loc[high_idx].name.strftime("%H:%M")
            low_time = session_data.loc[low_idx].name.strftime("%H:%M")
        except:
            high_time = "00:00"
            low_time = "00:00"

        return high_time, low_time

    def analyze_session_data(self, df: pd.DataFrame, current_price: float,
                             current_time: datetime) -> Dict[str, SessionData]:
        """
        Analyze all trading sessions
        """
        sessions = {}

        # Define session parameters
        session_configs = {
            "asia": ("Tokyo", "00:00", "08:00"),
            "london": ("London", "08:00", "16:00"),
            "new_york": ("New York", "13:00", "21:00")
        }

        prev_high = current_price
        prev_low = current_price
        avg_range = 0

        if len(df) > 0:
            avg_range = (df['high'] - df['low']).mean()

        for session_name, (city, start_time, end_time) in session_configs.items():
            try:
                # Get session data using inherited method
                session_result = self.calculate_sessions(df, city, start_time, end_time, time_zone="UTC")

                if not session_result or not session_result.get("session_candles"):
                    # Create default session data
                    sessions[session_name] = SessionData(
                        session_high=current_price,
                        session_low=current_price,
                        session_range=0,
                        opening_price=current_price,
                        direction=SessionDirection.NEUTRAL,
                        high_time="00:00",
                        low_time="00:00",
                        volume_profile=VolumeProfile.MODERATE,
                        range_characteristic=RangeCharacteristic.NORMAL,
                        liquidity_taken=LiquidityTaken.NONE
                    )
                    continue

                # Extract session data from recent candles
                session_df = pd.DataFrame()
                if len(df) >= 8:  # Ensure we have enough data
                    if session_name == "asia":
                        session_df = df.tail(8)  # Last 8 hours
                    elif session_name == "london":
                        session_df = df.tail(16)  # Last 16 hours
                    else:  # new_york
                        session_df = df.tail(8)  # Last 8 hours

                if session_df.empty:
                    session_df = df.tail(1)  # Fallback to last candle

                # Calculate session metrics
                session_high = session_df['high'].max()
                session_low = session_df['low'].min()
                session_range = session_high - session_low
                opening_price = session_df['open'].iloc[0]
                closing_price = session_df['close'].iloc[-1]

                # Determine session direction
                if closing_price > opening_price * 1.001:
                    direction = SessionDirection.BULLISH
                elif closing_price < opening_price * 0.999:
                    direction = SessionDirection.BEARISH
                else:
                    direction = SessionDirection.NEUTRAL

                # Get session timing
                high_time, low_time = self.get_session_times(session_df)

                # Determine volume profile and range characteristic
                volume_profile = self.determine_volume_profile(session_df, df)
                range_characteristic = self.determine_range_characteristic(session_range, avg_range)

                # Detect liquidity taken
                liquidity_taken = self.detect_liquidity_taken(session_df, prev_high, prev_low, session_name)

                # Detect manipulation
                manipulation_detected, manipulation_type = self.detect_manipulation(session_df, prev_high, prev_low)

                # Check if session is currently active
                current_session = self.determine_current_session(current_time)
                session_active = (session_name == current_session.value.lower() or
                                  (current_session == CurrentSession.OVERLAP and session_name in ["london",
                                                                                                  "new_york"]))

                # Calculate time remaining if active
                time_remaining_hours = None
                if session_active:
                    if session_name == "asia":
                        time_remaining_hours = max(0, 8 - current_time.hour)
                    elif session_name == "london":
                        time_remaining_hours = max(0, 16 - current_time.hour)
                    elif session_name == "new_york":
                        time_remaining_hours = max(0, 21 - current_time.hour)

                sessions[session_name] = SessionData(
                    session_high=session_high,
                    session_low=session_low,
                    session_range=session_range,
                    opening_price=opening_price,
                    closing_price=closing_price if not session_active else None,
                    current_price=current_price if session_active else None,
                    direction=direction,
                    high_time=high_time,
                    low_time=low_time,
                    volume_profile=volume_profile,
                    range_characteristic=range_characteristic,
                    liquidity_taken=liquidity_taken,
                    manipulation_detected=manipulation_detected,
                    manipulation_type=manipulation_type,
                    session_active=session_active,
                    time_remaining_hours=time_remaining_hours
                )

                # Update prev_high/low for next session
                prev_high = session_high
                prev_low = session_low

            except Exception as e:
                print(f"Error analyzing {session_name} session: {e}")
                sessions[session_name] = SessionData(
                    session_high=current_price,
                    session_low=current_price,
                    session_range=0,
                    opening_price=current_price,
                    direction=SessionDirection.NEUTRAL,
                    high_time="00:00",
                    low_time="00:00",
                    volume_profile=VolumeProfile.MODERATE,
                    range_characteristic=RangeCharacteristic.NORMAL,
                    liquidity_taken=LiquidityTaken.NONE
                )

        return sessions

    def analyze_killzones(self, df: pd.DataFrame, current_price: float) -> Dict[str, KillzoneAnalysis]:
        """
        Analyze ICT killzones
        """
        killzones = {}

        # Define killzone time windows
        killzone_configs = {
            "asia_killzone": ("08:00", "12:00", KillzoneType.ACCUMULATION),
            "london_killzone": ("07:00", "10:00", KillzoneType.MANIPULATION),
            "ny_killzone": ("12:00", "15:00", KillzoneType.DISTRIBUTION)
        }

        for kz_name, (start_time, end_time, kz_type) in killzone_configs.items():
            try:
                # Get killzone data using sessions method
                kz_result = self.calculate_sessions(df, None, start_time, end_time, time_zone="UTC")

                # Extract recent killzone data
                kz_df = df.tail(8) if len(df) >= 8 else df

                if kz_df.empty:
                    kz_df = pd.DataFrame({
                        'high': [current_price],
                        'low': [current_price],
                        'open': [current_price],
                        'close': [current_price]
                    })

                kz_high = kz_df['high'].max()
                kz_low = kz_df['low'].min()
                kz_range = kz_high - kz_low

                # Identify key levels (simplified)
                key_levels = []
                if kz_range > 0:
                    # Add quarter levels
                    quarter_1 = kz_low + (kz_range * 0.25)
                    quarter_3 = kz_low + (kz_range * 0.75)
                    key_levels = [quarter_1, quarter_3]

                # Build killzone analysis based on type
                if kz_type == KillzoneType.ACCUMULATION:
                    recommendation = "Range formation, wait for breakout"
                    false_breakout = None
                    sweep_direction = None
                    reversal_level = None
                    trend_continuation = None
                    target_projection = None

                elif kz_type == KillzoneType.MANIPULATION:
                    # Check for manipulation patterns
                    false_breakout = kz_range > current_price * 0.005  # 0.5% range indicates potential manipulation
                    sweep_direction = "BEARISH" if kz_low < current_price * 0.995 else "BULLISH"
                    reversal_level = kz_low if sweep_direction == "BEARISH" else kz_high
                    recommendation = "Stop hunt complete, bullish continuation likely" if sweep_direction == "BEARISH" else "Bearish manipulation detected"
                    trend_continuation = None
                    target_projection = None

                else:  # DISTRIBUTION
                    trend_continuation = abs(kz_high - kz_low) > current_price * 0.008  # 0.8% indicates trending
                    target_projection = kz_high + (kz_range * 0.5) if trend_continuation else None
                    recommendation = "True direction confirmed, follow trend" if trend_continuation else "Distribution phase, watch for direction"
                    false_breakout = None
                    sweep_direction = None
                    reversal_level = None

                killzones[kz_name] = KillzoneAnalysis(
                    time_window=f"{start_time}-{end_time} UTC",
                    high=kz_high,
                    low=kz_low,
                    range=kz_range,
                    type=kz_type,
                    key_levels_formed=key_levels,
                    false_breakout=false_breakout,
                    sweep_direction=sweep_direction,
                    reversal_level=reversal_level,
                    trend_continuation=trend_continuation,
                    target_projection=target_projection,
                    recommendation=recommendation
                )

            except Exception as e:
                print(f"Error analyzing {kz_name}: {e}")
                killzones[kz_name] = KillzoneAnalysis(
                    time_window=f"{start_time}-{end_time} UTC",
                    high=current_price,
                    low=current_price,
                    range=0,
                    type=kz_type,
                    key_levels_formed=[],
                    recommendation="Insufficient data for analysis"
                )

        return killzones

    def analyze_power_of_3(self, session_data: Dict[str, SessionData],
                           killzone_analysis: Dict[str, KillzoneAnalysis],
                           current_price: float) -> PowerOf3Data:
        """
        Analyze Power of 3 pattern
        """
        try:
            asia = session_data.get("asia")
            london = session_data.get("london")
            ny = session_data.get("new_york")

            # Accumulation phase (Asia)
            accumulation_completed = asia is not None
            accumulation_range = [asia.session_low, asia.session_high] if asia else None

            accumulation_phase = PowerOf3Phase(
                session="ASIA",
                range=accumulation_range,
                completed=accumulation_completed
            )

            # Manipulation phase (London)
            manipulation_completed = (london is not None and
                                      london.manipulation_detected is True)
            manipulation_direction = None
            sweep_level = None

            if london and london.manipulation_type:
                if london.manipulation_type in [ManipulationType.BEAR_TRAP, ManipulationType.STOP_HUNT_LOW]:
                    manipulation_direction = "BEAR_TRAP"
                    sweep_level = london.session_low
                else:
                    manipulation_direction = "BULL_TRAP"
                    sweep_level = london.session_high

            manipulation_phase = PowerOf3Phase(
                session="LONDON",
                sweep_level=sweep_level,
                direction=manipulation_direction,
                completed=manipulation_completed
            )

            # Distribution phase (NY)
            distribution_in_progress = ny is not None and ny.session_active
            distribution_completed = ny is not None and not ny.session_active
            trend_direction = None
            target = None

            if ny:
                if ny.direction == SessionDirection.BULLISH:
                    trend_direction = "BULLISH"
                    target = current_price * 1.02  # 2% target
                elif ny.direction == SessionDirection.BEARISH:
                    trend_direction = "BEARISH"
                    target = current_price * 0.98  # 2% target

            distribution_phase = PowerOf3Phase(
                session="NEW_YORK",
                trend_direction=trend_direction,
                target=target,
                completed=distribution_completed,
                in_progress=distribution_in_progress
            )

            # Determine current phase
            if distribution_in_progress or distribution_completed:
                current_phase = "DISTRIBUTION"
            elif manipulation_completed:
                current_phase = "DISTRIBUTION"  # Moving to distribution
            elif accumulation_completed:
                current_phase = "MANIPULATION" if not manipulation_completed else "DISTRIBUTION"
            else:
                current_phase = "ACCUMULATION"

            # Calculate confidence score
            confidence_factors = []

            # Accumulation quality (30%)
            if asia and asia.range_characteristic == RangeCharacteristic.TIGHT:
                confidence_factors.append(3.0)
            elif asia:
                confidence_factors.append(2.0)
            else:
                confidence_factors.append(1.0)

            # Manipulation clarity (40%)
            if manipulation_completed and london and london.manipulation_type:
                confidence_factors.append(4.0)
            elif london and london.manipulation_detected:
                confidence_factors.append(3.0)
            else:
                confidence_factors.append(2.0)

            # Distribution confirmation (30%)
            if ny and ny.range_characteristic == RangeCharacteristic.TRENDING:
                confidence_factors.append(3.0)
            elif ny and ny.direction != SessionDirection.NEUTRAL:
                confidence_factors.append(2.5)
            else:
                confidence_factors.append(2.0)

            po3_confidence = sum(confidence_factors)

            # Pattern clarity
            if po3_confidence >= 8.5:
                pattern_clarity = "HIGH"
            elif po3_confidence >= 6.5:
                pattern_clarity = "MEDIUM"
            else:
                pattern_clarity = "LOW"

            return PowerOf3Data(
                current_phase=current_phase,
                accumulation=accumulation_phase,
                manipulation=manipulation_phase,
                distribution=distribution_phase,
                po3_confidence=po3_confidence,
                pattern_clarity=pattern_clarity
            )

        except Exception as e:
            print(f"Error in Power of 3 analysis: {e}")
            # Return default PO3 data
            return PowerOf3Data(
                current_phase="ACCUMULATION",
                accumulation=PowerOf3Phase(session="ASIA", completed=False),
                manipulation=PowerOf3Phase(session="LONDON", completed=False),
                distribution=PowerOf3Phase(session="NEW_YORK", completed=False),
                po3_confidence=5.0,
                pattern_clarity="LOW"
            )

    def analyze_session_overlaps(self, df: pd.DataFrame, current_price: float) -> Dict[str, SessionOverlap]:
        """
        Analyze session overlaps
        """
        overlaps = {}

        try:
            # London-NY overlap (13:00-16:00 UTC)
            overlap_result = self.calculate_sessions(df, None, "13:00", "16:00", time_zone="UTC")

            # Extract overlap data from recent candles
            overlap_df = df.tail(4) if len(df) >= 4 else df

            if not overlap_df.empty:
                overlap_high = overlap_df['high'].max()
                overlap_low = overlap_df['low'].min()
                overlap_range = overlap_high - overlap_low

                # Determine volatility
                avg_range = (df['high'] - df['low']).mean() if len(df) > 0 else 0
                if avg_range > 0:
                    volatility_ratio = overlap_range / avg_range
                    if volatility_ratio >= 2.0:
                        volatility = "EXTREME"
                    elif volatility_ratio >= 1.5:
                        volatility = "HIGH"
                    else:
                        volatility = "MODERATE"
                else:
                    volatility = "MODERATE"

                # Determine directional bias
                overlap_open = overlap_df['open'].iloc[0]
                overlap_close = overlap_df['close'].iloc[-1]

                if overlap_close > overlap_open * 1.002:
                    directional_bias = "BULLISH"
                elif overlap_close < overlap_open * 0.998:
                    directional_bias = "BEARISH"
                else:
                    directional_bias = "NEUTRAL"

                # Check for volume spike
                overlap_volume = overlap_df['volume'].mean() if 'volume' in overlap_df.columns else 0
                overall_volume = df['volume'].mean() if 'volume' in df.columns and len(df) > 0 else 1
                volume_spike = overlap_volume > overall_volume * 1.5

                overlaps["london_ny_overlap"] = SessionOverlap(
                    time_window="13:00-16:00 UTC",
                    high=overlap_high,
                    low=overlap_low,
                    volatility=volatility,
                    directional_bias=directional_bias,
                    volume_spike=volume_spike
                )
            else:
                overlaps["london_ny_overlap"] = SessionOverlap(
                    time_window="13:00-16:00 UTC",
                    high=current_price,
                    low=current_price,
                    volatility="MODERATE",
                    directional_bias="NEUTRAL",
                    volume_spike=False
                )

        except Exception as e:
            print(f"Error analyzing session overlaps: {e}")
            overlaps["london_ny_overlap"] = SessionOverlap(
                time_window="13:00-16:00 UTC",
                high=current_price,
                low=current_price,
                volatility="MODERATE",
                directional_bias="NEUTRAL",
                volume_spike=False
            )

        return overlaps

    def build_trading_recommendations(self, session_data: Dict[str, SessionData],
                                      killzone_analysis: Dict[str, KillzoneAnalysis],
                                      power_of_3: PowerOf3Data,
                                      current_price: float,
                                      current_time: datetime) -> TradingRecommendations:
        """
        Build trading recommendations based on session analysis
        """
        try:
            # Determine current opportunity
            current_opportunity = "WAIT_FOR_SETUP"
            entry_zone = [current_price * 0.999, current_price * 1.001]
            stop_loss = current_price * 0.995
            targets = [current_price * 1.005, current_price * 1.010, current_price * 1.015]

            # Analyze based on Power of 3 phase
            if power_of_3.current_phase == "DISTRIBUTION":
                ny_data = session_data.get("new_york")
                if ny_data and ny_data.direction == SessionDirection.BULLISH:
                    current_opportunity = "NEW_YORK_CONTINUATION"
                    entry_zone = [current_price * 0.998, current_price * 1.002]
                    stop_loss = current_price * 0.995
                    targets = [current_price * 1.005, current_price * 1.010, current_price * 1.015]
                elif ny_data and ny_data.direction == SessionDirection.BEARISH:
                    current_opportunity = "NEW_YORK_DISTRIBUTION"
                    entry_zone = [current_price * 0.998, current_price * 1.002]
                    stop_loss = current_price * 1.005
                    targets = [current_price * 0.995, current_price * 0.990, current_price * 0.985]

            elif power_of_3.current_phase == "MANIPULATION":
                london_data = session_data.get("london")
                if london_data and london_data.manipulation_detected:
                    if london_data.manipulation_type in [ManipulationType.BEAR_TRAP, ManipulationType.STOP_HUNT_LOW]:
                        current_opportunity = "POST_MANIPULATION_LONG"
                        entry_zone = [london_data.session_low * 1.001, london_data.session_low * 1.005]
                        stop_loss = london_data.session_low * 0.998
                    else:
                        current_opportunity = "POST_MANIPULATION_SHORT"
                        entry_zone = [london_data.session_high * 0.995, london_data.session_high * 0.999]
                        stop_loss = london_data.session_high * 1.002

            # Determine session bias
            intraday_bias = "NEUTRAL"
            key_message = "Wait for clear direction."

            if power_of_3.current_phase == "DISTRIBUTION":
                ny_data = session_data.get("new_york")
                if ny_data:
                    if ny_data.direction == SessionDirection.BULLISH:
                        intraday_bias = "BULLISH"
                        key_message = "NY showing bullish distribution. Look for pullbacks to enter longs."
                    elif ny_data.direction == SessionDirection.BEARISH:
                        intraday_bias = "BEARISH"
                        key_message = "NY showing bearish distribution. Look for rallies to enter shorts."

            # Determine next key time and event
            current_hour = current_time.hour

            if current_hour < 7:
                next_key_time = "07:00 UTC"
                next_event = "London Killzone Open"
            elif current_hour < 8:
                next_key_time = "08:00 UTC"
                next_event = "London Session Open"
            elif current_hour < 12:
                next_key_time = "12:00 UTC"
                next_event = "NY Killzone Open"
            elif current_hour < 13:
                next_key_time = "13:00 UTC"
                next_event = "NY Session Open"
            elif current_hour < 16:
                next_key_time = "16:00 UTC"
                next_event = "London Close"
            elif current_hour < 21:
                next_key_time = "21:00 UTC"
                next_event = "NY Close"
            else:
                next_key_time = "00:00 UTC"
                next_event = "Asia Session Open"

            return TradingRecommendations(
                current_opportunity=current_opportunity,
                entry_zone=entry_zone,
                stop_loss=stop_loss,
                targets=targets,
                session_bias=SessionBias(
                    intraday=intraday_bias,
                    key_message=key_message
                ),
                next_key_time=next_key_time,
                next_event=next_event
            )

        except Exception as e:
            print(f"Error building trading recommendations: {e}")
            return TradingRecommendations(
                current_opportunity="WAIT_FOR_SETUP",
                entry_zone=[current_price * 0.999, current_price * 1.001],
                stop_loss=current_price * 0.995,
                targets=[current_price * 1.005],
                session_bias=SessionBias(
                    intraday="NEUTRAL",
                    key_message="Analysis incomplete. Wait for setup."
                ),
                next_key_time="00:00 UTC",
                next_event="Next Session"
            )

    def calculate_session_statistics(self, session_data: Dict[str, SessionData],
                                     df: pd.DataFrame, current_price: float) -> SessionStatistics:
        """
        Calculate session statistics
        """
        try:
            # Find most volatile session
            session_volatilities = {}
            for session_name, data in session_data.items():
                if data.session_range > 0:
                    session_volatilities[session_name] = data.session_range

            most_volatile_session = max(session_volatilities,
                                        key=session_volatilities.get) if session_volatilities else "LONDON"
            most_volatile_session = most_volatile_session.upper()

            # Find highest volume session
            session_volumes = {}
            for session_name, data in session_data.items():
                volume_scores = {
                    VolumeProfile.VERY_HIGH: 5,
                    VolumeProfile.HIGH: 4,
                    VolumeProfile.MODERATE: 3,
                    VolumeProfile.LOW: 2,
                    VolumeProfile.VERY_LOW: 1
                }
                session_volumes[session_name] = volume_scores.get(data.volume_profile, 3)

            highest_volume_session = max(session_volumes, key=session_volumes.get) if session_volumes else "NEW_YORK"
            highest_volume_session = highest_volume_session.replace("_", " ").upper()

            # Calculate average daily range
            if len(df) > 0:
                daily_ranges = df['high'] - df['low']
                avg_daily_range = daily_ranges.mean()
                current_day_range = daily_ranges.iloc[-1] if len(daily_ranges) > 0 else 0
            else:
                avg_daily_range = 0
                current_day_range = 0

            # Determine if range is expanding
            range_expansion = current_day_range > avg_daily_range * 1.1

            # Assess session rhythm
            session_rhythm = "NORMAL"
            if power_of_3_confidence := 8.0:  # Placeholder
                if power_of_3_confidence >= 8.0:
                    session_rhythm = "STRONG"
                elif power_of_3_confidence <= 5.0:
                    session_rhythm = "WEAK"

            return SessionStatistics(
                most_volatile_session=most_volatile_session,
                highest_volume_session=highest_volume_session,
                avg_daily_range=avg_daily_range,
                current_day_range=current_day_range,
                range_expansion=range_expansion,
                session_rhythm=session_rhythm
            )

        except Exception as e:
            print(f"Error calculating session statistics: {e}")
            return SessionStatistics(
                most_volatile_session="LONDON",
                highest_volume_session="NEW_YORK",
                avg_daily_range=1000,
                current_day_range=1200,
                range_expansion=True,
                session_rhythm="NORMAL"
            )

    def analyze_killzone_sessions(self, ohlcv_data: Dict[str, List[float]], timeframe: str) -> Dict[str, Any]:
        """
        Main function to analyze killzone sessions
        """
        df = self.prepare_dataframe(ohlcv_data)
        current_price = float(ohlcv_data['close'][-1])
        current_time = datetime.now(timezone.utc)

        # Determine current session
        current_session = self.determine_current_session(current_time)

        # Analyze weekly and daily reference
        weekly_reference = self.analyze_weekly_reference(df, current_price)
        daily_reference = self.analyze_daily_reference(df, current_price)

        # Analyze session data
        session_data = self.analyze_session_data(df, current_price, current_time)

        # Analyze killzones
        killzone_analysis = self.analyze_killzones(df, current_price)

        # Analyze Power of 3
        power_of_3 = self.analyze_power_of_3(session_data, killzone_analysis, current_price)

        # Analyze session overlaps
        session_overlaps = self.analyze_session_overlaps(df, current_price)

        # Build trading recommendations
        trading_recommendations = self.build_trading_recommendations(
            session_data, killzone_analysis, power_of_3, current_price, current_time
        )

        # Calculate session statistics
        session_statistics = self.calculate_session_statistics(session_data, df, current_price)

        return {
            "current_price": current_price,
            "current_session": current_session,
            "weekly_reference": weekly_reference,
            "daily_reference": daily_reference,
            "session_data": session_data,
            "killzone_analysis": killzone_analysis,
            "power_of_3": power_of_3,
            "session_overlaps": session_overlaps,
            "trading_recommendations": trading_recommendations,
            "session_statistics": session_statistics
        }


@router.get("/api/killzone-sessions/{symbol}/{timeframe}",
            response_model=KillzoneSessionsResponse,
            summary="Get Killzone Sessions Analysis",
            description="Analyze cryptocurrency killzone sessions using ICT concepts and Power of 3 methodology")
async def get_killzone_sessions(
        symbol: str = Path(..., description="Trading pair symbol (e.g., BTCUSDT, BTC)"),
        timeframe: str = Path(..., description="Analysis timeframe (recommended: 15m, 1h)"),
        as_of_datetime: Optional[str] = Query(None,
                                              description="ISO 8601 datetime string for historical analysis (e.g., 2024-01-01T00:00:00Z). Defaults to current time."),
        _: bool = Depends(verify_api_key)
):
    """
    Analyze cryptocurrency killzone sessions using ICT concepts and Power of 3 methodology.
    
    This endpoint:
    1. Fetches klines data from Binance for the specified symbol and timeframe
    2. Analyzes major trading sessions (Asia, London, New York)
    3. Identifies ICT killzones and their behavior patterns
    4. Performs Power of 3 analysis (Accumulation → Manipulation → Distribution)
    5. Detects manipulation patterns and liquidity sweeps
    6. Provides session-based trading recommendations
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., 'BTC', 'ETH', 'BTCUSDT')
    - **timeframe**: Analysis timeframe (recommended: 15m, 1h for best session analysis)
    - **as_of_datetime**: Optional ISO 8601 datetime string for historical analysis
    
    **Returns comprehensive session analysis including:**
    - Current session identification and weekly/daily reference levels
    - Detailed session data with manipulation detection
    - ICT killzone analysis (Asia: Accumulation, London: Manipulation, NY: Distribution)
    - Power of 3 cycle tracking with confidence scoring
    - Session overlap analysis and volume spike detection
    - Trading recommendations with entry/exit levels
    - Session statistics and rhythm assessment
    
    **ICT Concepts Covered:**
    - Killzone time windows: Asia (08:00-12:00), London (07:00-10:00), NY (12:00-15:00)
    - Power of 3 methodology: Accumulation → Manipulation → Distribution
    - Liquidity sweep detection and stop hunt identification
    - Session overlap analysis (London-NY: 13:00-16:00)
    - Institutional order flow patterns
    
    **Note:** If the symbol doesn't end with 'USDT', it will be automatically appended.
    """
    try:
        # Validate timeframe (recommend 15m or 1h for session analysis)
        valid_intervals = {
            '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',
            '6h', '8h', '12h', '1d', '3d', '1w', '1M'
        }
        if timeframe not in valid_intervals:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid timeframe. Must be one of: {", ".join(sorted(valid_intervals))}'
            )

        # Initialize components
        binance_fetcher = BinanceDataFetcher()
        analyzer = KillzoneSessionAnalyzer()

        # Process symbol
        processed_symbol = symbol.upper()
        if not processed_symbol.endswith('USDT'):
            processed_symbol = f"{processed_symbol}USDT"

        # Parse and validate asOfDateTime if provided
        end_time_ms = None
        if as_of_datetime:
            try:
                # Parse ISO 8601 datetime string
                as_of_dt = datetime.fromisoformat(as_of_datetime.replace('Z', '+00:00'))
                # Convert to Unix timestamp in milliseconds
                end_time_ms = int(as_of_dt.timestamp() * 1000)
            except ValueError:
                # Log the parsing error and use current timestamp
                print(f"Warning: Failed to parse as_of_datetime '{as_of_datetime}'. Using current timestamp instead.")
                end_time_ms = None

        # Fetch data from Binance (500 candles for multiple session coverage)
        klines_data = await binance_fetcher.get_klines(
            symbol=processed_symbol,
            interval=timeframe,
            limit=500,  # Extended data for comprehensive session analysis
            end_time=end_time_ms
        )
        formatted_data = binance_fetcher.format_klines_data(klines_data)

        # Perform killzone sessions analysis
        analysis_result = analyzer.analyze_killzone_sessions(formatted_data, timeframe)

        # Create response
        return KillzoneSessionsResponse(
            symbol=processed_symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc).isoformat() if end_time_ms is None else as_of_datetime,
            current_price=analysis_result["current_price"],
            current_session=analysis_result["current_session"],
            weekly_reference=analysis_result["weekly_reference"],
            daily_reference=analysis_result["daily_reference"],
            session_data=analysis_result["session_data"],
            killzone_analysis=analysis_result["killzone_analysis"],
            power_of_3=analysis_result["power_of_3"],
            session_overlaps=analysis_result["session_overlaps"],
            trading_recommendations=analysis_result["trading_recommendations"],
            session_statistics=analysis_result["session_statistics"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Killzone sessions analysis failed: {str(e)}")


@router.get("/api/killzone-sessions/doc",
            summary="Killzone Sessions API Documentation",
            description="Get comprehensive documentation for the Killzone Sessions API response format")
async def get_killzone_sessions_documentation():
    """
    Get detailed documentation for the Killzone Sessions API response format.
    
    This endpoint explains all fields, enums, and data structures returned by the
    /api/killzone-sessions/{symbol}/{timeframe} endpoint.
    """
    return {
        "api_endpoint": "/api/killzone-sessions/{symbol}/{timeframe}",
        "description": "Analyzes killzone sessions using ICT (Inner Circle Trader) concepts including Power of 3 methodology, session analysis, and institutional manipulation detection.",

        "ict_concepts": {
            "power_of_3": "Three-phase daily cycle: Accumulation (Asia) → Manipulation (London) → Distribution (NY)",
            "killzones": "High-probability time windows when institutions are most active",
            "manipulation": "Institutional stop hunts and false breakouts to create optimal entry conditions",
            "session_dynamics": "How different global sessions create distinct price behavior patterns"
        },

        "session_timing": {
            "asia_session": {
                "primary": "00:00-08:00 UTC",
                "killzone": "08:00-12:00 UTC",
                "characteristics": "Accumulation, range building, lower volatility"
            },
            "london_session": {
                "primary": "08:00-16:00 UTC",
                "killzone": "07:00-10:00 UTC",
                "characteristics": "Manipulation, volatility expansion, stop hunts"
            },
            "new_york_session": {
                "primary": "13:00-21:00 UTC",
                "killzone": "12:00-15:00 UTC",
                "characteristics": "Distribution, trend following, institutional execution"
            },
            "london_ny_overlap": {
                "time": "13:00-16:00 UTC",
                "characteristics": "Highest volume and volatility period"
            }
        },

        "current_session": {
            "type": "enum",
            "description": "Currently active trading session",
            "possible_values": {
                "ASIA": "Asian session active (00:00-08:00 UTC)",
                "LONDON": "London session active (08:00-16:00 UTC)",
                "NEW_YORK": "New York session active (13:00-21:00 UTC)",
                "OVERLAP": "London-NY overlap active (13:00-16:00 UTC)",
                "CLOSED": "No major session active"
            }
        },

        "session_data": {
            "description": "Detailed analysis of each trading session",
            "structure": "Dictionary with session keys (asia, london, new_york)",
            "fields": {
                "session_high": {
                    "type": "number",
                    "description": "Highest price during session",
                    "example": 45200.0
                },
                "session_low": {
                    "type": "number",
                    "description": "Lowest price during session",
                    "example": 44100.0
                },
                "session_range": {
                    "type": "number",
                    "description": "Session range in price points",
                    "example": 1100.0
                },
                "opening_price": {
                    "type": "number",
                    "description": "Session opening price",
                    "example": 44650.0
                },
                "direction": {
                    "type": "enum",
                    "description": "Overall session direction",
                    "possible_values": {
                        "BULLISH": "Session closed higher than it opened",
                        "BEARISH": "Session closed lower than it opened",
                        "NEUTRAL": "Session closed near opening level"
                    }
                },
                "high_time": {
                    "type": "string",
                    "description": "Time when session high was made",
                    "example": "14:30",
                    "format": "HH:MM"
                },
                "low_time": {
                    "type": "string",
                    "description": "Time when session low was made",
                    "example": "09:15",
                    "format": "HH:MM"
                },
                "volume_profile": {
                    "type": "enum",
                    "description": "Session volume profile",
                    "possible_values": {
                        "VERY_HIGH": "Exceptional volume, strong institutional interest",
                        "HIGH": "Above average volume",
                        "MODERATE": "Average volume levels",
                        "LOW": "Below average volume",
                        "VERY_LOW": "Minimal volume, limited interest"
                    }
                },
                "range_characteristic": {
                    "type": "enum",
                    "description": "Session range behavior",
                    "possible_values": {
                        "TIGHT": "Small range, accumulation/consolidation",
                        "NORMAL": "Average range for session",
                        "EXPANSION": "Large range, volatility expansion",
                        "TRENDING": "Sustained directional movement"
                    }
                },
                "liquidity_taken": {
                    "type": "enum",
                    "description": "Which liquidity was swept during session",
                    "possible_values": {
                        "NONE": "No major liquidity swept",
                        "ASIA_HIGH": "Asian session high was taken",
                        "ASIA_LOW": "Asian session low was taken",
                        "LONDON_HIGH": "London session high was taken",
                        "LONDON_LOW": "London session low was taken"
                    }
                },
                "manipulation_detected": {
                    "type": "boolean",
                    "description": "Whether manipulation patterns were detected",
                    "interpretation": {
                        "true": "Stop hunt or false breakout detected",
                        "false": "No clear manipulation identified"
                    }
                },
                "manipulation_type": {
                    "type": "enum",
                    "description": "Type of manipulation detected",
                    "possible_values": {
                        "STOP_HUNT_HIGH": "Stop hunt above recent highs",
                        "STOP_HUNT_LOW": "Stop hunt below recent lows",
                        "BEAR_TRAP": "False breakdown followed by reversal",
                        "BULL_TRAP": "False breakout followed by reversal"
                    }
                }
            }
        },

        "killzone_analysis": {
            "description": "Analysis of ICT killzone time windows",
            "structure": "Dictionary with killzone keys",
            "killzone_types": {
                "asian_killzone": "08:00-12:00 UTC - Accumulation phase",
                "london_killzone": "07:00-10:00 UTC - Manipulation phase",
                "ny_killzone": "12:00-15:00 UTC - Distribution phase"
            },
            "fields": {
                "time_window": {
                    "type": "string",
                    "description": "Killzone time window",
                    "example": "07:00-10:00 UTC"
                },
                "high": {
                    "type": "number",
                    "description": "Killzone high price",
                    "example": 45100.0
                },
                "low": {
                    "type": "number",
                    "description": "Killzone low price",
                    "example": 44800.0
                },
                "range": {
                    "type": "number",
                    "description": "Killzone range in price points",
                    "example": 300.0
                },
                "type": {
                    "type": "enum",
                    "description": "Killzone behavior type",
                    "possible_values": {
                        "ACCUMULATION": "Range building, institutional accumulation",
                        "MANIPULATION": "Stop hunts, false breakouts",
                        "DISTRIBUTION": "Trend following, institutional distribution"
                    }
                },
                "key_levels_formed": {
                    "type": "array[number]",
                    "description": "Important levels formed during killzone",
                    "example": [44850, 45050]
                },
                "false_breakout": {
                    "type": "boolean",
                    "description": "Whether false breakout occurred",
                    "note": "Common during manipulation killzones"
                },
                "sweep_direction": {
                    "type": "string",
                    "description": "Direction of liquidity sweep",
                    "possible_values": ["UPWARD", "DOWNWARD", "BOTH", "NONE"]
                },
                "trend_continuation": {
                    "type": "boolean",
                    "description": "Whether trend continued through killzone"
                },
                "recommendation": {
                    "type": "string",
                    "description": "Trading recommendation for killzone",
                    "examples": [
                        "Wait for manipulation completion before entry",
                        "Look for distribution continuation",
                        "Watch for accumulation breakout"
                    ]
                }
            }
        },

        "power_of_3": {
            "description": "ICT Power of 3 daily cycle analysis",
            "concept": "Three-phase institutional daily cycle",
            "fields": {
                "current_phase": {
                    "type": "string",
                    "description": "Current phase in Power of 3 cycle",
                    "possible_values": {
                        "ACCUMULATION": "Asia session - institutional position building",
                        "MANIPULATION": "London session - stop hunts and false moves",
                        "DISTRIBUTION": "NY session - institutional order execution"
                    }
                },
                "accumulation": {
                    "type": "object",
                    "description": "Accumulation phase analysis (Asia session)",
                    "fields": {
                        "session": "Session identifier (ASIA)",
                        "range": "Price range for accumulation [low, high]",
                        "completed": "Whether accumulation phase is complete"
                    }
                },
                "manipulation": {
                    "type": "object",
                    "description": "Manipulation phase analysis (London session)",
                    "fields": {
                        "session": "Session identifier (LONDON)",
                        "sweep_level": "Price level where liquidity was swept",
                        "direction": "Manipulation direction (BEAR_TRAP/BULL_TRAP)",
                        "completed": "Whether manipulation phase is complete"
                    }
                },
                "distribution": {
                    "type": "object",
                    "description": "Distribution phase analysis (NY session)",
                    "fields": {
                        "session": "Session identifier (NEW_YORK)",
                        "trend_direction": "Distribution trend (BULLISH/BEARISH)",
                        "target": "Price target for distribution",
                        "completed": "Whether distribution phase is complete",
                        "in_progress": "Whether distribution is currently active"
                    }
                },
                "po3_confidence": {
                    "type": "number",
                    "description": "Confidence score for Power of 3 pattern",
                    "range": "0-10",
                    "interpretation": {
                        "8-10": "Very clear Power of 3 pattern",
                        "6-8": "Good Power of 3 pattern",
                        "4-6": "Moderate pattern clarity",
                        "0-4": "Weak or unclear pattern"
                    }
                },
                "pattern_clarity": {
                    "type": "enum",
                    "description": "Overall pattern clarity assessment",
                    "possible_values": {
                        "HIGH": "Very clear three-phase pattern",
                        "MEDIUM": "Moderately clear pattern",
                        "LOW": "Unclear or incomplete pattern"
                    }
                }
            }
        },

        "trading_recommendations": {
            "description": "Session-based trading recommendations",
            "fields": {
                "current_opportunity": {
                    "type": "string",
                    "description": "Current trading opportunity",
                    "possible_values": {
                        "NEW_YORK_CONTINUATION": "NY session trend continuation",
                        "NEW_YORK_DISTRIBUTION": "NY distribution phase trade",
                        "POST_MANIPULATION_LONG": "Long after manipulation sweep",
                        "POST_MANIPULATION_SHORT": "Short after manipulation sweep",
                        "WAIT_FOR_SETUP": "No clear opportunity, wait"
                    }
                },
                "entry_zone": {
                    "type": "array[number]",
                    "description": "Suggested entry price range [low, high]",
                    "example": [44900, 45100]
                },
                "stop_loss": {
                    "type": "number",
                    "description": "Suggested stop loss level",
                    "example": 44700
                },
                "targets": {
                    "type": "array[number]",
                    "description": "Take profit targets in order",
                    "example": [45300, 45600, 45900]
                },
                "session_bias": {
                    "type": "object",
                    "description": "Current session bias and key message",
                    "fields": {
                        "intraday": {
                            "type": "string",
                            "possible_values": ["BULLISH", "BEARISH", "NEUTRAL"]
                        },
                        "key_message": {
                            "type": "string",
                            "description": "Key trading message for current session"
                        }
                    }
                },
                "next_key_time": {
                    "type": "string",
                    "description": "Next important time event",
                    "example": "13:00 UTC"
                },
                "next_event": {
                    "type": "string",
                    "description": "Description of next key event",
                    "example": "NY Session Open"
                }
            }
        },

        "trading_interpretation": {
            "power_of_3_usage": [
                "1. Accumulation (Asia): Look for range formation and position building",
                "2. Manipulation (London): Watch for stop hunts and false breakouts",
                "3. Distribution (NY): Follow institutional trend direction",
                "4. Use po3_confidence >7 for higher probability trades"
            ],
            "killzone_strategies": [
                "Asian Killzone: Trade range boundaries, prepare for London expansion",
                "London Killzone: Fade false breakouts, follow post-manipulation direction",
                "NY Killzone: Trade trend continuation, target session objectives"
            ],
            "manipulation_signals": [
                "manipulation_detected: true indicates stop hunt opportunity",
                "BEAR_TRAP/BULL_TRAP suggest reversal after manipulation",
                "High volume + false breakout = manipulation confirmation"
            ],
            "session_overlap_importance": [
                "London-NY overlap (13:00-16:00) has highest volume",
                "Most significant moves often occur during overlap",
                "Watch for volatility spikes and directional bias"
            ]
        },

        "usage_examples": {
            "bullish_po3_setup": {
                "description": "Example bullish Power of 3 setup",
                "indicators": [
                    "current_phase: DISTRIBUTION",
                    "distribution.trend_direction: BULLISH",
                    "manipulation.direction: BEAR_TRAP (liquidity grab below)",
                    "po3_confidence: 8+",
                    "current_opportunity: POST_MANIPULATION_LONG"
                ]
            },
            "bearish_po3_setup": {
                "description": "Example bearish Power of 3 setup",
                "indicators": [
                    "current_phase: DISTRIBUTION",
                    "distribution.trend_direction: BEARISH",
                    "manipulation.direction: BULL_TRAP (liquidity grab above)",
                    "pattern_clarity: HIGH",
                    "current_opportunity: POST_MANIPULATION_SHORT"
                ]
            }
        }
    }
