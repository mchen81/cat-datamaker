from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field

from binance_data import BinanceDataFetcher
from smc_analysis import SMCAnalyzer
from utils import verify_api_key


class LiquidityStatus(str, Enum):
    UNTAPPED = "UNTAPPED"
    SWEPT = "SWEPT"
    PARTIALLY_SWEPT = "PARTIALLY_SWEPT"


class SweepProbability(str, Enum):
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CurrentPosition(str, Enum):
    ABOVE_RANGE = "ABOVE_RANGE"
    UPPER_THIRD = "UPPER_THIRD"
    MID_RANGE = "MID_RANGE"
    LOWER_THIRD = "LOWER_THIRD"
    BELOW_RANGE = "BELOW_RANGE"
    INSIDE_RANGE = "INSIDE_RANGE"


class LiquidityImbalance(str, Enum):
    SELL_SIDE_HEAVY = "SELL_SIDE_HEAVY"
    BUY_SIDE_HEAVY = "BUY_SIDE_HEAVY"
    BALANCED = "BALANCED"


class RecommendedBias(str, Enum):
    BULLISH_SWEEP_LIKELY = "BULLISH_SWEEP_LIKELY"
    BEARISH_SWEEP_LIKELY = "BEARISH_SWEEP_LIKELY"
    RANGE_BOUND = "RANGE_BOUND"
    BREAKOUT_PENDING = "BREAKOUT_PENDING"


class LiquidityPool(BaseModel):
    level: float = Field(..., description="Price level of liquidity zone")
    distance_from_current: str = Field(..., description="Distance from current price as percentage")
    type: str = Field(..., description="Type of liquidity (Equal Highs/Lows, Previous Day High, etc.)")
    status: LiquidityStatus = Field(..., description="Current status of liquidity zone")
    formation_candles: Optional[List[int]] = Field(None, description="Candle indices where zone was formed")
    sweep_candle: Optional[int] = Field(None, description="Candle index where zone was swept")
    age_hours: int = Field(..., description="Age of liquidity zone in hours")
    strength: int = Field(..., description="Strength score 1-10", ge=1, le=10)
    sweep_probability: SweepProbability = Field(..., description="Probability of being swept")


class LiquidityPools(BaseModel):
    buy_side_liquidity: List[LiquidityPool] = Field(..., description="Liquidity below current price")
    sell_side_liquidity: List[LiquidityPool] = Field(..., description="Liquidity above current price")


class TimeframeLevel(BaseModel):
    previous_high: Optional[float] = Field(None, description="Previous timeframe high")
    previous_low: Optional[float] = Field(None, description="Previous timeframe low")
    current_position: CurrentPosition = Field(..., description="Current price position in range")
    high_distance: Optional[str] = Field(None, description="Distance to high as percentage")
    low_distance: Optional[str] = Field(None, description="Distance to low as percentage")


class SessionData(BaseModel):
    high: Optional[float] = Field(None, description="Session high")
    low: Optional[float] = Field(None, description="Session low")
    high_swept: bool = Field(..., description="Whether session high was swept")
    low_swept: bool = Field(..., description="Whether session low was swept")
    session_range: Optional[float] = Field(None, description="Session range in price points")


class LiquidityTarget(BaseModel):
    direction: str = Field(..., description="BUY_SIDE or SELL_SIDE")
    level: float = Field(..., description="Target price level")
    reason: str = Field(..., description="Reason for targeting this level")


class LiquidityAnalysis(BaseModel):
    primary_target: Optional[LiquidityTarget] = Field(None, description="Primary liquidity target")
    secondary_target: Optional[LiquidityTarget] = Field(None, description="Secondary liquidity target")
    liquidity_imbalance: LiquidityImbalance = Field(..., description="Overall liquidity distribution")
    recommended_bias: RecommendedBias = Field(..., description="Recommended trading bias")
    key_message: str = Field(..., description="Key takeaway message")


class LiquidityVoidZone(BaseModel):
    range: List[float] = Field(..., description="Price range of void zone [low, high]")
    description: str = Field(..., description="Description of the void zone")


class LiquidityMetrics(BaseModel):
    total_buy_side_pools: int = Field(..., description="Number of buy side liquidity pools")
    total_sell_side_pools: int = Field(..., description="Number of sell side liquidity pools")
    nearest_untapped_above: Optional[float] = Field(None, description="Nearest untapped liquidity above")
    nearest_untapped_below: Optional[float] = Field(None, description="Nearest untapped liquidity below")
    liquidity_void_zones: List[LiquidityVoidZone] = Field(..., description="Areas with thin liquidity")


class LiquidityZonesResponse(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Analysis timeframe")
    timestamp: str = Field(..., description="Analysis timestamp")
    current_price: float = Field(..., description="Current price")
    liquidity_pools: LiquidityPools = Field(..., description="Buy and sell side liquidity pools")
    timeframe_highs_lows: Dict[str, TimeframeLevel] = Field(..., description="Multi-timeframe levels")
    session_liquidity: Dict[str, SessionData] = Field(..., description="Trading session liquidity")
    liquidity_analysis: LiquidityAnalysis = Field(..., description="Comprehensive liquidity analysis")
    liquidity_metrics: LiquidityMetrics = Field(..., description="Liquidity summary metrics")


# Create router
router = APIRouter(tags=["Liquidity Zones"])


class LiquidityZoneAnalyzer(SMCAnalyzer):
    """
    Liquidity Zone Analysis using smartmoneyconcepts library
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

    def get_hours_ago(self, candle_index: int, timeframe: str) -> int:
        """
        Calculate hours ago based on timeframe and candle index
        """
        timeframe_hours = {
            '1m': 1 / 60, '3m': 3 / 60, '5m': 5 / 60, '15m': 15 / 60, '30m': 30 / 60,
            '1h': 1, '2h': 2, '4h': 4, '6h': 6, '8h': 8, '12h': 12,
            '1d': 24, '3d': 72, '1w': 168, '1M': 720
        }

        interval_hours = timeframe_hours.get(timeframe, 1)
        return int(abs(candle_index) * interval_hours)

    def calculate_sweep_probability(self, level: float, current_price: float,
                                    age_hours: int, strength: int,
                                    market_structure_direction: str = "NEUTRAL") -> SweepProbability:
        """
        Calculate probability of liquidity being swept
        """
        distance_percent = abs((level - current_price) / current_price) * 100

        # Base score from distance (closer = higher probability)
        if distance_percent < 1:
            distance_score = 4
        elif distance_percent < 3:
            distance_score = 3
        elif distance_percent < 5:
            distance_score = 2
        else:
            distance_score = 1

        # Age factor (older = higher probability)
        if age_hours > 168:  # Over a week
            age_score = 3
        elif age_hours > 72:  # Over 3 days
            age_score = 2
        else:
            age_score = 1

        # Strength factor
        strength_score = min(3, strength // 3)

        # Market structure alignment
        structure_score = 1
        if level > current_price and market_structure_direction == "BULLISH":
            structure_score = 2
        elif level < current_price and market_structure_direction == "BEARISH":
            structure_score = 2

        total_score = distance_score + age_score + strength_score + structure_score

        if total_score >= 10:
            return SweepProbability.VERY_HIGH
        elif total_score >= 8:
            return SweepProbability.HIGH
        elif total_score >= 6:
            return SweepProbability.MEDIUM
        else:
            return SweepProbability.LOW

    def calculate_liquidity_strength(self, formation_count: int, age_hours: int,
                                     volume_factor: float = 1.0) -> int:
        """
        Calculate liquidity strength score (1-10)
        """
        # Base score from formation touches
        touch_score = min(5, formation_count)

        # Age factor (older zones are stronger)
        if age_hours > 168:  # Over a week
            age_score = 3
        elif age_hours > 72:  # Over 3 days
            age_score = 2
        elif age_hours > 24:  # Over a day
            age_score = 1
        else:
            age_score = 0

        # Volume factor
        volume_score = min(2, int(volume_factor))

        return min(10, max(1, touch_score + age_score + volume_score))

    def determine_current_position(self, current_price: float, high: Optional[float],
                                   low: Optional[float]) -> CurrentPosition:
        """
        Determine current price position relative to high/low range
        """
        if high is None or low is None:
            return CurrentPosition.INSIDE_RANGE

        if current_price > high:
            return CurrentPosition.ABOVE_RANGE
        elif current_price < low:
            return CurrentPosition.BELOW_RANGE
        else:
            range_size = high - low
            position_in_range = (current_price - low) / range_size

            if position_in_range > 0.66:
                return CurrentPosition.UPPER_THIRD
            elif position_in_range > 0.33:
                return CurrentPosition.MID_RANGE
            else:
                return CurrentPosition.LOWER_THIRD

    def analyze_liquidity_pools(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame,
                                current_price: float, timeframe: str) -> LiquidityPools:
        """
        Analyze liquidity pools using smartmoneyconcepts
        """
        try:
            # Get liquidity data using inherited method
            liquidity_data = self.calculate_liquidity(df, swing_highs_lows, range_percent=0.01)

            buy_side_pools = []
            sell_side_pools = []

            if liquidity_data and liquidity_data["liquidity_levels"]:
                for liq_level in liquidity_data["liquidity_levels"]:
                    level = liq_level.get("level", 0)
                    liquidity_type = liq_level.get("liquidity", "")
                    swept_status = liq_level.get("swept", None)

                    # Determine if this is a buy-side or sell-side liquidity
                    is_buy_side = level < current_price

                    # Calculate relative position in dataframe for timing
                    try:
                        # Find the timestamp in the original dataframe
                        timestamp_str = liq_level.get("timestamp", "")
                        if timestamp_str:
                            timestamp = pd.to_datetime(timestamp_str)
                            if timestamp in df.index:
                                position_in_df = df.index.get_loc(timestamp)
                                candle_index = -(len(df) - position_in_df - 1)
                            else:
                                candle_index = -10  # Default fallback
                        else:
                            candle_index = -10
                    except Exception:
                        candle_index = -10

                    age_hours = self.get_hours_ago(candle_index, timeframe)

                    # Determine status
                    if swept_status is not None and swept_status:
                        status = LiquidityStatus.SWEPT
                    else:
                        status = LiquidityStatus.UNTAPPED

                    # Determine type based on liquidity field
                    if liquidity_type == "High":
                        liq_type = "Equal Highs"
                    elif liquidity_type == "Low":
                        liq_type = "Equal Lows"
                    else:
                        liq_type = "Liquidity Zone"

                    # Calculate strength and sweep probability
                    strength = self.calculate_liquidity_strength(2, age_hours)  # Assume 2 touches for equal highs/lows
                    sweep_prob = self.calculate_sweep_probability(level, current_price, age_hours, strength)

                    pool = LiquidityPool(
                        level=level,
                        distance_from_current=self.calculate_distance_percentage(level, current_price),
                        type=liq_type,
                        status=status,
                        formation_candles=[candle_index],
                        age_hours=age_hours,
                        strength=strength,
                        sweep_probability=sweep_prob
                    )

                    if is_buy_side:
                        buy_side_pools.append(pool)
                    else:
                        sell_side_pools.append(pool)

            # Sort pools by distance from current price
            buy_side_pools.sort(key=lambda x: abs(x.level - current_price))
            sell_side_pools.sort(key=lambda x: abs(x.level - current_price))

            return LiquidityPools(
                buy_side_liquidity=buy_side_pools[:10],  # Limit to top 10
                sell_side_liquidity=sell_side_pools[:10]
            )

        except Exception as e:
            print(f"Error in liquidity pool analysis: {e}")
            return LiquidityPools(buy_side_liquidity=[], sell_side_liquidity=[])

    def analyze_timeframe_levels(self, df: pd.DataFrame, current_price: float) -> Dict[str, TimeframeLevel]:
        """
        Analyze multiple timeframe previous highs and lows
        """
        timeframes = ["1D", "1W", "1M"]
        timeframe_names = ["daily", "weekly", "monthly"]
        results = {}

        for tf, name in zip(timeframes, timeframe_names):
            try:
                prev_hl_data = self.calculate_previous_high_low(df, time_frame=tf)

                previous_high = None
                previous_low = None

                # Extract the most recent previous high/low
                if prev_hl_data["previous_highs"]:
                    previous_high = prev_hl_data["previous_highs"][-1]["level"]
                if prev_hl_data["previous_lows"]:
                    previous_low = prev_hl_data["previous_lows"][-1]["level"]

                # Determine current position
                current_position = self.determine_current_position(current_price, previous_high, previous_low)

                # Calculate distances
                high_distance = None
                low_distance = None
                if previous_high:
                    high_distance = self.calculate_distance_percentage(previous_high, current_price)
                if previous_low:
                    low_distance = self.calculate_distance_percentage(previous_low, current_price)

                results[name] = TimeframeLevel(
                    previous_high=previous_high,
                    previous_low=previous_low,
                    current_position=current_position,
                    high_distance=high_distance,
                    low_distance=low_distance
                )

            except Exception as e:
                print(f"Error analyzing {name} timeframe: {e}")
                results[name] = TimeframeLevel(
                    current_position=CurrentPosition.INSIDE_RANGE
                )

        return results

    def analyze_session_liquidity(self, df: pd.DataFrame, current_price: float) -> Dict[str, SessionData]:
        """
        Analyze trading session liquidity
        """
        sessions = {
            "asia": ("Tokyo", "00:00", "08:00"),
            "london": ("London", "08:00", "16:00"),
            "new_york": ("New York", "13:00", "21:00")
        }

        results = {}

        for session_name, (city, start_time, end_time) in sessions.items():
            try:
                session_data = self.calculate_sessions(df, city, start_time, end_time, time_zone="UTC")

                session_high = None
                session_low = None
                high_swept = False
                low_swept = False

                # Extract session highs and lows
                if session_data["session_highs"]:
                    session_high = max([item["level"] for item in session_data["session_highs"]])
                    # Check if high was swept (current price > session high)
                    high_swept = current_price > session_high

                if session_data["session_lows"]:
                    session_low = min([item["level"] for item in session_data["session_lows"]])
                    # Check if low was swept (current price < session low)
                    low_swept = current_price < session_low

                # Calculate session range
                session_range = None
                if session_high and session_low:
                    session_range = session_high - session_low

                results[session_name] = SessionData(
                    high=session_high,
                    low=session_low,
                    high_swept=high_swept,
                    low_swept=low_swept,
                    session_range=session_range
                )

            except Exception as e:
                print(f"Error analyzing {session_name} session: {e}")
                results[session_name] = SessionData(
                    high_swept=False,
                    low_swept=False
                )

        return results

    def build_liquidity_analysis(self, liquidity_pools: LiquidityPools,
                                 timeframe_levels: Dict[str, TimeframeLevel],
                                 session_liquidity: Dict[str, SessionData],
                                 current_price: float) -> LiquidityAnalysis:
        """
        Build comprehensive liquidity analysis with trading recommendations
        """
        # Find primary and secondary targets
        primary_target = None
        secondary_target = None

        # Look for highest probability untapped liquidity
        all_pools = liquidity_pools.buy_side_liquidity + liquidity_pools.sell_side_liquidity
        untapped_pools = [p for p in all_pools if p.status == LiquidityStatus.UNTAPPED]

        if untapped_pools:
            # Sort by sweep probability and strength
            sorted_pools = sorted(untapped_pools,
                                  key=lambda x: (x.sweep_probability == SweepProbability.VERY_HIGH,
                                                 x.sweep_probability == SweepProbability.HIGH,
                                                 x.strength),
                                  reverse=True)

            if sorted_pools:
                best_pool = sorted_pools[0]
                direction = "SELL_SIDE" if best_pool.level > current_price else "BUY_SIDE"
                primary_target = LiquidityTarget(
                    direction=direction,
                    level=best_pool.level,
                    reason=f"{best_pool.type} with {best_pool.sweep_probability.lower()} sweep probability"
                )

                # Find secondary target
                if len(sorted_pools) > 1:
                    second_pool = sorted_pools[1]
                    direction = "SELL_SIDE" if second_pool.level > current_price else "BUY_SIDE"
                    secondary_target = LiquidityTarget(
                        direction=direction,
                        level=second_pool.level,
                        reason=f"{second_pool.type}, secondary target"
                    )

        # Determine liquidity imbalance
        buy_side_count = len([p for p in liquidity_pools.buy_side_liquidity if p.status == LiquidityStatus.UNTAPPED])
        sell_side_count = len([p for p in liquidity_pools.sell_side_liquidity if p.status == LiquidityStatus.UNTAPPED])

        if sell_side_count > buy_side_count + 1:
            imbalance = LiquidityImbalance.SELL_SIDE_HEAVY
        elif buy_side_count > sell_side_count + 1:
            imbalance = LiquidityImbalance.BUY_SIDE_HEAVY
        else:
            imbalance = LiquidityImbalance.BALANCED

        # Determine recommended bias
        if primary_target:
            if primary_target.direction == "SELL_SIDE":
                bias = RecommendedBias.BULLISH_SWEEP_LIKELY
            else:
                bias = RecommendedBias.BEARISH_SWEEP_LIKELY
        else:
            bias = RecommendedBias.RANGE_BOUND

        # Generate key message
        if primary_target and secondary_target:
            key_message = f"Price between key liquidity levels. {primary_target.direction.lower().replace('_', ' ')} sweep at {primary_target.level:.0f} likely."
        elif primary_target:
            key_message = f"Primary target: {primary_target.direction.lower().replace('_', ' ')} liquidity at {primary_target.level:.0f}."
        else:
            key_message = "Limited untapped liquidity. Price likely to remain range-bound."

        return LiquidityAnalysis(
            primary_target=primary_target,
            secondary_target=secondary_target,
            liquidity_imbalance=imbalance,
            recommended_bias=bias,
            key_message=key_message
        )

    def calculate_liquidity_metrics(self, liquidity_pools: LiquidityPools,
                                    current_price: float) -> LiquidityMetrics:
        """
        Calculate comprehensive liquidity metrics
        """
        # Count pools
        total_buy_side = len(liquidity_pools.buy_side_liquidity)
        total_sell_side = len(liquidity_pools.sell_side_liquidity)

        # Find nearest untapped liquidity
        nearest_above = None
        nearest_below = None

        untapped_above = [p for p in liquidity_pools.sell_side_liquidity
                          if p.status == LiquidityStatus.UNTAPPED and p.level > current_price]
        untapped_below = [p for p in liquidity_pools.buy_side_liquidity
                          if p.status == LiquidityStatus.UNTAPPED and p.level < current_price]

        if untapped_above:
            nearest_above = min(untapped_above, key=lambda x: abs(x.level - current_price)).level
        if untapped_below:
            nearest_below = max(untapped_below, key=lambda x: abs(x.level - current_price)).level

        # Identify void zones (areas with no liquidity)
        void_zones = []
        all_levels = [p.level for p in liquidity_pools.buy_side_liquidity + liquidity_pools.sell_side_liquidity]
        all_levels.sort()

        # Look for gaps between liquidity levels
        for i in range(len(all_levels) - 1):
            gap = all_levels[i + 1] - all_levels[i]
            gap_percent = (gap / current_price) * 100

            # Consider gaps > 1% as void zones
            if gap_percent > 1.0:
                void_zones.append(LiquidityVoidZone(
                    range=[all_levels[i], all_levels[i + 1]],
                    description=f"Thin liquidity zone, {gap_percent:.1f}% gap"
                ))

        return LiquidityMetrics(
            total_buy_side_pools=total_buy_side,
            total_sell_side_pools=total_sell_side,
            nearest_untapped_above=nearest_above,
            nearest_untapped_below=nearest_below,
            liquidity_void_zones=void_zones[:5]  # Limit to top 5 void zones
        )

    def analyze_liquidity_zones(self, ohlcv_data: Dict[str, List[float]], timeframe: str) -> Dict[str, Any]:
        """
        Main function to analyze liquidity zones using smartmoneyconcepts
        """
        df = self.prepare_dataframe(ohlcv_data)
        current_price = float(ohlcv_data['close'][-1])

        # Calculate swing highs and lows (required for liquidity analysis)
        swing_data = self.calculate_swing_highs_lows(df, swing_length=50)
        swing_highs_lows = swing_data["swing_dataframe"]

        # Analyze liquidity pools
        liquidity_pools = self.analyze_liquidity_pools(df, swing_highs_lows, current_price, timeframe)

        # Add previous timeframe highs/lows to liquidity pools
        timeframe_levels = self.analyze_timeframe_levels(df, current_price)

        # Add timeframe levels as liquidity pools
        for tf_name, tf_data in timeframe_levels.items():
            if tf_data.previous_high and tf_data.previous_high != current_price:
                # Add as sell-side liquidity
                age_hours = 24 if tf_name == "daily" else (168 if tf_name == "weekly" else 720)
                strength = self.calculate_liquidity_strength(1, age_hours, 1.5)
                sweep_prob = self.calculate_sweep_probability(tf_data.previous_high, current_price, age_hours, strength)

                tf_pool = LiquidityPool(
                    level=tf_data.previous_high,
                    distance_from_current=self.calculate_distance_percentage(tf_data.previous_high, current_price),
                    type=f"Previous {tf_name.title()} High",
                    status=LiquidityStatus.UNTAPPED,
                    age_hours=age_hours,
                    strength=strength,
                    sweep_probability=sweep_prob
                )
                liquidity_pools.sell_side_liquidity.append(tf_pool)

            if tf_data.previous_low and tf_data.previous_low != current_price:
                # Add as buy-side liquidity
                age_hours = 24 if tf_name == "daily" else (168 if tf_name == "weekly" else 720)
                strength = self.calculate_liquidity_strength(1, age_hours, 1.5)
                sweep_prob = self.calculate_sweep_probability(tf_data.previous_low, current_price, age_hours, strength)

                tf_pool = LiquidityPool(
                    level=tf_data.previous_low,
                    distance_from_current=self.calculate_distance_percentage(tf_data.previous_low, current_price),
                    type=f"Previous {tf_name.title()} Low",
                    status=LiquidityStatus.UNTAPPED,
                    age_hours=age_hours,
                    strength=strength,
                    sweep_probability=sweep_prob
                )
                liquidity_pools.buy_side_liquidity.append(tf_pool)

        # Sort and limit pools
        liquidity_pools.buy_side_liquidity.sort(key=lambda x: abs(x.level - current_price))
        liquidity_pools.sell_side_liquidity.sort(key=lambda x: abs(x.level - current_price))
        liquidity_pools.buy_side_liquidity = liquidity_pools.buy_side_liquidity[:8]
        liquidity_pools.sell_side_liquidity = liquidity_pools.sell_side_liquidity[:8]

        # Analyze session liquidity
        session_liquidity = self.analyze_session_liquidity(df, current_price)

        # Build comprehensive analysis
        liquidity_analysis = self.build_liquidity_analysis(liquidity_pools, timeframe_levels, session_liquidity,
                                                           current_price)

        # Calculate metrics
        liquidity_metrics = self.calculate_liquidity_metrics(liquidity_pools, current_price)

        return {
            "current_price": current_price,
            "liquidity_pools": liquidity_pools,
            "timeframe_highs_lows": timeframe_levels,
            "session_liquidity": session_liquidity,
            "liquidity_analysis": liquidity_analysis,
            "liquidity_metrics": liquidity_metrics
        }


@router.get("/api/liquidity-zones/{symbol}/{timeframe}",
            response_model=LiquidityZonesResponse,
            summary="Get Liquidity Zones Analysis",
            description="Analyze cryptocurrency liquidity zones using Smart Money Concepts (SMC)")
async def get_liquidity_zones(
        symbol: str = Path(..., description="Trading pair symbol (e.g., BTCUSDT, BTC)"),
        timeframe: str = Path(..., description="Analysis timeframe (e.g., 1h, 4h, 1d)"),
        as_of_datetime: Optional[str] = Query(None,
                                              description="ISO 8601 datetime string for historical analysis (e.g., 2024-01-01T00:00:00Z). Defaults to current time."),
        _: bool = Depends(verify_api_key)
):
    """
    Analyze cryptocurrency liquidity zones using Smart Money Concepts.
    
    This endpoint:
    1. Fetches klines data from Binance for the specified symbol and timeframe
    2. Identifies liquidity pools using equal highs/lows detection
    3. Analyzes multi-timeframe previous highs and lows
    4. Tracks trading session liquidity levels
    5. Calculates sweep probabilities and strength scores
    6. Returns comprehensive liquidity analysis with trading recommendations
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., 'BTC', 'ETH', 'BTCUSDT')
    - **timeframe**: Analysis timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
    - **as_of_datetime**: Optional ISO 8601 datetime string for historical analysis (e.g., '2024-01-01T00:00:00Z'). Defaults to current time.
    
    **Returns comprehensive liquidity analysis including:**
    - Buy-side and sell-side liquidity pools with sweep probabilities
    - Multi-timeframe previous highs and lows analysis
    - Trading session liquidity tracking (Asia/London/NY)
    - Primary and secondary liquidity targets
    - Liquidity imbalance assessment and trading bias
    - Comprehensive metrics and void zone identification
    
    **Note:** If the symbol doesn't end with 'USDT', it will be automatically appended.
    """
    try:
        # Validate timeframe
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
        analyzer = LiquidityZoneAnalyzer()

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

        # Fetch data from Binance
        klines_data = await binance_fetcher.get_klines(
            symbol=processed_symbol,
            interval=timeframe,
            limit=200,  # Get enough data for proper liquidity analysis
            end_time=end_time_ms
        )
        formatted_data = binance_fetcher.format_klines_data(klines_data)

        # Perform liquidity zones analysis
        analysis_result = analyzer.analyze_liquidity_zones(formatted_data, timeframe)

        # Create response
        return LiquidityZonesResponse(
            symbol=processed_symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc).isoformat() if end_time_ms is None else as_of_datetime,
            current_price=analysis_result["current_price"],
            liquidity_pools=analysis_result["liquidity_pools"],
            timeframe_highs_lows=analysis_result["timeframe_highs_lows"],
            session_liquidity=analysis_result["session_liquidity"],
            liquidity_analysis=analysis_result["liquidity_analysis"],
            liquidity_metrics=analysis_result["liquidity_metrics"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Liquidity zones analysis failed: {str(e)}")


@router.get("/api/liquidity-zones/doc",
            summary="Liquidity Zones API Documentation",
            description="Get comprehensive documentation for the Liquidity Zones API response format")
async def get_liquidity_zones_documentation():
    """
    Get detailed documentation for the Liquidity Zones API response format.
    
    This endpoint explains all fields, enums, and data structures returned by the
    /api/liquidity-zones/{symbol}/{timeframe} endpoint.
    """
    return {
        "api_endpoint": "/api/liquidity-zones/{symbol}/{timeframe}",
        "description": "Analyzes liquidity zones using Smart Money Concepts (SMC) to identify areas where institutional orders may be resting, including equal highs/lows, previous timeframe levels, and trading session liquidity.",

        "response_structure": {
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol",
                "example": "BTCUSDT"
            },
            "timeframe": {
                "type": "string",
                "description": "Analysis timeframe",
                "example": "4h"
            },
            "timestamp": {
                "type": "string",
                "description": "ISO 8601 timestamp of analysis",
                "example": "2024-01-01T12:00:00Z"
            },
            "current_price": {
                "type": "number",
                "description": "Current market price",
                "example": 45000.50
            }
        },

        "liquidity_pools": {
            "description": "Buy-side and sell-side liquidity pool collections",
            "fields": {
                "buy_side_liquidity": {
                    "type": "array",
                    "description": "Liquidity pools below current price (support areas where buy orders may rest)",
                    "max_items": 8,
                    "sorted_by": "Distance from current price (nearest first)"
                },
                "sell_side_liquidity": {
                    "type": "array",
                    "description": "Liquidity pools above current price (resistance areas where sell orders may rest)",
                    "max_items": 8,
                    "sorted_by": "Distance from current price (nearest first)"
                }
            },
            "liquidity_pool_fields": {
                "level": {
                    "type": "number",
                    "description": "Exact price level of the liquidity pool",
                    "example": 44200.0
                },
                "distance_from_current": {
                    "type": "string",
                    "description": "Percentage distance from current price",
                    "example": "-1.8%",
                    "note": "Negative values are below current price, positive above"
                },
                "type": {
                    "type": "string",
                    "description": "Type of liquidity pool identified",
                    "possible_values": {
                        "Equal Highs": "Multiple swing highs at same level",
                        "Equal Lows": "Multiple swing lows at same level",
                        "Previous Daily High": "Previous day's high level",
                        "Previous Daily Low": "Previous day's low level",
                        "Previous Weekly High": "Previous week's high level",
                        "Previous Weekly Low": "Previous week's low level"
                    }
                },
                "status": {
                    "type": "enum",
                    "description": "Current status of the liquidity pool",
                    "possible_values": {
                        "UNTAPPED": "Level has not been reached/broken",
                        "SWEPT": "Level has been broken and liquidity taken",
                        "PARTIALLY_SWEPT": "Level partially broken but some liquidity remains"
                    }
                },
                "formation_candles": {
                    "type": "array[integer]",
                    "description": "Candle indices where the liquidity level was formed",
                    "example": [-15, -12, -8],
                    "note": "Negative values indicate candles in the past"
                },
                "sweep_candle": {
                    "type": "integer",
                    "description": "Candle index where level was swept (if status is SWEPT)",
                    "example": -3,
                    "note": "Only present if status is SWEPT or PARTIALLY_SWEPT"
                },
                "age_hours": {
                    "type": "integer",
                    "description": "Hours since the liquidity level was formed",
                    "example": 48
                },
                "strength": {
                    "type": "integer",
                    "description": "Strength score of the liquidity pool",
                    "range": "1-10",
                    "interpretation": {
                        "1-3": "Weak liquidity, may not hold",
                        "4-6": "Moderate liquidity strength",
                        "7-10": "Strong liquidity, high probability of reaction"
                    }
                },
                "sweep_probability": {
                    "type": "enum",
                    "description": "Probability that this level will be swept",
                    "possible_values": {
                        "VERY_HIGH": "85-100% probability of being swept",
                        "HIGH": "65-85% probability of being swept",
                        "MEDIUM": "35-65% probability of being swept",
                        "LOW": "0-35% probability of being swept"
                    }
                }
            }
        },

        "timeframe_highs_lows": {
            "description": "Multi-timeframe previous highs and lows analysis",
            "structure": "Dictionary with timeframe keys (daily, weekly, monthly)",
            "fields": {
                "previous_high": {
                    "type": "number",
                    "description": "Previous timeframe high level",
                    "example": 46500.0,
                    "note": "Null if no previous high available"
                },
                "previous_low": {
                    "type": "number",
                    "description": "Previous timeframe low level",
                    "example": 43200.0,
                    "note": "Null if no previous low available"
                },
                "current_position": {
                    "type": "enum",
                    "description": "Current price position relative to timeframe range",
                    "possible_values": {
                        "ABOVE_RANGE": "Price above previous high",
                        "UPPER_THIRD": "Price in upper third of range",
                        "MID_RANGE": "Price in middle of range",
                        "LOWER_THIRD": "Price in lower third of range",
                        "BELOW_RANGE": "Price below previous low",
                        "INSIDE_RANGE": "Price within the range"
                    }
                },
                "high_distance": {
                    "type": "string",
                    "description": "Percentage distance to previous high",
                    "example": "+3.2%"
                },
                "low_distance": {
                    "type": "string",
                    "description": "Percentage distance to previous low",
                    "example": "-4.1%"
                }
            }
        },

        "session_liquidity": {
            "description": "Trading session liquidity analysis",
            "structure": "Dictionary with session keys (asia, london, new_york)",
            "fields": {
                "high": {
                    "type": "number",
                    "description": "Session high price",
                    "example": 45200.0
                },
                "low": {
                    "type": "number",
                    "description": "Session low price",
                    "example": 44100.0
                },
                "high_swept": {
                    "type": "boolean",
                    "description": "Whether session high has been broken",
                    "interpretation": {
                        "true": "Session high liquidity has been taken",
                        "false": "Session high liquidity still intact"
                    }
                },
                "low_swept": {
                    "type": "boolean",
                    "description": "Whether session low has been broken",
                    "interpretation": {
                        "true": "Session low liquidity has been taken",
                        "false": "Session low liquidity still intact"
                    }
                },
                "session_range": {
                    "type": "number",
                    "description": "Session range in price points",
                    "example": 1100.0,
                    "calculation": "session_high - session_low"
                }
            }
        },

        "liquidity_analysis": {
            "description": "Comprehensive liquidity analysis and trading recommendations",
            "fields": {
                "primary_target": {
                    "type": "object",
                    "description": "Primary liquidity target for price to reach",
                    "fields": {
                        "direction": {
                            "type": "string",
                            "description": "Target direction",
                            "possible_values": {
                                "BUY_SIDE": "Expecting move down to take buy-side liquidity",
                                "SELL_SIDE": "Expecting move up to take sell-side liquidity"
                            }
                        },
                        "level": {
                            "type": "number",
                            "description": "Target price level",
                            "example": 44000.0
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reasoning for this target",
                            "example": "Equal lows with high sweep probability"
                        }
                    }
                },
                "secondary_target": {
                    "type": "object",
                    "description": "Secondary liquidity target (same structure as primary_target)",
                    "note": "May be null if no secondary target identified"
                },
                "liquidity_imbalance": {
                    "type": "enum",
                    "description": "Overall distribution of liquidity pools",
                    "possible_values": {
                        "SELL_SIDE_HEAVY": "More liquidity above price, expect upward move",
                        "BUY_SIDE_HEAVY": "More liquidity below price, expect downward move",
                        "BALANCED": "Roughly equal liquidity distribution"
                    }
                },
                "recommended_bias": {
                    "type": "enum",
                    "description": "Recommended trading bias based on liquidity analysis",
                    "possible_values": {
                        "BULLISH_SWEEP_LIKELY": "Expect upward move to take sell-side liquidity",
                        "BEARISH_SWEEP_LIKELY": "Expect downward move to take buy-side liquidity",
                        "RANGE_BOUND": "Price likely to stay within current range",
                        "BREAKOUT_PENDING": "Expect significant move but direction unclear"
                    }
                },
                "key_message": {
                    "type": "string",
                    "description": "Summary message with key liquidity insights",
                    "example": "Price targeting buy-side liquidity at 44000. Watch for rejection at current level."
                }
            }
        },

        "liquidity_metrics": {
            "description": "Quantitative liquidity metrics and summary statistics",
            "fields": {
                "total_buy_side_pools": {
                    "type": "integer",
                    "description": "Number of buy-side liquidity pools identified",
                    "example": 5
                },
                "total_sell_side_pools": {
                    "type": "integer",
                    "description": "Number of sell-side liquidity pools identified",
                    "example": 3
                },
                "nearest_untapped_above": {
                    "type": "number",
                    "description": "Nearest untapped liquidity level above current price",
                    "example": 45500.0,
                    "note": "Null if no untapped liquidity above"
                },
                "nearest_untapped_below": {
                    "type": "number",
                    "description": "Nearest untapped liquidity level below current price",
                    "example": 43800.0,
                    "note": "Null if no untapped liquidity below"
                },
                "liquidity_void_zones": {
                    "type": "array",
                    "description": "Areas with thin liquidity where price may move quickly",
                    "item_fields": {
                        "range": {
                            "type": "array[number]",
                            "description": "Price range of void zone [low, high]",
                            "example": [44800, 45100]
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the void zone",
                            "example": "Gap between equal highs and session resistance"
                        }
                    }
                }
            }
        },

        "trading_concepts": {
            "liquidity_hunt": {
                "description": "Market makers often target areas of high liquidity concentration to fill large orders",
                "examples": [
                    "Equal highs/lows where retail stops cluster",
                    "Previous day/week highs and lows",
                    "Round numbers and psychological levels"
                ]
            },
            "sweep_patterns": {
                "description": "Common patterns when liquidity is taken",
                "patterns": {
                    "liquidity_grab": "Quick move to take liquidity then reverse",
                    "stop_hunt": "Temporary move beyond level to trigger stops",
                    "true_breakout": "Sustained move beyond level indicating trend change"
                }
            },
            "session_importance": {
                "asia": "Often sets the range for the day, liquidity at session highs/lows",
                "london": "Major volatility expansion, targets Asian range",
                "new_york": "Continuation or reversal of London moves, targets session liquidity"
            }
        },

        "usage_examples": {
            "bullish_liquidity_setup": {
                "description": "Example of bullish liquidity targeting",
                "indicators": [
                    "primary_target.direction: SELL_SIDE",
                    "recommended_bias: BULLISH_SWEEP_LIKELY",
                    "liquidity_imbalance: SELL_SIDE_HEAVY",
                    "Multiple untapped sell-side pools above"
                ]
            },
            "bearish_liquidity_setup": {
                "description": "Example of bearish liquidity targeting",
                "indicators": [
                    "primary_target.direction: BUY_SIDE",
                    "recommended_bias: BEARISH_SWEEP_LIKELY",
                    "liquidity_imbalance: BUY_SIDE_HEAVY",
                    "Equal lows with HIGH sweep_probability"
                ]
            }
        },

        "trading_interpretation": {
            "how_to_use": [
                "1. Check liquidity_imbalance for overall bias direction",
                "2. Identify primary_target for likely price destination",
                "3. Use nearest untapped levels for short-term targets",
                "4. Monitor session liquidity for intraday targets",
                "5. Higher strength pools are more likely to cause reactions",
                "6. VERY_HIGH sweep probability levels are prime targets"
            ],
            "risk_management": [
                "Avoid trading into liquidity void zones",
                "Place stops beyond strong liquidity clusters",
                "Expect quick moves through low-strength liquidity",
                "Be cautious of false breakouts at high-strength levels"
            ],
            "confluence_factors": [
                "Multiple liquidity pools at same level increase importance",
                "Session liquidity + timeframe levels = high probability targets",
                "Equal highs/lows with recent formation are strongest",
                "Untapped levels from higher timeframes take priority"
            ]
        }
    }
