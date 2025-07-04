from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field

from binance_data import BinanceDataFetcher
from smc_analysis import SMCAnalyzer
from utils import verify_api_key


class OrderBlockStatus(str, Enum):
    FRESH = "FRESH"
    TESTED_HOLDING = "TESTED_HOLDING"
    TESTED_BROKEN = "TESTED_BROKEN"
    MITIGATED = "MITIGATED"


class VolumeProfile(str, Enum):
    EXTREME = "EXTREME"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class ZoneImbalance(str, Enum):
    DEMAND_HEAVY = "DEMAND_HEAVY"
    SUPPLY_HEAVY = "SUPPLY_HEAVY"
    BALANCED = "BALANCED"


class PremiumDiscount(str, Enum):
    PREMIUM = "PREMIUM"
    EQUILIBRIUM = "EQUILIBRIUM"
    DISCOUNT = "DISCOUNT"


class FVGStatus(str, Enum):
    FRESH = "FRESH"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FULLY_FILLED = "FULLY_FILLED"


class BreakerStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class OrderBlock(BaseModel):
    top: float = Field(..., description="Top of the order block")
    bottom: float = Field(..., description="Bottom of the order block")
    ob_type: str = Field(..., description="Type of order block (Bullish OB/Bearish OB)")
    formation_index: int = Field(..., description="Candle index where OB was formed")
    age_hours: int = Field(..., description="Age of order block in hours")
    status: OrderBlockStatus = Field(..., description="Current status of order block")
    test_count: int = Field(..., description="Number of times OB was tested", ge=0)
    volume_profile: VolumeProfile = Field(..., description="Volume profile during formation")
    strength: int = Field(..., description="Strength score 1-10", ge=1, le=10)
    distance_from_current: str = Field(..., description="Distance from current price as percentage")
    zone_height: float = Field(..., description="Height of the zone in price points")
    confluence: List[str] = Field(..., description="List of confluences with other levels")


class OrderBlocks(BaseModel):
    bullish_ob: List[OrderBlock] = Field(..., description="Bullish order blocks (demand zones)")
    bearish_ob: List[OrderBlock] = Field(..., description="Bearish order blocks (supply zones)")


class FairValueGap(BaseModel):
    top: float = Field(..., description="Top of the fair value gap")
    bottom: float = Field(..., description="Bottom of the fair value gap")
    type: str = Field(..., description="Type of FVG (Bullish FVG/Bearish FVG)")
    formation_index: int = Field(..., description="Candle index where FVG was formed")
    age_hours: int = Field(..., description="Age of FVG in hours")
    status: FVGStatus = Field(..., description="Current status of FVG")
    mitigation_index: Optional[int] = Field(None, description="Candle index where FVG was mitigated")
    fill_percentage: float = Field(..., description="Percentage of gap that has been filled", ge=0, le=100)
    distance_from_current: str = Field(..., description="Distance from current price as percentage")
    gap_size: float = Field(..., description="Size of the gap in price points")


class FairValueGaps(BaseModel):
    bullish_fvg: List[FairValueGap] = Field(..., description="Bullish fair value gaps")
    bearish_fvg: List[FairValueGap] = Field(..., description="Bearish fair value gaps")


class BreakerBlock(BaseModel):
    level: float = Field(..., description="Price level of breaker block")
    type: str = Field(..., description="Type of breaker (Bullish Breaker/Bearish Breaker)")
    original_ob_type: str = Field(..., description="Original order block type before break")
    break_candle_index: int = Field(..., description="Candle index where OB was broken")
    age_hours: int = Field(..., description="Age since break occurred")
    status: BreakerStatus = Field(..., description="Current status of breaker block")
    strength: int = Field(..., description="Strength score 1-10", ge=1, le=10)
    distance_from_current: str = Field(..., description="Distance from current price as percentage")
    description: str = Field(..., description="Description of the breaker block")


class ZoneTarget(BaseModel):
    zone: List[float] = Field(..., description="Zone range [bottom, top]")
    type: str = Field(..., description="Type of zone")
    strength: int = Field(..., description="Strength score", ge=1, le=10)
    distance: str = Field(..., description="Distance from current price")
    recommendation: str = Field(..., description="Trading recommendation")


class ZoneAnalysisData(BaseModel):
    nearest_demand_zone: Optional[ZoneTarget] = Field(None, description="Nearest demand zone below price")
    nearest_supply_zone: Optional[ZoneTarget] = Field(None, description="Nearest supply zone above price")
    strongest_zone: Optional[ZoneTarget] = Field(None, description="Strongest zone regardless of position")
    zone_density: ZoneImbalance = Field(..., description="Overall zone distribution")
    premium_discount: PremiumDiscount = Field(..., description="Current market positioning")
    key_message: str = Field(..., description="Key takeaway message")


class ConfluenceZone(BaseModel):
    zone: List[float] = Field(..., description="Zone range [bottom, top]")
    confluences: List[str] = Field(..., description="List of confluences")
    score: float = Field(..., description="Confluence score", ge=0, le=10)


class ConfluenceMatrix(BaseModel):
    high_probability_zones: List[ConfluenceZone] = Field(..., description="High probability zones with confluences")


class SupplyDemandMetrics(BaseModel):
    total_demand_zones: int = Field(..., description="Total number of demand zones", ge=0)
    total_supply_zones: int = Field(..., description="Total number of supply zones", ge=0)
    fresh_zones_count: int = Field(..., description="Number of fresh/untested zones", ge=0)
    mitigated_today: int = Field(..., description="Zones mitigated in recent period", ge=0)
    zone_imbalance: ZoneImbalance = Field(..., description="Overall zone balance")
    avg_zone_height: float = Field(..., description="Average zone height in price points")
    strongest_zone_distance: str = Field(..., description="Distance to strongest zone")


class SupplyDemandZonesResponse(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Analysis timeframe")
    timestamp: str = Field(..., description="Analysis timestamp")
    current_price: float = Field(..., description="Current price")
    order_blocks: OrderBlocks = Field(..., description="Order blocks analysis")
    fair_value_gaps: FairValueGaps = Field(..., description="Fair value gaps analysis")
    breaker_blocks: List[BreakerBlock] = Field(..., description="Breaker blocks analysis")
    zone_analysis: ZoneAnalysisData = Field(..., description="Comprehensive zone analysis")
    confluence_matrix: ConfluenceMatrix = Field(..., description="Confluence analysis")
    supply_demand_metrics: SupplyDemandMetrics = Field(..., description="Supply demand metrics")


# Create router
router = APIRouter(tags=["Supply Demand Zones"])


class SupplyDemandAnalyzer(SMCAnalyzer):
    """
    Supply Demand Zone Analysis using smartmoneyconcepts library
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

    def determine_volume_profile(self, volume_data: List[float], ob_index: int) -> VolumeProfile:
        """
        Determine volume profile during order block formation
        """
        if not volume_data or ob_index >= len(volume_data):
            return VolumeProfile.MODERATE

        # Get volume at OB formation and surrounding candles
        start_idx = max(0, ob_index - 2)
        end_idx = min(len(volume_data), ob_index + 3)
        ob_volumes = volume_data[start_idx:end_idx]

        if not ob_volumes:
            return VolumeProfile.MODERATE

        ob_volume = volume_data[ob_index]
        avg_volume = sum(volume_data) / len(volume_data)

        # Calculate volume relative to average
        volume_ratio = ob_volume / avg_volume if avg_volume > 0 else 1

        if volume_ratio >= 3.0:
            return VolumeProfile.EXTREME
        elif volume_ratio >= 2.0:
            return VolumeProfile.HIGH
        elif volume_ratio >= 1.2:
            return VolumeProfile.MODERATE
        else:
            return VolumeProfile.LOW

    def calculate_zone_strength(self, ob_data: Dict, volume_profile: VolumeProfile,
                                test_count: int, age_hours: int) -> int:
        """
        Calculate zone strength score (1-10)
        """
        # Volume factor (40% weight)
        volume_scores = {
            VolumeProfile.EXTREME: 4,
            VolumeProfile.HIGH: 3,
            VolumeProfile.MODERATE: 2,
            VolumeProfile.LOW: 1
        }
        volume_score = volume_scores[volume_profile]

        # Test factor (30% weight) - fewer tests = stronger
        if test_count == 0:
            test_score = 3
        elif test_count == 1:
            test_score = 2
        else:
            test_score = 1

        # Formation method (20% weight)
        zone_height = abs(ob_data.get('top', 0) - ob_data.get('bottom', 0))
        if zone_height > 0:
            formation_score = 2  # Sharp rejection
        else:
            formation_score = 1  # Gradual formation

        # Age factor (10% weight) - slight preference for recent zones
        if age_hours <= 24:
            age_score = 1
        else:
            age_score = 0

        total_score = volume_score + test_score + formation_score + age_score
        return min(10, max(1, total_score))

    def find_confluences(self, zone_level: float, current_price: float,
                         swing_highs_lows: pd.DataFrame,
                         timeframe_levels: Dict) -> List[str]:
        """
        Find confluences with other technical levels
        """
        confluences = []
        tolerance = current_price * 0.005  # 0.5% tolerance

        # Check swing high/low confluences
        if not swing_highs_lows.empty:
            for _, row in swing_highs_lows.iterrows():
                level = row.get('Level', 0)
                if abs(zone_level - level) <= tolerance:
                    if row.get('HighLow', 0) == 1:
                        confluences.append("Swing High")
                    elif row.get('HighLow', 0) == -1:
                        confluences.append("Swing Low")

        # Check previous high/low confluences
        for tf_name, tf_data in timeframe_levels.items():
            if hasattr(tf_data, 'previous_high') and tf_data.previous_high:
                if abs(zone_level - tf_data.previous_high) <= tolerance:
                    confluences.append(f"Previous {tf_name.title()} High")
            if hasattr(tf_data, 'previous_low') and tf_data.previous_low:
                if abs(zone_level - tf_data.previous_low) <= tolerance:
                    confluences.append(f"Previous {tf_name.title()} Low")

        # Add generic confluences if none found
        if not confluences:
            if zone_level > current_price:
                confluences.append("Resistance Level")
            else:
                confluences.append("Support Level")

        return confluences

    def determine_ob_status(self, ob_data: Dict, current_price: float, df: pd.DataFrame,
                            formation_index: int) -> tuple[OrderBlockStatus, int]:
        """
        Determine order block status and test count
        """
        top = ob_data.get('top', 0)
        bottom = ob_data.get('bottom', 0)
        ob_direction = ob_data.get('direction', 0)

        test_count = 0

        # Check if price has interacted with the OB since formation
        if formation_index < len(df) - 1:
            subsequent_data = df.iloc[formation_index + 1:]

            for _, candle in subsequent_data.iterrows():
                high = candle['high']
                low = candle['low']
                close = candle['close']

                # Check if price entered the OB zone
                if low <= top and high >= bottom:
                    test_count += 1

                    # For bullish OB, check if price closed below bottom (mitigation)
                    if ob_direction == 1 and close < bottom:
                        return OrderBlockStatus.MITIGATED, test_count

                    # For bearish OB, check if price closed above top (mitigation)
                    if ob_direction == -1 and close > top:
                        return OrderBlockStatus.MITIGATED, test_count

        # Determine status based on tests
        if test_count == 0:
            return OrderBlockStatus.FRESH, 0
        elif test_count == 1:
            return OrderBlockStatus.TESTED_HOLDING, test_count
        else:
            return OrderBlockStatus.TESTED_BROKEN, test_count

    def analyze_order_blocks(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame,
                             current_price: float, timeframe: str,
                             timeframe_levels: Dict) -> OrderBlocks:
        """
        Analyze order blocks using smartmoneyconcepts
        """
        try:
            # Get order blocks data using inherited method
            ob_data = self.calculate_order_blocks(df, swing_highs_lows, close_mitigation=False)

            bullish_obs = []
            bearish_obs = []

            if ob_data and (ob_data["bullish_ob"] or ob_data["bearish_ob"]):
                volume_data = df['volume'].tolist()

                # Process bullish order blocks
                for ob in ob_data["bullish_ob"]:
                    try:
                        # Calculate formation index
                        timestamp_str = ob.get("timestamp", "")
                        if timestamp_str:
                            timestamp = pd.to_datetime(timestamp_str)
                            if timestamp in df.index:
                                formation_index = df.index.get_loc(timestamp)
                                candle_index = -(len(df) - formation_index - 1)
                            else:
                                formation_index = len(df) - 10
                                candle_index = -10
                        else:
                            formation_index = len(df) - 10
                            candle_index = -10

                        age_hours = self.get_hours_ago(candle_index, timeframe)

                        # Get OB details
                        top = self._convert_numpy_types(ob.get('top', current_price))
                        bottom = self._convert_numpy_types(ob.get('bottom', current_price))
                        zone_height = abs(top - bottom)

                        # Determine volume profile
                        volume_profile = self.determine_volume_profile(volume_data, formation_index)

                        # Determine status and test count
                        status, test_count = self.determine_ob_status(
                            {'top': top, 'bottom': bottom, 'direction': 1},
                            current_price, df, formation_index
                        )

                        # Find confluences
                        confluences = self.find_confluences(
                            (top + bottom) / 2, current_price, swing_highs_lows, timeframe_levels
                        )

                        # Calculate strength
                        strength = self.calculate_zone_strength(
                            {'top': top, 'bottom': bottom}, volume_profile, test_count, age_hours
                        )

                        bullish_ob = OrderBlock(
                            top=top,
                            bottom=bottom,
                            ob_type="Bullish OB",
                            formation_index=candle_index,
                            age_hours=age_hours,
                            status=status,
                            test_count=test_count,
                            volume_profile=volume_profile,
                            strength=strength,
                            distance_from_current=self.calculate_distance_percentage(
                                (top + bottom) / 2, current_price
                            ),
                            zone_height=zone_height,
                            confluence=confluences
                        )
                        bullish_obs.append(bullish_ob)

                    except Exception as e:
                        print(f"Error processing bullish OB: {e}")
                        continue

                # Process bearish order blocks
                for ob in ob_data["bearish_ob"]:
                    try:
                        # Calculate formation index
                        timestamp_str = ob.get("timestamp", "")
                        if timestamp_str:
                            timestamp = pd.to_datetime(timestamp_str)
                            if timestamp in df.index:
                                formation_index = df.index.get_loc(timestamp)
                                candle_index = -(len(df) - formation_index - 1)
                            else:
                                formation_index = len(df) - 10
                                candle_index = -10
                        else:
                            formation_index = len(df) - 10
                            candle_index = -10

                        age_hours = self.get_hours_ago(candle_index, timeframe)

                        # Get OB details
                        top = self._convert_numpy_types(ob.get('top', current_price))
                        bottom = self._convert_numpy_types(ob.get('bottom', current_price))
                        zone_height = abs(top - bottom)

                        # Determine volume profile
                        volume_profile = self.determine_volume_profile(volume_data, formation_index)

                        # Determine status and test count
                        status, test_count = self.determine_ob_status(
                            {'top': top, 'bottom': bottom, 'direction': -1},
                            current_price, df, formation_index
                        )

                        # Find confluences
                        confluences = self.find_confluences(
                            (top + bottom) / 2, current_price, swing_highs_lows, timeframe_levels
                        )

                        # Calculate strength
                        strength = self.calculate_zone_strength(
                            {'top': top, 'bottom': bottom}, volume_profile, test_count, age_hours
                        )

                        bearish_ob = OrderBlock(
                            top=top,
                            bottom=bottom,
                            ob_type="Bearish OB",
                            formation_index=candle_index,
                            age_hours=age_hours,
                            status=status,
                            test_count=test_count,
                            volume_profile=volume_profile,
                            strength=strength,
                            distance_from_current=self.calculate_distance_percentage(
                                (top + bottom) / 2, current_price
                            ),
                            zone_height=zone_height,
                            confluence=confluences
                        )
                        bearish_obs.append(bearish_ob)

                    except Exception as e:
                        print(f"Error processing bearish OB: {e}")
                        continue

            # Sort and limit results
            bullish_obs.sort(key=lambda x: abs((x.top + x.bottom) / 2 - current_price))
            bearish_obs.sort(key=lambda x: abs((x.top + x.bottom) / 2 - current_price))

            return OrderBlocks(
                bullish_ob=bullish_obs[:10],
                bearish_ob=bearish_obs[:10]
            )

        except Exception as e:
            print(f"Error in order blocks analysis: {e}")
            return OrderBlocks(bullish_ob=[], bearish_ob=[])

    def calculate_fvg_fill_percentage(self, fvg_data: Dict, current_price: float,
                                      df: pd.DataFrame, formation_index: int) -> float:
        """
        Calculate how much of the FVG has been filled
        """
        top = fvg_data.get('top', 0)
        bottom = fvg_data.get('bottom', 0)
        gap_size = abs(top - bottom)

        if gap_size == 0:
            return 100.0

        # Check subsequent price action
        if formation_index < len(df) - 1:
            subsequent_data = df.iloc[formation_index + 1:]

            max_penetration = 0
            for _, candle in subsequent_data.iterrows():
                high = candle['high']
                low = candle['low']

                # Calculate penetration into the gap
                if bottom < top:  # Bullish FVG
                    if high > bottom:
                        penetration = min(high, top) - bottom
                        max_penetration = max(max_penetration, penetration)
                else:  # Bearish FVG
                    if low < top:
                        penetration = top - max(low, bottom)
                        max_penetration = max(max_penetration, penetration)

            fill_percentage = (max_penetration / gap_size) * 100
            return min(100.0, max(0.0, fill_percentage))

        return 0.0

    def analyze_fair_value_gaps(self, df: pd.DataFrame, current_price: float,
                                timeframe: str) -> FairValueGaps:
        """
        Analyze fair value gaps using smartmoneyconcepts
        """
        try:
            # Get FVG data using inherited method
            fvg_data = self.calculate_fair_value_gaps(df, join_consecutive=False)

            bullish_fvgs = []
            bearish_fvgs = []

            if fvg_data and (fvg_data["bullish_fvg"] or fvg_data["bearish_fvg"]):

                # Process bullish FVGs
                for fvg in fvg_data["bullish_fvg"]:
                    try:
                        # Calculate formation index
                        timestamp_str = fvg.get("timestamp", "")
                        if timestamp_str:
                            timestamp = pd.to_datetime(timestamp_str)
                            if timestamp in df.index:
                                formation_index = df.index.get_loc(timestamp)
                                candle_index = -(len(df) - formation_index - 1)
                            else:
                                formation_index = len(df) - 10
                                candle_index = -10
                        else:
                            formation_index = len(df) - 10
                            candle_index = -10

                        age_hours = self.get_hours_ago(candle_index, timeframe)

                        # Get FVG details
                        top = self._convert_numpy_types(fvg.get('top', current_price))
                        bottom = self._convert_numpy_types(fvg.get('bottom', current_price))
                        gap_size = abs(top - bottom)

                        # Calculate fill percentage
                        fill_percentage = self.calculate_fvg_fill_percentage(
                            {'top': top, 'bottom': bottom}, current_price, df, formation_index
                        )

                        # Determine status
                        if fill_percentage >= 100:
                            status = FVGStatus.FULLY_FILLED
                        elif fill_percentage > 0:
                            status = FVGStatus.PARTIALLY_FILLED
                        else:
                            status = FVGStatus.FRESH

                        # Mitigation index (if mitigated)
                        mitigation_index = None
                        if fill_percentage >= 50:
                            mitigation_index = candle_index + int(age_hours / 2)  # Approximate

                        bullish_fvg = FairValueGap(
                            top=top,
                            bottom=bottom,
                            type="Bullish FVG",
                            formation_index=candle_index,
                            age_hours=age_hours,
                            status=status,
                            mitigation_index=mitigation_index,
                            fill_percentage=fill_percentage,
                            distance_from_current=self.calculate_distance_percentage(
                                (top + bottom) / 2, current_price
                            ),
                            gap_size=gap_size
                        )
                        bullish_fvgs.append(bullish_fvg)

                    except Exception as e:
                        print(f"Error processing bullish FVG: {e}")
                        continue

                # Process bearish FVGs
                for fvg in fvg_data["bearish_fvg"]:
                    try:
                        # Calculate formation index
                        timestamp_str = fvg.get("timestamp", "")
                        if timestamp_str:
                            timestamp = pd.to_datetime(timestamp_str)
                            if timestamp in df.index:
                                formation_index = df.index.get_loc(timestamp)
                                candle_index = -(len(df) - formation_index - 1)
                            else:
                                formation_index = len(df) - 10
                                candle_index = -10
                        else:
                            formation_index = len(df) - 10
                            candle_index = -10

                        age_hours = self.get_hours_ago(candle_index, timeframe)

                        # Get FVG details
                        top = self._convert_numpy_types(fvg.get('top', current_price))
                        bottom = self._convert_numpy_types(fvg.get('bottom', current_price))
                        gap_size = abs(top - bottom)

                        # Calculate fill percentage
                        fill_percentage = self.calculate_fvg_fill_percentage(
                            {'top': top, 'bottom': bottom}, current_price, df, formation_index
                        )

                        # Determine status
                        if fill_percentage >= 100:
                            status = FVGStatus.FULLY_FILLED
                        elif fill_percentage > 0:
                            status = FVGStatus.PARTIALLY_FILLED
                        else:
                            status = FVGStatus.FRESH

                        # Mitigation index (if mitigated)
                        mitigation_index = None
                        if fill_percentage >= 50:
                            mitigation_index = candle_index + int(age_hours / 2)  # Approximate

                        bearish_fvg = FairValueGap(
                            top=top,
                            bottom=bottom,
                            type="Bearish FVG",
                            formation_index=candle_index,
                            age_hours=age_hours,
                            status=status,
                            mitigation_index=mitigation_index,
                            fill_percentage=fill_percentage,
                            distance_from_current=self.calculate_distance_percentage(
                                (top + bottom) / 2, current_price
                            ),
                            gap_size=gap_size
                        )
                        bearish_fvgs.append(bearish_fvg)

                    except Exception as e:
                        print(f"Error processing bearish FVG: {e}")
                        continue

            # Sort by distance from current price
            bullish_fvgs.sort(key=lambda x: abs((x.top + x.bottom) / 2 - current_price))
            bearish_fvgs.sort(key=lambda x: abs((x.top + x.bottom) / 2 - current_price))

            return FairValueGaps(
                bullish_fvg=bullish_fvgs[:8],
                bearish_fvg=bearish_fvgs[:8]
            )

        except Exception as e:
            print(f"Error in FVG analysis: {e}")
            return FairValueGaps(bullish_fvg=[], bearish_fvg=[])

    def identify_breaker_blocks(self, order_blocks: OrderBlocks, df: pd.DataFrame,
                                swing_highs_lows: pd.DataFrame, current_price: float,
                                timeframe: str) -> List[BreakerBlock]:
        """
        Identify breaker blocks from broken order blocks
        """
        try:
            # Get BOS/CHoCH data
            bos_choch_data = self.calculate_bos_choch(df, swing_highs_lows, close_break=True)

            breaker_blocks = []

            if not bos_choch_data or not (bos_choch_data["bos"] or bos_choch_data["choch"]):
                return breaker_blocks

            # Combine all structural breaks
            all_breaks = bos_choch_data["bos"] + bos_choch_data["choch"]

            # Check each order block for breaks
            all_obs = order_blocks.bullish_ob + order_blocks.bearish_ob

            for ob in all_obs:
                if ob.status == OrderBlockStatus.MITIGATED:
                    # This OB was broken, check if it was due to a structural break
                    ob_center = (ob.top + ob.bottom) / 2

                    for break_event in all_breaks:
                        break_level = break_event.get('level', 0)

                        # Check if break level is close to OB level
                        if abs(break_level - ob_center) <= current_price * 0.01:  # 1% tolerance

                            # Determine breaker type (opposite of original OB)
                            if ob.ob_type == "Bullish OB":
                                breaker_type = "Bearish Breaker"
                                description = "Former demand zone turned resistance after BOS"
                            else:
                                breaker_type = "Bullish Breaker"
                                description = "Former supply zone turned support after BOS"

                            # Calculate break timing
                            break_candle_index = ob.formation_index + ob.age_hours // self.get_hours_ago(1, timeframe)
                            break_age_hours = self.get_hours_ago(break_candle_index, timeframe)

                            # Determine breaker strength (reduced from original OB)
                            breaker_strength = max(1, ob.strength - 2)

                            # Determine status
                            status = BreakerStatus.ACTIVE if break_age_hours <= 168 else BreakerStatus.INACTIVE

                            breaker = BreakerBlock(
                                level=ob_center,
                                type=breaker_type,
                                original_ob_type=ob.ob_type,
                                break_candle_index=break_candle_index,
                                age_hours=break_age_hours,
                                status=status,
                                strength=breaker_strength,
                                distance_from_current=self.calculate_distance_percentage(
                                    ob_center, current_price
                                ),
                                description=description
                            )
                            breaker_blocks.append(breaker)
                            break  # Only create one breaker per OB

            # Sort by distance from current price
            breaker_blocks.sort(key=lambda x: abs(x.level - current_price))

            return breaker_blocks[:5]  # Limit to 5 most relevant breakers

        except Exception as e:
            print(f"Error identifying breaker blocks: {e}")
            return []

    def determine_premium_discount(self, current_price: float, order_blocks: OrderBlocks) -> PremiumDiscount:
        """
        Determine if market is in premium, discount, or equilibrium
        """
        all_obs = order_blocks.bullish_ob + order_blocks.bearish_ob
        if not all_obs:
            return PremiumDiscount.EQUILIBRIUM

        # Find range of significant zones
        levels = [(ob.top + ob.bottom) / 2 for ob in all_obs if ob.strength >= 6]
        if len(levels) < 2:
            return PremiumDiscount.EQUILIBRIUM

        range_high = max(levels)
        range_low = min(levels)
        range_size = range_high - range_low

        if range_size == 0:
            return PremiumDiscount.EQUILIBRIUM

        position = (current_price - range_low) / range_size

        if position > 0.7:
            return PremiumDiscount.PREMIUM
        elif position < 0.3:
            return PremiumDiscount.DISCOUNT
        else:
            return PremiumDiscount.EQUILIBRIUM

    def build_zone_analysis(self, order_blocks: OrderBlocks, fair_value_gaps: FairValueGaps,
                            breaker_blocks: List[BreakerBlock], current_price: float) -> ZoneAnalysisData:
        """
        Build comprehensive zone analysis
        """
        # Find nearest demand zone (below current price)
        demand_zones = []
        for ob in order_blocks.bullish_ob:
            if (ob.top + ob.bottom) / 2 < current_price:
                demand_zones.append(ob)

        nearest_demand_zone = None
        if demand_zones:
            nearest_ob = min(demand_zones, key=lambda x: abs((x.top + x.bottom) / 2 - current_price))
            nearest_demand_zone = ZoneTarget(
                zone=[nearest_ob.bottom, nearest_ob.top],
                type=nearest_ob.ob_type,
                strength=nearest_ob.strength,
                distance=nearest_ob.distance_from_current,
                recommendation="Strong support zone, expect bounce" if nearest_ob.strength >= 7 else "Support zone, watch for reaction"
            )

        # Find nearest supply zone (above current price)
        supply_zones = []
        for ob in order_blocks.bearish_ob:
            if (ob.top + ob.bottom) / 2 > current_price:
                supply_zones.append(ob)

        nearest_supply_zone = None
        if supply_zones:
            nearest_ob = min(supply_zones, key=lambda x: abs((x.top + x.bottom) / 2 - current_price))
            nearest_supply_zone = ZoneTarget(
                zone=[nearest_ob.bottom, nearest_ob.top],
                type=nearest_ob.ob_type,
                strength=nearest_ob.strength,
                distance=nearest_ob.distance_from_current,
                recommendation="Strong resistance zone, expect rejection" if nearest_ob.strength >= 7 else "Resistance zone, watch for reaction"
            )

        # Find strongest zone overall
        all_obs = order_blocks.bullish_ob + order_blocks.bearish_ob
        strongest_zone = None
        if all_obs:
            strongest_ob = max(all_obs, key=lambda x: x.strength)
            reason_parts = []
            if strongest_ob.volume_profile in [VolumeProfile.EXTREME, VolumeProfile.HIGH]:
                reason_parts.append(f"{strongest_ob.volume_profile.lower()} volume")
            if strongest_ob.status == OrderBlockStatus.FRESH:
                reason_parts.append("untested")
            if len(strongest_ob.confluence) > 1:
                reason_parts.append("multiple confluences")

            reason = ", ".join(reason_parts) if reason_parts else "high strength zone"

            strongest_zone = ZoneTarget(
                zone=[strongest_ob.bottom, strongest_ob.top],
                type=strongest_ob.ob_type,
                strength=strongest_ob.strength,
                distance=strongest_ob.distance_from_current,
                recommendation=f"Key zone: {reason}"
            )

        # Determine zone density
        demand_count = len([ob for ob in order_blocks.bullish_ob if ob.status != OrderBlockStatus.MITIGATED])
        supply_count = len([ob for ob in order_blocks.bearish_ob if ob.status != OrderBlockStatus.MITIGATED])

        if demand_count > supply_count + 1:
            zone_density = ZoneImbalance.DEMAND_HEAVY
        elif supply_count > demand_count + 1:
            zone_density = ZoneImbalance.SUPPLY_HEAVY
        else:
            zone_density = ZoneImbalance.BALANCED

        # Determine premium/discount
        premium_discount = self.determine_premium_discount(current_price, order_blocks)

        # Generate key message
        if nearest_demand_zone and nearest_supply_zone:
            key_message = f"Price between key zones. Watch for reaction at {nearest_demand_zone.zone[0]:.0f}-{nearest_demand_zone.zone[1]:.0f} support and {nearest_supply_zone.zone[0]:.0f}-{nearest_supply_zone.zone[1]:.0f} resistance."
        elif nearest_demand_zone:
            key_message = f"Above key support at {nearest_demand_zone.zone[0]:.0f}-{nearest_demand_zone.zone[1]:.0f}. Watch for bounce or break."
        elif nearest_supply_zone:
            key_message = f"Below key resistance at {nearest_supply_zone.zone[0]:.0f}-{nearest_supply_zone.zone[1]:.0f}. Watch for rejection or break."
        else:
            key_message = "No significant supply/demand zones nearby. Market in transition."

        return ZoneAnalysisData(
            nearest_demand_zone=nearest_demand_zone,
            nearest_supply_zone=nearest_supply_zone,
            strongest_zone=strongest_zone,
            zone_density=zone_density,
            premium_discount=premium_discount,
            key_message=key_message
        )

    def create_confluence_matrix(self, order_blocks: OrderBlocks, fair_value_gaps: FairValueGaps,
                                 breaker_blocks: List[BreakerBlock]) -> ConfluenceMatrix:
        """
        Create confluence matrix for high probability zones
        """
        high_probability_zones = []

        # Analyze order blocks for confluences
        all_obs = order_blocks.bullish_ob + order_blocks.bearish_ob
        for ob in all_obs:
            if ob.strength >= 7:  # Only high strength zones
                confluences = [ob.ob_type]
                score = ob.strength

                # Add status-based confluences
                if ob.status == OrderBlockStatus.FRESH:
                    confluences.append("Untested Zone")
                    score += 0.5
                elif ob.status == OrderBlockStatus.TESTED_HOLDING:
                    confluences.append("Tested Support/Resistance")
                    score += 0.3

                # Add volume-based confluences
                if ob.volume_profile == VolumeProfile.EXTREME:
                    confluences.append("Extreme Volume")
                    score += 1.0
                elif ob.volume_profile == VolumeProfile.HIGH:
                    confluences.append("High Volume")
                    score += 0.5

                # Add existing confluences
                confluences.extend(ob.confluence)

                # Check for FVG overlaps
                for fvg in fair_value_gaps.bullish_fvg + fair_value_gaps.bearish_fvg:
                    if (fvg.bottom <= ob.top and fvg.top >= ob.bottom):
                        confluences.append("FVG Overlap")
                        score += 0.3
                        break

                confluence_zone = ConfluenceZone(
                    zone=[ob.bottom, ob.top],
                    confluences=list(set(confluences)),  # Remove duplicates
                    score=min(10.0, score)
                )
                high_probability_zones.append(confluence_zone)

        # Sort by score
        high_probability_zones.sort(key=lambda x: x.score, reverse=True)

        return ConfluenceMatrix(
            high_probability_zones=high_probability_zones[:8]  # Top 8 zones
        )

    def calculate_supply_demand_metrics(self, order_blocks: OrderBlocks, fair_value_gaps: FairValueGaps,
                                        breaker_blocks: List[BreakerBlock],
                                        current_price: float) -> SupplyDemandMetrics:
        """
        Calculate comprehensive supply demand metrics
        """
        # Count zones
        demand_zones = [ob for ob in order_blocks.bullish_ob if ob.status != OrderBlockStatus.MITIGATED]
        supply_zones = [ob for ob in order_blocks.bearish_ob if ob.status != OrderBlockStatus.MITIGATED]

        total_demand_zones = len(demand_zones)
        total_supply_zones = len(supply_zones)

        # Count fresh zones
        fresh_zones = []
        fresh_zones.extend([ob for ob in order_blocks.bullish_ob if ob.status == OrderBlockStatus.FRESH])
        fresh_zones.extend([ob for ob in order_blocks.bearish_ob if ob.status == OrderBlockStatus.FRESH])
        fresh_zones.extend([fvg for fvg in fair_value_gaps.bullish_fvg if fvg.status == FVGStatus.FRESH])
        fresh_zones.extend([fvg for fvg in fair_value_gaps.bearish_fvg if fvg.status == FVGStatus.FRESH])
        fresh_zones_count = len(fresh_zones)

        # Count recently mitigated zones (assume within last 24 hours)
        mitigated_today = 0
        all_obs = order_blocks.bullish_ob + order_blocks.bearish_ob
        mitigated_today += len([ob for ob in all_obs if ob.status == OrderBlockStatus.MITIGATED and ob.age_hours <= 24])

        all_fvgs = fair_value_gaps.bullish_fvg + fair_value_gaps.bearish_fvg
        mitigated_today += len(
            [fvg for fvg in all_fvgs if fvg.status == FVGStatus.FULLY_FILLED and fvg.age_hours <= 24])

        # Determine zone imbalance
        if total_demand_zones > total_supply_zones + 1:
            zone_imbalance = ZoneImbalance.DEMAND_HEAVY
        elif total_supply_zones > total_demand_zones + 1:
            zone_imbalance = ZoneImbalance.SUPPLY_HEAVY
        else:
            zone_imbalance = ZoneImbalance.BALANCED

        # Calculate average zone height
        zone_heights = []
        zone_heights.extend([ob.zone_height for ob in all_obs])
        zone_heights.extend([fvg.gap_size for fvg in all_fvgs])
        avg_zone_height = sum(zone_heights) / len(zone_heights) if zone_heights else 0

        # Find strongest zone distance
        strongest_zone_distance = "N/A"
        if all_obs:
            strongest_ob = max(all_obs, key=lambda x: x.strength)
            strongest_zone_distance = strongest_ob.distance_from_current

        return SupplyDemandMetrics(
            total_demand_zones=total_demand_zones,
            total_supply_zones=total_supply_zones,
            fresh_zones_count=fresh_zones_count,
            mitigated_today=mitigated_today,
            zone_imbalance=zone_imbalance,
            avg_zone_height=avg_zone_height,
            strongest_zone_distance=strongest_zone_distance
        )

    def analyze_supply_demand_zones(self, ohlcv_data: Dict[str, List[float]], timeframe: str) -> Dict[str, Any]:
        """
        Main function to analyze supply demand zones using smartmoneyconcepts
        """
        df = self.prepare_dataframe(ohlcv_data)
        current_price = float(ohlcv_data['close'][-1])

        # Calculate swing highs and lows (required for OB analysis)
        swing_data = self.calculate_swing_highs_lows(df, swing_length=50)
        swing_highs_lows = swing_data["swing_dataframe"]

        # Get timeframe levels for confluence analysis
        timeframe_levels = {}
        for tf in ["1D", "1W", "1M"]:
            try:
                prev_hl_data = self.calculate_previous_high_low(df, time_frame=tf)
                timeframe_levels[tf.lower()] = prev_hl_data
            except:
                pass

        # Analyze order blocks
        order_blocks = self.analyze_order_blocks(df, swing_highs_lows, current_price, timeframe, timeframe_levels)

        # Analyze fair value gaps
        fair_value_gaps = self.analyze_fair_value_gaps(df, current_price, timeframe)

        # Identify breaker blocks
        breaker_blocks = self.identify_breaker_blocks(order_blocks, df, swing_highs_lows, current_price, timeframe)

        # Build zone analysis
        zone_analysis = self.build_zone_analysis(order_blocks, fair_value_gaps, breaker_blocks, current_price)

        # Create confluence matrix
        confluence_matrix = self.create_confluence_matrix(order_blocks, fair_value_gaps, breaker_blocks)

        # Calculate metrics
        supply_demand_metrics = self.calculate_supply_demand_metrics(order_blocks, fair_value_gaps, breaker_blocks,
                                                                     current_price)

        return {
            "current_price": current_price,
            "order_blocks": order_blocks,
            "fair_value_gaps": fair_value_gaps,
            "breaker_blocks": breaker_blocks,
            "zone_analysis": zone_analysis,
            "confluence_matrix": confluence_matrix,
            "supply_demand_metrics": supply_demand_metrics
        }


@router.get("/api/supply-demand-zones/{symbol}/{timeframe}",
            response_model=SupplyDemandZonesResponse,
            summary="Get Supply Demand Zones Analysis",
            description="Analyze cryptocurrency supply and demand zones using Smart Money Concepts (SMC)")
async def get_supply_demand_zones(
        symbol: str = Path(..., description="Trading pair symbol (e.g., BTCUSDT, BTC)"),
        timeframe: str = Path(..., description="Analysis timeframe (e.g., 1h, 4h, 1d)"),
        as_of_datetime: Optional[str] = Query(None,
                                              description="ISO 8601 datetime string for historical analysis (e.g., 2024-01-01T00:00:00Z). Defaults to current time."),
        _: bool = Depends(verify_api_key)
):
    """
    Analyze cryptocurrency supply and demand zones using Smart Money Concepts.
    
    This endpoint:
    1. Fetches klines data from Binance for the specified symbol and timeframe
    2. Identifies Order Blocks using institutional order flow analysis
    3. Detects Fair Value Gaps (FVG) indicating price imbalances
    4. Identifies Breaker Blocks from broken Order Blocks
    5. Performs confluence analysis for high-probability zones
    6. Returns comprehensive supply/demand analysis with trading recommendations
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., 'BTC', 'ETH', 'BTCUSDT')
    - **timeframe**: Analysis timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
    - **as_of_datetime**: Optional ISO 8601 datetime string for historical analysis (e.g., '2024-01-01T00:00:00Z'). Defaults to current time.
    
    **Returns comprehensive supply/demand analysis including:**
    - Order Blocks (institutional demand/supply zones) with volume profiling
    - Fair Value Gaps (price imbalances) with fill percentage tracking
    - Breaker Blocks (broken OBs turned opposite zones)
    - Zone analysis with nearest demand/supply zones and strength assessment
    - Confluence matrix identifying high-probability zones
    - Comprehensive metrics including zone balance and statistics
    
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
        analyzer = SupplyDemandAnalyzer()

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

        # Fetch data from Binance (use more data for better zone analysis)
        klines_data = await binance_fetcher.get_klines(
            symbol=processed_symbol,
            interval=timeframe,
            limit=300,  # More data for comprehensive zone analysis
            end_time=end_time_ms
        )
        formatted_data = binance_fetcher.format_klines_data(klines_data)

        # Perform supply demand zones analysis
        analysis_result = analyzer.analyze_supply_demand_zones(formatted_data, timeframe)

        # Create response
        return SupplyDemandZonesResponse(
            symbol=processed_symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc).isoformat() if end_time_ms is None else as_of_datetime,
            current_price=analysis_result["current_price"],
            order_blocks=analysis_result["order_blocks"],
            fair_value_gaps=analysis_result["fair_value_gaps"],
            breaker_blocks=analysis_result["breaker_blocks"],
            zone_analysis=analysis_result["zone_analysis"],
            confluence_matrix=analysis_result["confluence_matrix"],
            supply_demand_metrics=analysis_result["supply_demand_metrics"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supply demand zones analysis failed: {str(e)}")


@router.get("/api/supply-demand-zones/doc",
            summary="Supply Demand Zones API Documentation",
            description="Get comprehensive documentation for the Supply Demand Zones API response format")
async def get_supply_demand_zones_documentation():
    """
    Get detailed documentation for the Supply Demand Zones API response format.
    
    This endpoint explains all fields, enums, and data structures returned by the
    /api/supply-demand-zones/{symbol}/{timeframe} endpoint.
    """
    return {
        "api_endpoint": "/api/supply-demand-zones/{symbol}/{timeframe}",
        "description": "Analyzes supply and demand zones using Smart Money Concepts (SMC) including Order Blocks, Fair Value Gaps, and Breaker Blocks to identify institutional order flow and price imbalances.",

        "core_concepts": {
            "order_blocks": "Areas where institutions placed large orders, creating supply/demand imbalances",
            "fair_value_gaps": "Price gaps indicating rapid institutional movement and potential retracement areas",
            "breaker_blocks": "Former order blocks that were broken and now act as opposite zones",
            "premium_discount": "Market positioning relative to key supply/demand levels"
        },

        "order_blocks": {
            "description": "Institutional demand (bullish) and supply (bearish) zones",
            "fields": {
                "top": {
                    "type": "number",
                    "description": "Upper boundary of the order block",
                    "example": 45200.0
                },
                "bottom": {
                    "type": "number",
                    "description": "Lower boundary of the order block",
                    "example": 44800.0
                },
                "ob_type": {
                    "type": "string",
                    "description": "Type of order block",
                    "possible_values": {
                        "Bullish OB": "Demand zone where institutions accumulated long positions",
                        "Bearish OB": "Supply zone where institutions accumulated short positions"
                    }
                },
                "formation_index": {
                    "type": "integer",
                    "description": "Candle index where order block was formed",
                    "example": -25,
                    "note": "Negative values indicate candles in the past"
                },
                "age_hours": {
                    "type": "integer",
                    "description": "Hours since order block formation",
                    "example": 72
                },
                "status": {
                    "type": "enum",
                    "description": "Current status of the order block",
                    "possible_values": {
                        "FRESH": "Untested zone with highest reaction probability",
                        "TESTED_HOLDING": "Zone tested but held, still valid",
                        "TESTED_BROKEN": "Zone tested and partially broken",
                        "MITIGATED": "Zone fully broken and no longer valid"
                    }
                },
                "test_count": {
                    "type": "integer",
                    "description": "Number of times zone was tested",
                    "range": "0-10+",
                    "interpretation": {
                        "0": "Fresh zone, highest probability",
                        "1-2": "Lightly tested, still strong",
                        "3+": "Heavily tested, weakening"
                    }
                },
                "volume_profile": {
                    "type": "enum",
                    "description": "Volume profile during formation",
                    "possible_values": {
                        "EXTREME": "Exceptional volume, very strong zone",
                        "HIGH": "High volume, strong zone",
                        "MODERATE": "Average volume, moderate strength",
                        "LOW": "Low volume, weaker zone"
                    }
                },
                "strength": {
                    "type": "integer",
                    "description": "Overall strength score",
                    "range": "1-10",
                    "factors": ["Volume profile", "Test count", "Age", "Formation quality"],
                    "interpretation": {
                        "1-3": "Weak zone, low probability",
                        "4-6": "Moderate zone, decent probability",
                        "7-8": "Strong zone, high probability",
                        "9-10": "Very strong zone, excellent probability"
                    }
                },
                "distance_from_current": {
                    "type": "string",
                    "description": "Percentage distance from current price",
                    "example": "-2.5%"
                },
                "zone_height": {
                    "type": "number",
                    "description": "Height of zone in price points",
                    "example": 400.0
                },
                "confluence": {
                    "type": "array[string]",
                    "description": "List of confluences with other levels",
                    "examples": ["Previous Daily High", "Fibonacci 61.8%", "Session Resistance"]
                }
            }
        },

        "fair_value_gaps": {
            "description": "Price gaps indicating institutional movement and potential fill zones",
            "types": {
                "bullish_fvg": "Gaps formed during upward movement, potential support",
                "bearish_fvg": "Gaps formed during downward movement, potential resistance"
            },
            "fields": {
                "top": {
                    "type": "number",
                    "description": "Upper boundary of the gap",
                    "example": 45100.0
                },
                "bottom": {
                    "type": "number",
                    "description": "Lower boundary of the gap",
                    "example": 44900.0
                },
                "type": {
                    "type": "string",
                    "possible_values": {
                        "Bullish FVG": "Upward gap, potential support area",
                        "Bearish FVG": "Downward gap, potential resistance area"
                    }
                },
                "status": {
                    "type": "enum",
                    "possible_values": {
                        "FRESH": "Gap not yet filled, highest probability",
                        "PARTIALLY_FILLED": "Gap partially filled, some reaction expected",
                        "FULLY_FILLED": "Gap completely filled, no longer valid"
                    }
                },
                "fill_percentage": {
                    "type": "number",
                    "description": "Percentage of gap that has been filled",
                    "range": "0-100",
                    "interpretation": {
                        "0-25": "Fresh gap, strong reaction expected",
                        "25-75": "Partially filled, moderate reaction",
                        "75-100": "Mostly/fully filled, weak reaction"
                    }
                },
                "gap_size": {
                    "type": "number",
                    "description": "Size of the gap in price points",
                    "example": 200.0
                }
            }
        },

        "breaker_blocks": {
            "description": "Former order blocks that were broken and now act as opposite zones",
            "concept": "When a bullish OB is broken, it becomes a bearish breaker (resistance)",
            "fields": {
                "level": {
                    "type": "number",
                    "description": "Price level of the breaker block",
                    "example": 44500.0
                },
                "type": {
                    "type": "string",
                    "possible_values": {
                        "Bullish Breaker": "Former supply zone now acting as support",
                        "Bearish Breaker": "Former demand zone now acting as resistance"
                    }
                },
                "original_ob_type": {
                    "type": "string",
                    "description": "Original order block type before break",
                    "example": "Bullish OB"
                },
                "break_candle_index": {
                    "type": "integer",
                    "description": "Candle index where break occurred",
                    "example": -8
                },
                "status": {
                    "type": "enum",
                    "possible_values": {
                        "ACTIVE": "Breaker recently formed, still relevant",
                        "INACTIVE": "Breaker old, less relevant"
                    }
                },
                "strength": {
                    "type": "integer",
                    "description": "Strength score (reduced from original OB)",
                    "range": "1-10"
                }
            }
        },

        "zone_analysis": {
            "description": "Comprehensive analysis of supply/demand zones relative to current price",
            "fields": {
                "nearest_demand_zone": {
                    "type": "object",
                    "description": "Closest demand zone below current price",
                    "structure": {
                        "zone": "Array [bottom, top] price levels",
                        "type": "Zone type (e.g., 'Bullish OB')",
                        "strength": "Strength score 1-10",
                        "distance": "Percentage distance from current price",
                        "recommendation": "Trading recommendation"
                    }
                },
                "nearest_supply_zone": {
                    "type": "object",
                    "description": "Closest supply zone above current price",
                    "note": "Same structure as nearest_demand_zone"
                },
                "strongest_zone": {
                    "type": "object",
                    "description": "Highest strength zone regardless of position",
                    "note": "Key zone for major reactions"
                },
                "zone_density": {
                    "type": "enum",
                    "description": "Distribution of supply vs demand zones",
                    "possible_values": {
                        "DEMAND_HEAVY": "More demand zones, bullish bias",
                        "SUPPLY_HEAVY": "More supply zones, bearish bias",
                        "BALANCED": "Equal distribution, neutral bias"
                    }
                },
                "premium_discount": {
                    "type": "enum",
                    "description": "Current market positioning",
                    "possible_values": {
                        "PREMIUM": "Price in upper range, consider sells",
                        "EQUILIBRIUM": "Price in middle range, wait for direction",
                        "DISCOUNT": "Price in lower range, consider buys"
                    }
                }
            }
        },

        "confluence_matrix": {
            "description": "High-probability zones with multiple confluences",
            "purpose": "Identify zones with highest reaction probability",
            "fields": {
                "high_probability_zones": {
                    "type": "array",
                    "description": "Zones with multiple confluences",
                    "max_items": 8,
                    "sorted_by": "Score (highest first)",
                    "item_fields": {
                        "zone": {
                            "type": "array[number]",
                            "description": "Zone boundaries [bottom, top]",
                            "example": [44800, 45200]
                        },
                        "confluences": {
                            "type": "array[string]",
                            "description": "List of confluences at this zone",
                            "examples": ["Bullish OB", "Untested Zone", "High Volume", "FVG Overlap"]
                        },
                        "score": {
                            "type": "number",
                            "description": "Combined confluence score",
                            "range": "0-10",
                            "interpretation": {
                                "8-10": "Exceptional confluence, very high probability",
                                "6-8": "Strong confluence, high probability",
                                "4-6": "Moderate confluence, decent probability",
                                "0-4": "Weak confluence, low probability"
                            }
                        }
                    }
                }
            }
        },

        "supply_demand_metrics": {
            "description": "Quantitative metrics and statistics",
            "fields": {
                "total_demand_zones": {
                    "type": "integer",
                    "description": "Number of active demand zones",
                    "example": 4
                },
                "total_supply_zones": {
                    "type": "integer",
                    "description": "Number of active supply zones",
                    "example": 3
                },
                "fresh_zones_count": {
                    "type": "integer",
                    "description": "Number of fresh (untested) zones",
                    "example": 2,
                    "note": "Higher count indicates more opportunities"
                },
                "mitigated_today": {
                    "type": "integer",
                    "description": "Zones broken in last 24 hours",
                    "example": 1,
                    "interpretation": "Higher values indicate active market"
                },
                "zone_imbalance": {
                    "type": "enum",
                    "description": "Overall zone distribution",
                    "same_as": "zone_analysis.zone_density"
                },
                "avg_zone_height": {
                    "type": "number",
                    "description": "Average height of zones in price points",
                    "example": 350.0
                },
                "strongest_zone_distance": {
                    "type": "string",
                    "description": "Distance to strongest zone",
                    "example": "+1.8%"
                }
            }
        },

        "trading_interpretation": {
            "entry_signals": [
                "Price approaching FRESH order block",
                "PREMIUM/DISCOUNT positioning for direction",
                "High confluence zones (score >7)",
                "Fresh FVGs with <25% fill"
            ],
            "exit_signals": [
                "Order block MITIGATED",
                "FVG 100% filled",
                "Breaker block broken again",
                "Multiple zone rejections"
            ],
            "risk_management": [
                "Place stops beyond zone boundaries",
                "Reduce size at heavily tested zones",
                "Avoid trading against strongest zones",
                "Monitor zone_imbalance for bias"
            ],
            "confluence_priorities": [
                "1. Fresh zones (highest priority)",
                "2. High volume zones",
                "3. Multiple timeframe confluences",
                "4. Untested breaker blocks",
                "5. Premium/discount positioning"
            ]
        },

        "usage_examples": {
            "bullish_setup": {
                "description": "Example bullish supply/demand setup",
                "indicators": [
                    "premium_discount: DISCOUNT",
                    "nearest_demand_zone with strength 8+",
                    "zone_density: DEMAND_HEAVY",
                    "Fresh Bullish OB below price"
                ]
            },
            "bearish_setup": {
                "description": "Example bearish supply/demand setup",
                "indicators": [
                    "premium_discount: PREMIUM",
                    "nearest_supply_zone with strength 8+",
                    "zone_density: SUPPLY_HEAVY",
                    "Fresh Bearish OB above price"
                ]
            }
        }
    }
