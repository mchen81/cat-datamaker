from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field
from smartmoneyconcepts import smc

from binance_data import BinanceDataFetcher
from smc_analysis import SMCAnalyzer
from utils import verify_api_key


class TrendDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SwingType(str, Enum):
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low


class SwingStatus(str, Enum):
    UNBROKEN_RESISTANCE = "UNBROKEN_RESISTANCE"
    BROKEN_NOW_SUPPORT = "BROKEN_NOW_SUPPORT"
    PROTECTED_LOW = "PROTECTED_LOW"
    KEY_SUPPORT = "KEY_SUPPORT"
    KEY_RESISTANCE = "KEY_RESISTANCE"


class RetracementData(BaseModel):
    direction: str = Field(..., description="Pullback or Extension")
    current_percent: float = Field(..., description="Current retracement percentage")
    deepest_percent: float = Field(..., description="Deepest retracement percentage")


class MarketStructureData(BaseModel):
    trend: TrendDirection = Field(..., description="Overall trend direction")
    trend_strength: int = Field(..., description="Trend strength score (1-10)", ge=1, le=10)
    structure_clarity: int = Field(..., description="Structure clarity score (1-10)", ge=1, le=10)
    last_update: str = Field(..., description="Last structure update timestamp")
    retracement: RetracementData = Field(..., description="Current retracement data")


class BOSCHoCHEvent(BaseModel):
    type: str = Field(..., description="Type of event (Bullish BOS, Bearish CHoCH, etc.)")
    level: float = Field(..., description="Price level of the event")
    distance_from_current: str = Field(..., description="Distance from current price as percentage")
    candle_index: int = Field(..., description="Relative candle index position")
    time_ago_hours: int = Field(..., description="Hours since the event")
    broke_level: float = Field(..., description="Price level that was broken")


class SwingPoint(BaseModel):
    level: float = Field(..., description="Price level of swing point")
    distance_from_current: str = Field(..., description="Distance from current price as percentage")
    candle_index: int = Field(..., description="Relative candle index position")
    age_hours: int = Field(..., description="Age of swing point in hours")
    swing_type: SwingType = Field(..., description="Type of swing point")
    status: SwingStatus = Field(..., description="Current status of swing point")


class SwingPoints(BaseModel):
    recent_highs: List[SwingPoint] = Field(..., description="Recent swing highs")
    recent_lows: List[SwingPoint] = Field(..., description="Recent swing lows")


class StructureAnalysis(BaseModel):
    swing_pattern: str = Field(..., description="Current swing pattern")
    last_significant_move: str = Field(..., description="Description of last significant move")
    structure_intact: bool = Field(..., description="Whether structure is intact")
    invalidation_level: float = Field(..., description="Level that would invalidate structure")
    confirmation_level: float = Field(..., description="Level that would confirm continuation")
    key_message: str = Field(..., description="Key takeaway message")


class SMCMetrics(BaseModel):
    structure_score: float = Field(..., description="Overall structure score")
    momentum_score: float = Field(..., description="Momentum score")
    clarity_score: float = Field(..., description="Clarity score")
    overall_bias: TrendDirection = Field(..., description="Overall trading bias")


class MarketStructureResponse(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Analysis timeframe")
    timestamp: str = Field(..., description="Analysis timestamp")
    current_price: float = Field(..., description="Current price")
    market_structure: MarketStructureData = Field(..., description="Market structure analysis")
    recent_bos_choch: List[BOSCHoCHEvent] = Field(..., description="Recent BOS/CHoCH events")
    swing_points: SwingPoints = Field(..., description="Recent swing points")
    structure_analysis: StructureAnalysis = Field(..., description="Structure analysis summary")
    smc_metrics: SMCMetrics = Field(..., description="SMC scoring metrics")


# Create router
router = APIRouter(tags=["Market Structure"])


class MarketStructureAnalyzer(SMCAnalyzer):
    """
    Market Structure Analysis using smartmoneyconcepts library
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

    def determine_swing_type(self, highs: List[float], lows: List[float], current_idx: int) -> SwingType:
        """
        Determine swing type (HH, HL, LH, LL) based on previous swings
        """
        if len(highs) < 2 and len(lows) < 2:
            return SwingType.HH  # Default for first swing

        # Simple logic: compare current level with previous swing
        if current_idx < len(highs) - 1:  # This is a high
            current_high = highs[current_idx]
            if current_idx > 0:
                previous_high = highs[current_idx - 1]
                return SwingType.HH if current_high > previous_high else SwingType.LH
        elif current_idx < len(lows) - 1:  # This is a low
            current_low = lows[current_idx]
            if current_idx > 0:
                previous_low = lows[current_idx - 1]
                return SwingType.HL if current_low > previous_low else SwingType.LL

        return SwingType.HH  # Default

    def determine_swing_status(self, level: float, current_price: float, is_high: bool) -> SwingStatus:
        """
        Determine swing point status based on current price and recent breaks
        """
        if is_high:
            if level > current_price:
                return SwingStatus.UNBROKEN_RESISTANCE
            else:
                return SwingStatus.BROKEN_NOW_SUPPORT
        else:
            if level < current_price:
                return SwingStatus.PROTECTED_LOW
            else:
                return SwingStatus.KEY_SUPPORT

    def analyze_bos_choch_events(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame,
                                 current_price: float, timeframe: str) -> List[BOSCHoCHEvent]:
        """
        Analyze BOS and CHoCH events and return structured data
        """
        try:
            # Use inherited method to get BOS/CHoCH data, then extract what we need
            bos_choch_data = self.calculate_bos_choch(df, swing_highs_lows, close_break=True)
            bos_choch = smc.bos_choch(df, swing_highs_lows, close_break=True)
            events = []

            if not bos_choch.empty:
                # Get the last 15 events for analysis
                recent_events = bos_choch.tail(15)

                for idx, (df_idx, row) in enumerate(recent_events.iterrows()):
                    # Calculate candle index (negative for past candles)
                    try:
                        # Get position of this timestamp in the original dataframe
                        if df_idx in df.index:
                            position_in_df = df.index.get_loc(df_idx)
                            candle_index = -(len(df) - position_in_df - 1)
                        else:
                            # Fallback: use the enumeration index
                            candle_index = -(len(recent_events) - idx - 1)
                    except (KeyError, ValueError):
                        # Fallback: use the enumeration index
                        candle_index = -(len(recent_events) - idx - 1)

                    # Determine event type
                    event_type = "Neutral"
                    if row.get('BOS', 0) == 1:
                        if row.get('Direction', 0) == 1:
                            event_type = "Bullish BOS"
                        else:
                            event_type = "Bearish BOS"
                    elif row.get('CHOCH', 0) == 1:
                        if row.get('Direction', 0) == 1:
                            event_type = "Bullish CHoCH"
                        else:
                            event_type = "Bearish CHoCH"

                    level = self._convert_numpy_types(row.get('Level', current_price))
                    broke_level = self._convert_numpy_types(row.get('BrokenIndex', level))

                    events.append(BOSCHoCHEvent(
                        type=event_type,
                        level=level,
                        distance_from_current=self.calculate_distance_percentage(level, current_price),
                        candle_index=candle_index,
                        time_ago_hours=self.get_hours_ago(candle_index, timeframe),
                        broke_level=broke_level
                    ))

            return events[:10]  # Return last 10 events

        except Exception as e:
            print(f"Error in BOS/CHoCH analysis: {e}")
            return []

    def analyze_swing_points(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame,
                             current_price: float, timeframe: str,
                             recent_bos_choch: List[BOSCHoCHEvent]) -> SwingPoints:
        """
        Analyze swing points and classify them
        """
        recent_highs = []
        recent_lows = []

        if not swing_highs_lows.empty:
            # Get recent swing highs
            swing_highs = swing_highs_lows[swing_highs_lows['HighLow'] == 1].tail(5)
            for idx, (df_idx, row) in enumerate(swing_highs.iterrows()):
                try:
                    if df_idx in df.index:
                        position_in_df = df.index.get_loc(df_idx)
                        candle_index = -(len(df) - position_in_df - 1)
                    else:
                        candle_index = -(len(swing_highs) - idx - 1)
                except (KeyError, ValueError):
                    candle_index = -(len(swing_highs) - idx - 1)
                level = self._convert_numpy_types(row.get('Level', current_price))

                swing_point = SwingPoint(
                    level=level,
                    distance_from_current=self.calculate_distance_percentage(level, current_price),
                    candle_index=candle_index,
                    age_hours=self.get_hours_ago(candle_index, timeframe),
                    swing_type=SwingType.HH if level > current_price else SwingType.LH,
                    status=self.determine_swing_status(level, current_price, True)
                )
                recent_highs.append(swing_point)

            # Get recent swing lows
            swing_lows = swing_highs_lows[swing_highs_lows['HighLow'] == -1].tail(5)
            for idx, (df_idx, row) in enumerate(swing_lows.iterrows()):
                try:
                    if df_idx in df.index:
                        position_in_df = df.index.get_loc(df_idx)
                        candle_index = -(len(df) - position_in_df - 1)
                    else:
                        candle_index = -(len(swing_lows) - idx - 1)
                except (KeyError, ValueError):
                    candle_index = -(len(swing_lows) - idx - 1)
                level = self._convert_numpy_types(row.get('Level', current_price))

                swing_point = SwingPoint(
                    level=level,
                    distance_from_current=self.calculate_distance_percentage(level, current_price),
                    candle_index=candle_index,
                    age_hours=self.get_hours_ago(candle_index, timeframe),
                    swing_type=SwingType.HL if level > current_price else SwingType.LL,
                    status=self.determine_swing_status(level, current_price, False)
                )
                recent_lows.append(swing_point)

        return SwingPoints(recent_highs=recent_highs, recent_lows=recent_lows)

    def analyze_retracements(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame) -> RetracementData:
        """
        Analyze price retracements using smartmoneyconcepts
        """
        try:
            # Use inherited method to get retracement data
            retracements_data = self.calculate_retracements(df, swing_highs_lows)

            if retracements_data["retracements"]:
                latest = retracements_data["retracements"][-1]  # Get the last retracement
                direction = latest.get('direction', 0)
                current_percent = latest.get('current_retracement_percent', 0)
                deepest_percent = latest.get('deepest_retracement_percent', 0)

                # Convert direction code to readable string
                direction_str = "Pullback" if direction == -1 else "Extension" if direction == 1 else "Pullback"

                return RetracementData(
                    direction=direction_str,
                    current_percent=abs(current_percent) if current_percent else 0,
                    deepest_percent=abs(deepest_percent) if deepest_percent else 0
                )
        except Exception as e:
            print(f"Error in retracement analysis: {e}")

        return RetracementData(direction="Pullback", current_percent=0, deepest_percent=0)

    def calculate_trend_metrics(self, recent_bos_choch: List[BOSCHoCHEvent]) -> tuple[TrendDirection, int, int]:
        """
        Calculate trend direction, strength, and clarity based on BOS/CHoCH events
        """
        if not recent_bos_choch:
            return TrendDirection.NEUTRAL, 5, 5

        # Analyze recent events for trend
        bullish_events = sum(1 for event in recent_bos_choch[:5] if "Bullish" in event.type)
        bearish_events = sum(1 for event in recent_bos_choch[:5] if "Bearish" in event.type)

        # Determine trend direction
        if bullish_events > bearish_events + 1:
            trend = TrendDirection.BULLISH
        elif bearish_events > bullish_events + 1:
            trend = TrendDirection.BEARISH
        else:
            trend = TrendDirection.NEUTRAL

        # Calculate trend strength (1-10)
        trend_strength = min(10, max(1, abs(bullish_events - bearish_events) * 2 + 3))

        # Calculate structure clarity (1-10) - higher if events are consistent
        bos_count = sum(1 for event in recent_bos_choch[:5] if "BOS" in event.type)
        choch_count = sum(1 for event in recent_bos_choch[:5] if "CHoCH" in event.type)

        structure_clarity = min(10, max(1, (bos_count * 2) + 2))

        return trend, trend_strength, structure_clarity

    def build_structure_analysis(self, swing_points: SwingPoints, recent_bos_choch: List[BOSCHoCHEvent],
                                 current_price: float, trend: TrendDirection) -> StructureAnalysis:
        """
        Build comprehensive structure analysis
        """
        # Determine swing pattern
        swing_pattern = "NEUTRAL"
        if swing_points.recent_highs and swing_points.recent_lows:
            if len(swing_points.recent_highs) >= 2:
                if swing_points.recent_highs[0].swing_type == SwingType.HH:
                    swing_pattern = "HH-HL" if swing_points.recent_lows and swing_points.recent_lows[
                        0].swing_type == SwingType.HL else "HH"
                else:
                    swing_pattern = "LH-LL" if swing_points.recent_lows and swing_points.recent_lows[
                        0].swing_type == SwingType.LL else "LH"

        # Last significant move
        last_significant_move = "No significant moves detected"
        if recent_bos_choch:
            latest_event = recent_bos_choch[0]
            last_significant_move = f"{latest_event.type} at {latest_event.level}"

        # Structure intact
        structure_intact = trend != TrendDirection.NEUTRAL

        # Calculate invalidation and confirmation levels
        invalidation_level = current_price * 0.95  # Default 5% below
        confirmation_level = current_price * 1.05  # Default 5% above

        if swing_points.recent_lows:
            invalidation_level = min([point.level for point in swing_points.recent_lows[:2]])

        if swing_points.recent_highs:
            confirmation_level = max([point.level for point in swing_points.recent_highs[:2]])

        # Key message
        if trend == TrendDirection.BULLISH:
            key_message = f"Bullish structure maintained. Price holding above recent support at {invalidation_level:.0f}."
        elif trend == TrendDirection.BEARISH:
            key_message = f"Bearish structure maintained. Price below recent resistance at {confirmation_level:.0f}."
        else:
            key_message = "Market in consolidation. Watch for breakout above resistance or below support."

        return StructureAnalysis(
            swing_pattern=swing_pattern,
            last_significant_move=last_significant_move,
            structure_intact=structure_intact,
            invalidation_level=invalidation_level,
            confirmation_level=confirmation_level,
            key_message=key_message
        )

    def calculate_smc_metrics(self, trend: TrendDirection, trend_strength: int,
                              structure_clarity: int, recent_bos_choch: List[BOSCHoCHEvent]) -> SMCMetrics:
        """
        Calculate SMC scoring metrics
        """
        structure_score = (trend_strength + structure_clarity) / 2.0

        # Momentum score based on recent events timing
        momentum_score = 6.0
        if recent_bos_choch:
            recent_events = [e for e in recent_bos_choch if e.time_ago_hours <= 24]
            momentum_score = min(10.0, max(1.0, len(recent_events) * 2))

        clarity_score = float(structure_clarity)

        return SMCMetrics(
            structure_score=structure_score,
            momentum_score=momentum_score,
            clarity_score=clarity_score,
            overall_bias=trend
        )

    def analyze_market_structure(self, ohlcv_data: Dict[str, List[float]], timeframe: str) -> Dict[str, Any]:
        """
        Main function to analyze market structure using smartmoneyconcepts
        """
        df = self.prepare_dataframe(ohlcv_data)
        current_price = float(ohlcv_data['close'][-1])

        # Calculate swing highs and lows using inherited method
        swing_data = self.calculate_swing_highs_lows(df, swing_length=50)
        swing_highs_lows = swing_data["swing_dataframe"]

        # Analyze BOS/CHoCH events
        recent_bos_choch = self.analyze_bos_choch_events(df, swing_highs_lows, current_price, timeframe)

        # Analyze swing points
        swing_points = self.analyze_swing_points(df, swing_highs_lows, current_price, timeframe, recent_bos_choch)

        # Analyze retracements
        retracement_data = self.analyze_retracements(df, swing_highs_lows)

        # Calculate trend metrics
        trend, trend_strength, structure_clarity = self.calculate_trend_metrics(recent_bos_choch)

        # Build structure analysis
        structure_analysis = self.build_structure_analysis(swing_points, recent_bos_choch, current_price, trend)

        # Calculate SMC metrics
        smc_metrics = self.calculate_smc_metrics(trend, trend_strength, structure_clarity, recent_bos_choch)

        # Get last update time (most recent BOS/CHoCH)
        # Use the latest data point timestamp as the reference
        reference_time = df.index[-1] if not df.empty else datetime.now(timezone.utc)

        last_update = reference_time.isoformat()
        if recent_bos_choch:
            # Calculate approximate timestamp for the most recent event
            hours_ago = recent_bos_choch[0].time_ago_hours
            last_update = (reference_time - pd.Timedelta(hours=hours_ago)).isoformat()

        market_structure_data = MarketStructureData(
            trend=trend,
            trend_strength=trend_strength,
            structure_clarity=structure_clarity,
            last_update=last_update,
            retracement=retracement_data
        )

        return {
            "current_price": current_price,
            "market_structure": market_structure_data,
            "recent_bos_choch": recent_bos_choch,
            "swing_points": swing_points,
            "structure_analysis": structure_analysis,
            "smc_metrics": smc_metrics
        }


@router.get("/api/market-structure/{symbol}/{timeframe}",
            response_model=MarketStructureResponse,
            summary="Get Market Structure Analysis",
            description="Analyze cryptocurrency market structure using Smart Money Concepts (SMC)")
async def get_market_structure(
        symbol: str = Path(..., description="Trading pair symbol (e.g., BTCUSDT, BTC)"),
        timeframe: str = Path(..., description="Analysis timeframe (e.g., 1h, 4h, 1d)"),
        as_of_datetime: Optional[str] = Query(None,
                                              description="ISO 8601 datetime string for historical analysis (e.g., 2024-01-01T00:00:00Z). Defaults to current time."),
        _: bool = Depends(verify_api_key)
):
    """
    Analyze cryptocurrency market structure using Smart Money Concepts.
    
    This endpoint:
    1. Fetches klines data from Binance for the specified symbol and timeframe
    2. Performs comprehensive market structure analysis using SMC concepts
    3. Identifies BOS (Break of Structure) and CHoCH (Change of Character) events
    4. Analyzes swing highs and lows with classifications
    5. Calculates trend metrics and structure scores
    6. Returns comprehensive market structure data
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., 'BTC', 'ETH', 'BTCUSDT')
    - **timeframe**: Analysis timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
    - **as_of_datetime**: Optional ISO 8601 datetime string for historical analysis (e.g., '2024-01-01T00:00:00Z'). Defaults to current time.
    
    **Returns comprehensive market structure analysis including:**
    - Market structure trend and strength metrics
    - Recent BOS/CHoCH events with timestamps and levels
    - Swing points classification (HH, HL, LH, LL)
    - Structure analysis with invalidation/confirmation levels
    - SMC scoring metrics
    
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
        analyzer = MarketStructureAnalyzer()

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
                end_time_ms = None  # Will default to current time in get_klines

        # Fetch data from Binance
        klines_data = await binance_fetcher.get_klines(
            symbol=processed_symbol,
            interval=timeframe,
            limit=200,  # Get enough data for proper SMC analysis
            end_time=end_time_ms
        )
        formatted_data = binance_fetcher.format_klines_data(klines_data)

        # Perform market structure analysis
        analysis_result = analyzer.analyze_market_structure(formatted_data, timeframe)

        # Create response
        return MarketStructureResponse(
            symbol=processed_symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc).isoformat() if end_time_ms is None else as_of_datetime,
            current_price=analysis_result["current_price"],
            market_structure=analysis_result["market_structure"],
            recent_bos_choch=analysis_result["recent_bos_choch"],
            swing_points=analysis_result["swing_points"],
            structure_analysis=analysis_result["structure_analysis"],
            smc_metrics=analysis_result["smc_metrics"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market structure analysis failed: {str(e)}")


@router.get("/api/market-structure/doc",
            summary="Market Structure API Documentation",
            description="Get comprehensive documentation for the Market Structure API response format")
async def get_market_structure_documentation():
    """
    Get detailed documentation for the Market Structure API response format.
    
    This endpoint explains all fields, enums, and data structures returned by the 
    /api/market-structure/{symbol}/{timeframe} endpoint.
    """
    return {
        "api_endpoint": "/api/market-structure/{symbol}/{timeframe}",
        "description": "Analyzes market structure using Smart Money Concepts (SMC) methodology to identify trend direction, swing points, and Break of Structure (BOS) / Change of Character (CHoCH) events.",

        "response_structure": {
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol (e.g., BTCUSDT)",
                "example": "BTCUSDT"
            },
            "timeframe": {
                "type": "string",
                "description": "Analysis timeframe",
                "example": "4h",
                "possible_values": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d",
                                    "1w", "1M"]
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

        "market_structure": {
            "description": "Core market structure analysis data",
            "fields": {
                "trend": {
                    "type": "enum",
                    "description": "Overall market trend direction based on BOS/CHoCH analysis",
                    "possible_values": {
                        "BULLISH": "Upward trending market with bullish BOS events",
                        "BEARISH": "Downward trending market with bearish BOS events",
                        "NEUTRAL": "No clear trend direction, consolidating market"
                    }
                },
                "trend_strength": {
                    "type": "integer",
                    "description": "Trend strength score from 1-10 based on recent BOS/CHoCH events",
                    "range": "1-10",
                    "interpretation": {
                        "1-3": "Weak trend, potential reversal or consolidation",
                        "4-6": "Moderate trend strength",
                        "7-10": "Strong trend with high probability of continuation"
                    }
                },
                "structure_clarity": {
                    "type": "integer",
                    "description": "How clear and well-defined the market structure is",
                    "range": "1-10",
                    "interpretation": {
                        "1-3": "Unclear structure, avoid trading",
                        "4-6": "Moderate clarity, trade with caution",
                        "7-10": "Very clear structure, high confidence trades"
                    }
                },
                "last_update": {
                    "type": "string",
                    "description": "Timestamp of last significant structure change",
                    "example": "2024-01-01T10:30:00Z"
                },
                "retracement": {
                    "description": "Current retracement analysis",
                    "fields": {
                        "direction": {
                            "type": "string",
                            "description": "Type of current price movement",
                            "possible_values": {
                                "Pullback": "Price retracing against main trend",
                                "Extension": "Price extending in direction of main trend"
                            }
                        },
                        "current_percent": {
                            "type": "number",
                            "description": "Current retracement percentage from swing point",
                            "example": 38.2
                        },
                        "deepest_percent": {
                            "type": "number",
                            "description": "Deepest retracement percentage reached",
                            "example": 50.0
                        }
                    }
                }
            }
        },

        "recent_bos_choch": {
            "description": "Array of recent Break of Structure and Change of Character events",
            "note": "Returns up to 10 most recent events, ordered by recency",
            "fields": {
                "type": {
                    "type": "string",
                    "description": "Type of structural event",
                    "possible_values": {
                        "Bullish BOS": "Break above previous high, continuation of uptrend",
                        "Bearish BOS": "Break below previous low, continuation of downtrend",
                        "Bullish CHoCH": "Break above previous high after downtrend, trend reversal to bullish",
                        "Bearish CHoCH": "Break below previous low after uptrend, trend reversal to bearish"
                    }
                },
                "level": {
                    "type": "number",
                    "description": "Price level where the BOS/CHoCH occurred",
                    "example": 44500.0
                },
                "distance_from_current": {
                    "type": "string",
                    "description": "Percentage distance from current price",
                    "example": "+2.25%",
                    "note": "Positive values are above current price, negative below"
                },
                "candle_index": {
                    "type": "integer",
                    "description": "Relative candle position (negative values indicate past candles)",
                    "example": -15,
                    "note": "-1 = previous candle, -15 = 15 candles ago"
                },
                "time_ago_hours": {
                    "type": "integer",
                    "description": "Hours since the event occurred",
                    "example": 60
                },
                "broke_level": {
                    "type": "number",
                    "description": "The swing high/low level that was broken",
                    "example": 44200.0
                }
            }
        },

        "swing_points": {
            "description": "Recent swing highs and lows identified in the market structure",
            "fields": {
                "recent_highs": {
                    "type": "array",
                    "description": "Up to 5 most recent swing highs",
                    "item_fields": {
                        "level": {
                            "type": "number",
                            "description": "Price level of swing high",
                            "example": 45200.0
                        },
                        "distance_from_current": {
                            "type": "string",
                            "description": "Percentage distance from current price",
                            "example": "+1.5%"
                        },
                        "candle_index": {
                            "type": "integer",
                            "description": "Relative candle position",
                            "example": -8
                        },
                        "age_hours": {
                            "type": "integer",
                            "description": "Hours since swing point was formed",
                            "example": 32
                        },
                        "swing_type": {
                            "type": "enum",
                            "description": "Type of swing point relative to previous swings",
                            "possible_values": {
                                "HH": "Higher High - new peak above previous high",
                                "LH": "Lower High - peak below previous high"
                            }
                        },
                        "status": {
                            "type": "enum",
                            "description": "Current status of the swing point",
                            "possible_values": {
                                "UNBROKEN_RESISTANCE": "Level acting as resistance, not yet broken",
                                "BROKEN_NOW_SUPPORT": "Previously broken resistance now acting as support",
                                "KEY_RESISTANCE": "Important resistance level"
                            }
                        }
                    }
                },
                "recent_lows": {
                    "type": "array",
                    "description": "Up to 5 most recent swing lows",
                    "item_fields": {
                        "swing_type": {
                            "possible_values": {
                                "HL": "Higher Low - trough above previous low",
                                "LL": "Lower Low - trough below previous low"
                            }
                        },
                        "status": {
                            "possible_values": {
                                "PROTECTED_LOW": "Low level holding as support",
                                "KEY_SUPPORT": "Important support level"
                            }
                        }
                    },
                    "note": "Other fields same as recent_highs"
                }
            }
        },

        "structure_analysis": {
            "description": "Comprehensive analysis and interpretation of market structure",
            "fields": {
                "swing_pattern": {
                    "type": "string",
                    "description": "Current swing pattern being formed",
                    "possible_values": {
                        "HH-HL": "Higher Highs and Higher Lows (bullish structure)",
                        "LH-LL": "Lower Highs and Lower Lows (bearish structure)",
                        "HH": "Making higher highs only",
                        "LH": "Making lower highs only",
                        "NEUTRAL": "No clear pattern"
                    }
                },
                "last_significant_move": {
                    "type": "string",
                    "description": "Description of the most recent significant structural move",
                    "example": "Bullish BOS at 44500"
                },
                "structure_intact": {
                    "type": "boolean",
                    "description": "Whether the current trend structure remains valid",
                    "interpretation": {
                        "true": "Structure is intact, trend likely to continue",
                        "false": "Structure compromised, expect consolidation or reversal"
                    }
                },
                "invalidation_level": {
                    "type": "number",
                    "description": "Price level that would invalidate current structure",
                    "example": 43800.0,
                    "note": "Usually the most recent significant swing low/high"
                },
                "confirmation_level": {
                    "type": "number",
                    "description": "Price level that would confirm structure continuation",
                    "example": 45500.0,
                    "note": "Usually the next resistance/support level to break"
                },
                "key_message": {
                    "type": "string",
                    "description": "Summary message with key insights and trading bias",
                    "example": "Bullish structure maintained. Price holding above recent support at 43800."
                }
            }
        },

        "smc_metrics": {
            "description": "Smart Money Concepts scoring metrics",
            "fields": {
                "structure_score": {
                    "type": "number",
                    "description": "Overall structure quality score",
                    "range": "0-10",
                    "interpretation": {
                        "0-3": "Poor structure, avoid trading",
                        "4-6": "Average structure, trade with caution",
                        "7-10": "Excellent structure, high probability setups"
                    }
                },
                "momentum_score": {
                    "type": "number",
                    "description": "Momentum strength score based on recent price action",
                    "range": "0-10"
                },
                "clarity_score": {
                    "type": "number",
                    "description": "How clear and readable the market structure is",
                    "range": "0-10"
                },
                "overall_bias": {
                    "type": "enum",
                    "description": "Recommended overall trading bias",
                    "possible_values": {
                        "BULLISH": "Look for long opportunities",
                        "BEARISH": "Look for short opportunities",
                        "NEUTRAL": "No clear bias, avoid trading or wait"
                    }
                }
            }
        },

        "usage_examples": {
            "bullish_scenario": {
                "description": "Example of bullish market structure",
                "indicators": [
                    "trend: BULLISH",
                    "trend_strength: 8",
                    "structure_clarity: 7",
                    "Recent Bullish BOS events",
                    "swing_pattern: HH-HL",
                    "structure_intact: true"
                ]
            },
            "bearish_scenario": {
                "description": "Example of bearish market structure",
                "indicators": [
                    "trend: BEARISH",
                    "trend_strength: 7",
                    "Recent Bearish CHoCH event",
                    "swing_pattern: LH-LL",
                    "overall_bias: BEARISH"
                ]
            }
        },

        "trading_interpretation": {
            "how_to_use": [
                "1. Check overall_bias for directional bias",
                "2. Verify structure_intact is true before trading",
                "3. Use invalidation_level for stop loss placement",
                "4. Target confirmation_level for take profit",
                "5. Higher trend_strength and structure_clarity indicate higher probability trades",
                "6. BOS events suggest trend continuation, CHoCH events suggest reversal"
            ],
            "risk_management": [
                "Always place stops beyond invalidation_level",
                "Avoid trading when structure_clarity < 5",
                "Wait for structure confirmation after CHoCH events",
                "Use position sizing based on structure_score"
            ]
        }
    }
