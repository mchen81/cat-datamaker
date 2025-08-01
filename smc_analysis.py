import math
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from smartmoneyconcepts import smc


class SMCAnalyzer:
    """
    Smart Money Concepts analysis using smartmoneyconcepts library
    完全符合庫規範的實現
    """

    def __init__(self):
        pass

    def _format_timestamp(self, timestamp_obj) -> str:
        """
        Safely convert timestamp to ISO format string

        Args:
            timestamp_obj: Can be datetime, Timestamp, or integer

        Returns:
            ISO format timestamp string
        """
        if hasattr(timestamp_obj, 'isoformat'):
            return timestamp_obj.isoformat()
        elif isinstance(timestamp_obj, (int, float)):
            # Convert millisecond timestamp to datetime
            if timestamp_obj > 1e10:  # Likely milliseconds
                return datetime.fromtimestamp(timestamp_obj / 1000).isoformat()
            else:  # Likely seconds
                return datetime.fromtimestamp(timestamp_obj).isoformat()
        else:
            # Fallback: convert to string
            return str(timestamp_obj)

    def _convert_numpy_types(self, obj):
        """
        Recursively convert numpy types to Python native types for JSON serialization

        Args:
            obj: Object that may contain numpy types

        Returns:
            Object with numpy types converted to Python native types
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            val = float(obj)
            # Handle special float values that are not JSON compliant
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        elif isinstance(obj, (int, float)):
            # Handle Python native float types that might be inf or nan
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj

    def prepare_dataframe(self, ohlcv_data: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Convert OHLCV data to pandas DataFrame for SMC analysis

        Args:
            ohlcv_data: Dictionary containing OHLCV data

        Returns:
            Pandas DataFrame with proper datetime index and required columns
        """
        df = pd.DataFrame(ohlcv_data)

        # Filter out invalid timestamps (too small, likely corrupted data)
        # Valid Binance timestamps should be > 1000000000000 (Sep 2001 in ms)
        valid_timestamp_mask = df['timestamp'] > 1000000000000
        df = df[valid_timestamp_mask]

        if df.empty:
            raise ValueError("No valid timestamp data found")

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # smartmoneyconcepts requires lowercase column names
        # No renaming needed as our data is already lowercase

        return df

    def calculate_fair_value_gaps(self, df: pd.DataFrame, join_consecutive: bool = False) -> Dict[str, Any]:
        """
        Calculate Fair Value Gaps (FVG) using smartmoneyconcepts

        Args:
            df: DataFrame with OHLCV data
            join_consecutive: Whether to join consecutive FVGs

        Returns:
            Dictionary with FVG data
        """
        fvg = smc.fvg(df, join_consecutive=join_consecutive)

        # Convert to serializable format
        fvg_data = {
            "bullish_fvg": [],
            "bearish_fvg": []
        }

        if not fvg.empty:
            for idx, row in fvg.iterrows():
                gap_info = {
                    "timestamp": self._format_timestamp(df.index[idx]),
                    "top": self._convert_numpy_types(row.get('Top', 0)),
                    "bottom": self._convert_numpy_types(row.get('Bottom', 0)),
                    "direction": self._convert_numpy_types(row.get('FVG', 0)),
                    "mitigated_index": self._convert_numpy_types(row.get('MitigatedIndex', None))
                }

                if row.get('FVG', 0) == 1:  # Bullish FVG
                    fvg_data["bullish_fvg"].append(gap_info)
                elif row.get('FVG', 0) == -1:  # Bearish FVG
                    fvg_data["bearish_fvg"].append(gap_info)

        return fvg_data

    def calculate_swing_highs_lows(self, df: pd.DataFrame, swing_length: int = 50) -> Dict[str, Any]:
        """
        Calculate Swing Highs and Lows using smartmoneyconcepts

        Args:
            df: DataFrame with OHLCV data
            swing_length: Look back period for swing detection

        Returns:
            Dictionary with swing points data and the swing DataFrame for other functions
        """
        swing_result = smc.swing_highs_lows(df, swing_length=swing_length)

        swing_data = {
            "swing_highs": [],
            "swing_lows": [],
            "swing_dataframe": swing_result  # Store for use in other functions
        }

        if not swing_result.empty:
            for idx, row in swing_result.iterrows():
                if row.get('HighLow', 0) == 1:  # Swing High
                    swing_data["swing_highs"].append({
                        "timestamp": self._format_timestamp(df.index[idx]),
                        "price": self._convert_numpy_types(row.get('Level', 0))
                    })
                elif row.get('HighLow', 0) == -1:  # Swing Low
                    swing_data["swing_lows"].append({
                        "timestamp": self._format_timestamp(df.index[idx]),
                        "price": self._convert_numpy_types(row.get('Level', 0))
                    })

        return swing_data

    def calculate_bos_choch(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame, close_break: bool = True) -> Dict[
        str, Any]:
        """
        Calculate Break of Structure (BOS) and Change of Character (CHoCH)

        Args:
            df: DataFrame with OHLCV data
            swing_highs_lows: Swing highs and lows DataFrame from calculate_swing_highs_lows
            close_break: Whether to use close break

        Returns:
            Dictionary with BOS and CHoCH data
        """
        bos_choch = smc.bos_choch(df, swing_highs_lows, close_break=close_break)

        structure_data = {
            "bos": [],
            "choch": []
        }

        if not bos_choch.empty:
            for idx, row in bos_choch.iterrows():
                structure_info = {
                    "timestamp": self._format_timestamp(df.index[idx]),
                    "level": self._convert_numpy_types(row.get('Level', 0)),
                    "broken_index": row.get('BrokenIndex', 0)
                }

                if row.get('BOS', 0) == 1:
                    structure_data["bos"].append(structure_info)
                elif row.get('CHOCH', 0) == 1:
                    structure_data["choch"].append(structure_info)

        return structure_data

    def calculate_order_blocks(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame,
                               close_mitigation: bool = False) -> Dict[str, Any]:
        """
        Calculate Order Blocks (OB) using smartmoneyconcepts

        Args:
            df: DataFrame with OHLCV data
            swing_highs_lows: Swing highs and lows DataFrame from calculate_swing_highs_lows
            close_mitigation: Whether to use close mitigation

        Returns:
            Dictionary with Order Blocks data
        """
        ob = smc.ob(df, swing_highs_lows, close_mitigation=close_mitigation)

        ob_data = {
            "bullish_ob": [],
            "bearish_ob": []
        }

        if not ob.empty:
            for idx, row in ob.iterrows():
                ob_info = {
                    "timestamp": self._format_timestamp(df.index[idx]),
                    "top": self._convert_numpy_types(row.get('Top', 0)),
                    "bottom": self._convert_numpy_types(row.get('Bottom', 0)),
                    "direction": row.get('OB', 0),
                    "volume": self._convert_numpy_types(row.get('OBVolume', 0)),
                    "percentage": self._convert_numpy_types(row.get('Percentage', 0))
                }

                if row.get('OB', 0) == 1:  # Bullish OB
                    ob_data["bullish_ob"].append(ob_info)
                elif row.get('OB', 0) == -1:  # Bearish OB
                    ob_data["bearish_ob"].append(ob_info)

        return ob_data

    def calculate_liquidity(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame, range_percent: float = 0.01) -> \
            Dict[str, Any]:
        """
        Calculate Liquidity levels using smartmoneyconcepts

        Args:
            df: DataFrame with OHLCV data
            swing_highs_lows: Swing highs and lows DataFrame from calculate_swing_highs_lows
            range_percent: Range percentage for liquidity detection

        Returns:
            Dictionary with liquidity data
        """
        liquidity = smc.liquidity(df, swing_highs_lows, range_percent=range_percent)

        liquidity_data = {
            "liquidity_levels": []
        }

        if not liquidity.empty:
            for idx, row in liquidity.iterrows():
                liq_info = {
                    "timestamp": self._format_timestamp(df.index[idx]),
                    "level": self._convert_numpy_types(row.get('Level', 0)),
                    "end": self._convert_numpy_types(row.get('End', None)),
                    "swept": self._convert_numpy_types(row.get('Swept', None)),
                    "liquidity": self._convert_numpy_types(row.get('Liquidity', 0))
                }

                liquidity_data["liquidity_levels"].append(liq_info)

        return liquidity_data

    def calculate_previous_high_low(self, df: pd.DataFrame, time_frame: str = "1D") -> Dict[str, Any]:
        """
        Calculate Previous High and Low levels

        Args:
            df: DataFrame with OHLCV data
            time_frame: Time frame for previous high/low calculation (15m, 1H, 4H, 1D, 1W, 1M)

        Returns:
            Dictionary with previous high/low data
        """
        prev_high_low = smc.previous_high_low(df, time_frame=time_frame)

        phl_data = {
            "previous_highs": [],
            "previous_lows": [],
            "broken_highs": [],
            "broken_lows": []
        }

        if not prev_high_low.empty:
            for idx, row in prev_high_low.iterrows():
                if not pd.isna(row.get('PreviousHigh', None)):
                    phl_data["previous_highs"].append({
                        "timestamp": self._format_timestamp(df.index[idx]),
                        "level": self._convert_numpy_types(row.get('PreviousHigh', 0))
                    })

                if not pd.isna(row.get('PreviousLow', None)):
                    phl_data["previous_lows"].append({
                        "timestamp": self._format_timestamp(df.index[idx]),
                        "level": self._convert_numpy_types(row.get('PreviousLow', 0))
                    })

                if row.get('BrokenHigh', 0) == 1:
                    phl_data["broken_highs"].append({
                        "timestamp": self._format_timestamp(df.index[idx]),
                        "level": self._convert_numpy_types(row.get('PreviousHigh', 0))
                    })

                if row.get('BrokenLow', 0) == 1:
                    phl_data["broken_lows"].append({
                        "timestamp": self._format_timestamp(df.index[idx]),
                        "level": self._convert_numpy_types(row.get('PreviousLow', 0))
                    })

        return phl_data

    def calculate_sessions(self, df: pd.DataFrame, session: str = "London",
                           start_time: str = None, end_time: str = None, time_zone: str = "UTC") -> Dict[str, Any]:
        """
        Calculate Trading Sessions using smartmoneyconcepts

        Args:
            df: DataFrame with OHLCV data
            session: Session name (Sydney, Tokyo, London, New York, Asian kill zone, etc.)
            start_time: Start time for custom session (HH:MM format)
            end_time: End time for custom session (HH:MM format)
            time_zone: Time zone (UTC, UTC+1, etc.)

        Returns:
            Dictionary with session data
        """
        sessions_result = smc.sessions(df, session, start_time, end_time, time_zone=time_zone)

        sessions_data = {
            "session_candles": [],
            "session_highs": [],
            "session_lows": []
        }

        if not sessions_result.empty:
            for idx, row in sessions_result.iterrows():
                if row.get('Active', 0) == 1:
                    sessions_data["session_candles"].append({
                        "timestamp": self._format_timestamp(df.index[idx]),
                        "active": bool(row.get('Active', 0)),
                        "high": self._convert_numpy_types(row.get('High', 0)),
                        "low": self._convert_numpy_types(row.get('Low', 0))
                    })

                if not pd.isna(row.get('High', None)):
                    sessions_data["session_highs"].append({
                        "timestamp": self._format_timestamp(df.index[idx]),
                        "level": self._convert_numpy_types(row.get('High', 0))
                    })

                if not pd.isna(row.get('Low', None)):
                    sessions_data["session_lows"].append({
                        "timestamp": self._format_timestamp(df.index[idx]),
                        "level": self._convert_numpy_types(row.get('Low', 0))
                    })

        return sessions_data

    def calculate_retracements(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Price Retracements using smartmoneyconcepts

        Args:
            df: DataFrame with OHLCV data
            swing_highs_lows: Swing highs and lows DataFrame from calculate_swing_highs_lows

        Returns:
            Dictionary with retracement data
        """
        retracements = smc.retracements(df, swing_highs_lows)

        retracements_data = {
            "retracements": []
        }

        if not retracements.empty:
            for idx, row in retracements.iterrows():
                retracement_info = {
                    "timestamp": self._format_timestamp(df.index[idx]),
                    "direction": self._convert_numpy_types(row.get('Direction', 0)),
                    "current_retracement_percent": self._convert_numpy_types(row.get('CurrentRetracement%', 0)),
                    "deepest_retracement_percent": self._convert_numpy_types(row.get('DeepestRetracement%', 0))
                }

                retracements_data["retracements"].append(retracement_info)

        return retracements_data

    def analyze_all_smc(self, ohlcv_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate all Smart Money Concepts indicators

        Args:
            ohlcv_data: Dictionary containing OHLCV data

        Returns:
            Dictionary with all SMC indicators
        """
        df = self.prepare_dataframe(ohlcv_data)

        # Calculate swing highs and lows first (required for other indicators)
        swing_data = self.calculate_swing_highs_lows(df)
        swing_df = swing_data["swing_dataframe"]

        smc_indicators = {
            "fair_value_gaps": self.calculate_fair_value_gaps(df),
            "swing_highs_lows": {
                "swing_highs": swing_data["swing_highs"],
                "swing_lows": swing_data["swing_lows"]
            },
            "bos_choch": self.calculate_bos_choch(df, swing_df),
            "order_blocks": self.calculate_order_blocks(df, swing_df),
            "liquidity": self.calculate_liquidity(df, swing_df),
            "previous_high_low": self.calculate_previous_high_low(df),
            "sessions": self.calculate_sessions(df),  # 新增
            "retracements": self.calculate_retracements(df, swing_df)  # 新增
        }

        # Convert all numpy types to Python native types for JSON serialization
        return self._convert_numpy_types(smc_indicators)
