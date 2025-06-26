import pandas as pd
from typing import Dict, Any, List
from smartmoneyconcepts import smc


class TechnicalAnalyzer:
    """
    Performs technical analysis using smartmoneyconcepts library
    """
    
    def __init__(self):
        pass
        
    def prepare_dataframe(self, ohlcv_data: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Convert OHLCV data to pandas DataFrame for analysis
        
        Args:
            ohlcv_data: Dictionary containing OHLCV data
            
        Returns:
            Pandas DataFrame with proper datetime index
        """
        df = pd.DataFrame(ohlcv_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> List[float]:
        """
        Calculate RSI using smartmoneyconcepts
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            
        Returns:
            RSI values
        """
        rsi = smc.rsi(df['close'], period)
        return rsi.tolist()
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """
        Calculate MACD using smartmoneyconcepts
        
        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        macd_line, macd_signal, macd_histogram = smc.macd(df['close'], fast, slow, signal)
        
        return {
            "macd_line": macd_line.tolist(),
            "signal_line": macd_signal.tolist(),
            "histogram": macd_histogram.tolist()
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: int = 2) -> Dict[str, List[float]]:
        """
        Calculate Bollinger Bands using smartmoneyconcepts
        
        Args:
            df: DataFrame with OHLCV data
            period: Moving average period
            std: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        upper, middle, lower = smc.bollinger_bands(df['close'], period, std)
        
        return {
            "upper_band": upper.tolist(),
            "middle_band": middle.tolist(),
            "lower_band": lower.tolist()
        }
    
    def analyze_all_indicators(self, ohlcv_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate all technical indicators
        
        Args:
            ohlcv_data: Dictionary containing OHLCV data
            
        Returns:
            Dictionary with all calculated indicators
        """
        df = self.prepare_dataframe(ohlcv_data)
        
        indicators = {
            "rsi": self.calculate_rsi(df),
            "macd": self.calculate_macd(df),
            "bollinger_bands": self.calculate_bollinger_bands(df)
        }
        
        return indicators