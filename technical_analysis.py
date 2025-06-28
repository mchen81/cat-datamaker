import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, List


class TechnicalAnalyzer:
    """
    Performs technical analysis using pandas-ta library
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
        Calculate RSI using pandas-ta
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            
        Returns:
            RSI values
        """
        rsi = ta.rsi(df['close'], length=period)
        return rsi.dropna().tolist()
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """
        Calculate MACD using pandas-ta
        
        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        macd_data = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
        
        return {
            "macd_line": macd_data[f'MACD_{fast}_{slow}_{signal}'].dropna().tolist(),
            "signal_line": macd_data[f'MACDs_{fast}_{slow}_{signal}'].dropna().tolist(),
            "histogram": macd_data[f'MACDh_{fast}_{slow}_{signal}'].dropna().tolist()
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: int = 2) -> Dict[str, List[float]]:
        """
        Calculate Bollinger Bands using pandas-ta
        
        Args:
            df: DataFrame with OHLCV data
            period: Moving average period
            std: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        bb_data = ta.bbands(df['close'], length=period, std=std)
        
        return {
            "upper_band": bb_data[f'BBU_{period}_{std}.0'].dropna().tolist(),
            "middle_band": bb_data[f'BBM_{period}_{std}.0'].dropna().tolist(),
            "lower_band": bb_data[f'BBL_{period}_{std}.0'].dropna().tolist()
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