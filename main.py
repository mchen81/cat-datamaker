from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from binance_data import BinanceDataFetcher
from smc_analysis import SMCAnalyzer
from technical_analysis import TechnicalAnalyzer

app = FastAPI(
    title="Cryptocurrency Technical Analysis API",
    description="A comprehensive API for analyzing cryptocurrency data using technical indicators and Smart Money Concepts (SMC)",
    version="1.0.0",
    contact={
        "name": "DataMaker API",
        "url": "https://github.com/your-repo/DataMaker",
    },
    license_info={
        "name": "MIT",
    },
)


class AnalyzeRequest(BaseModel):
    symbol: str = Field(
        ...,
        description="Cryptocurrency symbol to analyze (e.g., 'BTC', 'ETH', 'BTCUSDT')",
        example="BTC",
        min_length=2,
        max_length=20
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC"
            }
        }


class AnalyzeResponse(BaseModel):
    symbol: str = Field(..., description="The analyzed trading pair symbol")
    data: Dict[str, Any] = Field(..., description="Raw OHLCV data from Binance")
    technical_indicators: Dict[str, Any] = Field(...,
                                                 description="Traditional technical indicators (RSI, MACD, Bollinger Bands)")
    smc_indicators: Dict[str, Any] = Field(...,
                                           description="Smart Money Concepts indicators (FVG, Order Blocks, Liquidity, etc.)")
    timestamp: str = Field(..., description="ISO timestamp when the analysis was performed")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "data": {
                    "open": [45000.0, 45100.0],
                    "high": [45200.0, 45300.0],
                    "low": [44900.0, 45000.0],
                    "close": [45100.0, 45250.0],
                    "volume": [1000.5, 1200.3],
                    "timestamp": [1640995200000, 1640998800000]
                },
                "technical_indicators": {
                    "rsi": [65.5, 67.2],
                    "macd": {
                        "macd_line": [100.5, 105.2],
                        "signal_line": [98.3, 102.1],
                        "histogram": [2.2, 3.1]
                    },
                    "bollinger_bands": {
                        "upper_band": [46000.0, 46100.0],
                        "middle_band": [45000.0, 45100.0],
                        "lower_band": [44000.0, 44100.0]
                    }
                },
                "smc_indicators": {
                    "fair_value_gaps": {
                        "bullish_fvg": [],
                        "bearish_fvg": []
                    },
                    "swing_highs_lows": {
                        "swing_highs": [],
                        "swing_lows": []
                    }
                },
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


@app.get("/",
         summary="API Health Check",
         description="Returns basic information about the API",
         response_description="API status and information")
async def root():
    """Root endpoint that returns API information and status."""
    return {
        "message": "Cryptocurrency Technical Analysis API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "/docs": "Swagger UI documentation",
            "/redoc": "ReDoc documentation",
            "/v1/analyze": "Main analysis endpoint"
        }
    }


@app.post("/v1/analyze",
          response_model=AnalyzeResponse,
          summary="Analyze Cryptocurrency",
          description="Fetch cryptocurrency data from Binance and perform comprehensive technical analysis including traditional indicators and Smart Money Concepts",
          response_description="Complete analysis results with technical and SMC indicators")
async def analyze_crypto(request: AnalyzeRequest):
    """
    Analyze cryptocurrency data with comprehensive technical indicators.
    
    This endpoint:
    1. Fetches the latest 90 1-hour candles from Binance for the specified symbol
    2. Calculates traditional technical indicators (RSI, MACD, Bollinger Bands)
    3. Performs Smart Money Concepts analysis (Fair Value Gaps, Order Blocks, Liquidity, etc.)
    4. Returns all data and calculated indicators
    
    **Supported symbols:** Any valid Binance trading pair (e.g., BTC, ETH, ADA, etc.)
    **Note:** If the symbol doesn't end with 'USDT', it will be automatically appended.
    """
    try:
        # Initialize analyzers
        binance_fetcher = BinanceDataFetcher()
        technical_analyzer = TechnicalAnalyzer()
        smc_analyzer = SMCAnalyzer()

        # Fetch data from Binance (90 klines, 1 hour timeframe)
        symbol = request.symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        klines_data = await binance_fetcher.get_klines(symbol=symbol, interval="1h", limit=90)
        formatted_data = binance_fetcher.format_klines_data(klines_data)

        # Run technical analysis
        technical_indicators = technical_analyzer.analyze_all_indicators(formatted_data)

        # Run SMC analysis
        smc_indicators = smc_analyzer.analyze_all_smc(formatted_data)

        # Return combined results
        return {
            "symbol": symbol,
            "data": formatted_data,
            "technical_indicators": technical_indicators,
            "smc_indicators": smc_indicators,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
