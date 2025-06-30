import os

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader

from binance_data import BinanceDataFetcher
from smc_analysis import SMCAnalyzer
from technical_analysis import TechnicalAnalyzer
from utils import *

# Get API key from environment variable
API_KEY = os.getenv("API_KEY", "")

# Define API key security scheme for Swagger
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key for protected endpoints"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

app = FastAPI(
    title="Cryptocurrency Technical Analysis API",
    description="A comprehensive API for analyzing cryptocurrency data using technical indicators and Smart Money Concepts (SMC)",
    version="1.0.0",
    contact={
        "name": "DataMaker API",
        "url": "https://github.com/mchen81/cat-datamaker",
    },
    license_info={
        "name": "MIT",
    },
)

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
            "/v1/analyze": "Comprehensive analysis endpoint",
            "/v1/trading-signal": "GPT-optimized trading signals endpoint"
        }
    }


@app.post("/v1/analyze",
          # response_model=AnalyzeResponse,
          summary="Analyze Cryptocurrency",
          description="Fetch cryptocurrency data from Binance and perform comprehensive technical analysis including traditional indicators and Smart Money Concepts",
          response_description="Complete analysis results with technical and SMC indicators")
async def analyze_crypto(request: AnalyzeRequest, _: bool = Depends(verify_api_key)):
    """
    Analyze cryptocurrency data with comprehensive technical indicators.
    
    This endpoint:
    1. Fetches klines data from Binance for the specified symbol, interval, and limit
    2. Calculates traditional technical indicators (RSI, MACD, Bollinger Bands)
    3. Performs Smart Money Concepts analysis (Fair Value Gaps, Order Blocks, Liquidity, etc.)
    4. Returns all data and calculated indicators
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., BTC, ETH, ADA, etc.)
    - **interval**: Optional kline interval (default: 1h) - Valid values: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    - **limit**: Optional number of klines to fetch (default: 90, max: 1000)
    
    **Note:** If the symbol doesn't end with 'USDT', it will be automatically appended.
    """
    try:
        # Initialize analyzers
        binance_fetcher = BinanceDataFetcher()
        technical_analyzer = TechnicalAnalyzer()
        smc_analyzer = SMCAnalyzer()

        # Fetch data from Binance using request parameters
        symbol = request.symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        klines_data = await binance_fetcher.get_klines(
            symbol=symbol,
            interval=request.interval,
            limit=request.limit
        )
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


@app.post("/v1/trading-signal",
          response_model=TradingSignalResponse,
          summary="Get Trading Signal for GPT",
          description="Fetch cryptocurrency data and return GPT-optimized trading signals with key decision points",
          response_description="Concise trading signal optimized for automated decision making")
async def get_trading_signal(request: AnalyzeRequest, _: bool = Depends(verify_api_key)):
    """
    Get GPT-optimized trading signals for automated decision making.
    
    This endpoint:
    1. Fetches klines data from Binance for the specified symbol, interval, and limit
    2. Performs comprehensive technical and SMC analysis (same as /v1/analyze)
    3. Processes the data into concise, decision-focused signals
    4. Returns key indicators, trends, and overall trading recommendation
    
    **Optimized for GPT consumption - uses ~200-500 tokens vs ~2000+ tokens from /v1/analyze**
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., BTC, ETH, ADA, etc.)
    - **interval**: Optional kline interval (default: 1h) - Valid values: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    - **limit**: Optional number of klines to fetch (default: 90, max: 1000)
    
    **Returns:**
    - Current market data and price changes
    - Technical indicator signals (RSI, MACD, Bollinger Bands)
    - Smart Money Concepts summary (key levels, liquidity)
    - Overall trading signal with confidence score
    
    **Note:** If the symbol doesn't end with 'USDT', it will be automatically appended.
    """
    try:
        # Initialize analyzers (same as existing endpoint)
        binance_fetcher = BinanceDataFetcher()
        technical_analyzer = TechnicalAnalyzer()
        smc_analyzer = SMCAnalyzer()

        # Fetch data from Binance using request parameters
        symbol = request.symbol.upper()
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        klines_data = await binance_fetcher.get_klines(
            symbol=symbol,
            interval=request.interval,
            limit=request.limit
        )
        formatted_data = binance_fetcher.format_klines_data(klines_data)

        # Run technical analysis
        technical_indicators = technical_analyzer.analyze_all_indicators(formatted_data)

        # Run SMC analysis
        smc_indicators = smc_analyzer.analyze_all_smc(formatted_data)

        # Process data for GPT optimization
        trading_signal = process_for_trading_decision(
            symbol=symbol,
            formatted_data=formatted_data,
            technical_indicators=technical_indicators,
            smc_indicators=smc_indicators,
            timeframe=request.interval
        )

        return trading_signal

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trading signal analysis failed: {str(e)}")
