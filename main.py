from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
import asyncio

from binance_data import BinanceDataFetcher
from technical_analysis import TechnicalAnalyzer
from smc_analysis import SMCAnalyzer

app = FastAPI()


class AnalyzeRequest(BaseModel):
    symbol: str


class AnalyzeResponse(BaseModel):
    symbol: str
    data: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    smc_indicators: Dict[str, Any]
    timestamp: str


@app.get("/")
async def root():
    return {"message": "BTC Data Analysis API"}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_crypto(request: AnalyzeRequest):
    """
    Analyze cryptocurrency data from Binance with technical and SMC indicators
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
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
