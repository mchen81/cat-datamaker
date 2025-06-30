# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency data analysis API built with FastAPI that:

- Fetches latest cryptocurrency data from Binance API
- Performs comprehensive technical analysis using smartmoneyconcepts library
- Provides Smart Money Concepts (SMC) analysis
- Returns both detailed analysis and GPT-optimized trading signals

### API Endpoints
- `GET /` - Root endpoint returning API information
- `POST /v1/analyze` - Comprehensive analysis endpoint that returns full technical indicators and SMC data
- `POST /v1/trading-signal` - GPT-optimized endpoint that returns concise trading signals for automated decision making

## Development Commands

### Installing Dependencies
```bash
# Install from requirements file
pip install -r requirements.txt

# Or install packages individually
pip install fastapi uvicorn aiohttp pandas smartmoneyconcepts stockstats pandas-ta
```

### Running the Application
```bash
# Run the development server
uvicorn main:app --reload

# Run on specific host/port
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
- HTTP test file available at `test_main.http` for testing endpoints
- Default server runs on `http://127.0.0.1:8000`

## Architecture

### File Structure
- `main.py` - FastAPI application with REST endpoints
- `models.py` - Pydantic models for request/response validation and data structures
- `utils.py` - Helper functions for data processing, trend analysis, and signal calculation
- `binance_data.py` - Module for fetching cryptocurrency data from Binance API
- `technical_analysis.py` - Module for calculating traditional technical indicators
- `smc_analysis.py` - Module for Smart Money Concepts analysis
- `test_main.http` - HTTP test file for testing endpoints

### Data Flow

#### `/v1/analyze` - Comprehensive Analysis

1. API receives POST request with symbol, interval, and limit parameters
2. `BinanceDataFetcher` fetches OHLCV data from Binance API
3. `TechnicalAnalyzer` calculates traditional indicators (RSI, MACD, Bollinger Bands)
4. `SMCAnalyzer` performs Smart Money Concepts analysis (FVG, Order Blocks, Liquidity)
5. API returns comprehensive response with all raw data and calculated indicators

#### `/v1/trading-signal` - GPT-Optimized Analysis

1. API receives POST request with same parameters as `/v1/analyze`
2. Performs same comprehensive analysis as above
3. `process_for_trading_decision()` processes data into GPT-optimized format:
    - Extracts key market data (current price, price changes, trends)
    - Summarizes technical indicators into actionable signals
    - Identifies key SMC levels (support/resistance)
    - Calculates overall trading signal and confidence score
4. API returns concise response optimized for automated trading decisions (~45% fewer tokens)

## Use Context7 when using Third-Party Libraries
Here are the third party libraries available for technical analysis. Make sure you use context7 when writing code from them:
- **smartmoneyconcepts** (https://github.com/joshyattridge/smart-money-concepts) - Smart Money Concepts indicators
- **stockstats** - Stock statistics and technical analysis
- **pandas-ta** - Pandas Technical Analysis library

### Technical Analysis Options
You can use any of these libraries for calculations:
- `smartmoneyconcepts` for advanced trading concepts
- `stockstats` for traditional technical indicators
- `pandas_ta` for comprehensive technical analysis

## Trading Signal Endpoint for GPT Integration

### Purpose

The `/v1/trading-signal` endpoint is specifically designed for automated trading workflows where the response will be
fed into ChatGPT or other language models for trading decisions.

### Key Features

- **Token Efficient**: ~45% fewer tokens compared to `/v1/analyze`
- **Decision Focused**: Returns only essential data for buy/sell/wait decisions
- **Structured Signals**: Clear signal strength indicators (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL)
- **Confidence Scoring**: Provides confidence percentage for signal reliability

### Response Structure

```json
{
  "symbol": "BTCUSDT",
  "market_data": {
    "current_price": 45600.0,
    "price_change_1h": 0.22,
    "price_change_4h": 0.77,
    "price_change_24h": 1.5,
    "volume_trend": "increasing",
    "trend_short": "BULLISH",
    "trend_medium": "BULLISH"
  },
  "technical_signals": {
    "rsi_level": 65.0,
    "rsi_signal": "NEUTRAL",
    "macd_signal": "bullish_crossover",
    "bb_position": "above_middle_band",
    "overall_technical": "BUY"
  },
  "smc_summary": {
    "key_support": 45000.0,
    "key_resistance": 46500.0,
    "liquidity_sweep": "recent_lows_swept",
    "fvg_count": 2
  },
  "overall_signal": "BUY",
  "confidence": 75.0,
  "timeframe": "1h",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Usage in Automated Workflows

1. Call `/v1/trading-signal` with desired symbol and timeframe
2. Feed the response to GPT with a prompt like: "Based on this trading signal data, should I buy, sell, or wait?"
3. GPT can make informed decisions using the structured signal data

### Signal Calculation Methodology

- **Technical Indicators (40% weight)**: RSI, MACD, Bollinger Bands analysis
- **Trend Analysis (30% weight)**: Short-term and medium-term trend alignment
- **Price Momentum (20% weight)**: Recent price action strength
- **Volume Confirmation (10% weight)**: Volume trend supporting signals
