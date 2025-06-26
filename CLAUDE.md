# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a BTC data analysis API built with FastAPI that:
- Fetches latest BTC data from Binance API
- Performs technical analysis using smartmoneyconcepts library
- Returns calculated indicators like RSI, MACD, and Bollinger Bands

### API Endpoints
- `GET /` - Root endpoint returning API information
- `POST /v1/analyze` - Main endpoint that analyzes BTC data and returns technical indicators

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
- `binance_data.py` - Module for fetching BTC data from Binance API
- `technical_analysis.py` - Module for calculating technical indicators using smartmoneyconcepts

### Data Flow
1. API receives POST request to `/v1/analyze`
2. `BinanceDataFetcher` fetches latest BTC OHLCV data from Binance
3. `TechnicalAnalyzer` processes data and calculates indicators (RSI, MACD, Bollinger Bands)
4. API returns formatted response with raw data and calculated indicators

## Used Third Party Libraries
Here are the third party libraries available for technical analysis. Make sure you use context7 when writing code from them:
- **smartmoneyconcepts** (https://github.com/joshyattridge/smart-money-concepts) - Smart Money Concepts indicators
- **stockstats** - Stock statistics and technical analysis
- **pandas-ta** - Pandas Technical Analysis library

### Technical Analysis Options
You can use any of these libraries for calculations:
- `smartmoneyconcepts` for advanced trading concepts
- `stockstats` for traditional technical indicators
- `pandas_ta` for comprehensive technical analysis
