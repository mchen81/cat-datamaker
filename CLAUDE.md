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
- `utils.py` - Helper functions for common tasks, such as validate API key
- `binance_data.py` - Module for fetching cryptocurrency data from Binance API
- `technical_analysis.py` - Module for calculating traditional technical indicators
- `smc_analysis.py` - Module for Smart Money Concepts analysis
- `test_main.http` - HTTP test file for testing endpoints
- `api_market_structure.py` - An API for analyzing market structure, detailed in `docs/market-structure.md`

## Use Context7 when using Third-Party Libraries

Here are the third party libraries available for technical analysis. Make sure you use context7 when writing code from
them:

- **smartmoneyconcepts** (https://github.com/joshyattridge/smart-money-concepts) - Smart Money Concepts indicators
- **stockstats** - Stock statistics and technical analysis
- **pandas-ta** - Pandas Technical Analysis library

### Technical Analysis Options

You can use any of these libraries for calculations:

- `smartmoneyconcepts` for advanced trading concepts
- `stockstats` for traditional technical indicators
- `pandas_ta` for comprehensive technical analysis

### Signal Calculation Methodology

- **Technical Indicators (40% weight)**: RSI, MACD, Bollinger Bands analysis
- **Trend Analysis (30% weight)**: Short-term and medium-term trend alignment
- **Price Momentum (20% weight)**: Recent price action strength
- **Volume Confirmation (10% weight)**: Volume trend supporting signals
