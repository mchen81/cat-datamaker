from fastapi import FastAPI

from api_get_smc_all import router as smc_all_router
from api_killzone_sessions import router as killzone_sessions_router
from api_liquidity_zones import router as liquidity_zones_router
from api_market_structure import router as market_structure_router
from api_mtf_structure import router as mtf_structure_router
from api_supply_demand_zones import router as supply_demand_zones_router
from api_technical_analysis import router as technical_analysis_router
from api_trade_signal import router as trade_signal_router

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

# Include routers
app.include_router(smc_all_router)  # Add comprehensive SMC endpoint first
app.include_router(market_structure_router)
app.include_router(liquidity_zones_router)
app.include_router(supply_demand_zones_router)
app.include_router(killzone_sessions_router)
app.include_router(mtf_structure_router)
app.include_router(technical_analysis_router)
app.include_router(trade_signal_router)


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
            "/api/smc-all/{symbol}/{timeframe}": "Comprehensive SMC analysis (all modules combined)",
            "/api/market-structure/{symbol}/{timeframe}": "Market structure analysis endpoint",
            "/api/liquidity-zones/{symbol}/{timeframe}": "Liquidity zones analysis endpoint",
            "/api/supply-demand-zones/{symbol}/{timeframe}": "Supply demand zones analysis endpoint",
            "/api/killzone-sessions/{symbol}/{timeframe}": "Killzone sessions analysis endpoint",
            "/api/mtf-structure/{symbol}": "Multi-timeframe structure analysis endpoint",
            "/api/technical-indicators/{symbol}/{timeframe}": "Technical indicators analysis endpoint",
            "/api/trade-signal/{symbol}/{timeframe}": "Comprehensive trade signal generation endpoint"
        }
    }
