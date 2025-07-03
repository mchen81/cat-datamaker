from fastapi import FastAPI

from api_liquidity_zones import router as liquidity_zones_router
from api_market_structure import router as market_structure_router

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
app.include_router(market_structure_router)
app.include_router(liquidity_zones_router)


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
            "/api/market-structure/{symbol}/{timeframe}": "Market structure analysis endpoint"
        }
    }
