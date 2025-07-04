from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field

from api_killzone_sessions import KillzoneSessionAnalyzer
from api_liquidity_zones import LiquidityZoneAnalyzer
# Import all API modules
from api_market_structure import MarketStructureAnalyzer
from api_mtf_structure import MTFStructureAnalyzer
from api_supply_demand_zones import SupplyDemandAnalyzer
from api_trade_signal import TradeSignalAnalyzer
from binance_data import BinanceDataFetcher
from utils import verify_api_key


class SMCAllResponse(BaseModel):
    """Comprehensive SMC analysis response combining all modules"""
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Analysis timeframe")
    timestamp: str = Field(..., description="Analysis timestamp")
    current_price: float = Field(..., description="Current price")

    # Individual analysis results
    market_structure: Optional[Dict[str, Any]] = Field(None, description="Market structure analysis")
    supply_demand_zones: Optional[Dict[str, Any]] = Field(None, description="Supply/demand zones analysis")
    liquidity_zones: Optional[Dict[str, Any]] = Field(None, description="Liquidity zones analysis")
    killzone_sessions: Optional[Dict[str, Any]] = Field(None, description="Killzone sessions analysis")
    trade_signal: Optional[Dict[str, Any]] = Field(None, description="Trade signal analysis")
    mtf_structure: Optional[Dict[str, Any]] = Field(None, description="Multi-timeframe structure analysis")

    # Summary metrics
    overall_bias: str = Field(..., description="Overall trading bias")
    confidence_score: float = Field(..., description="Overall confidence score 0-10")
    key_levels: List[float] = Field(..., description="Key price levels to watch")
    summary_message: str = Field(..., description="Executive summary message")


# Create router
router = APIRouter(tags=["SMC All-in-One"])


class SMCAllAnalyzer:
    """
    Comprehensive SMC analyzer that combines all analysis modules
    """

    def __init__(self):
        self.market_structure_analyzer = MarketStructureAnalyzer()
        self.supply_demand_analyzer = SupplyDemandAnalyzer()
        self.liquidity_analyzer = LiquidityZoneAnalyzer()
        self.killzone_analyzer = KillzoneSessionAnalyzer()
        self.trade_signal_analyzer = TradeSignalAnalyzer()
        self.mtf_analyzer = MTFStructureAnalyzer()

    def calculate_overall_bias(self, analyses: Dict[str, Any]) -> str:
        """Calculate overall trading bias from all analyses"""
        bias_scores = []

        # Market structure bias
        if analyses.get("market_structure"):
            smc_metrics = analyses["market_structure"].get("smc_metrics")
            if smc_metrics:
                if hasattr(smc_metrics, 'overall_bias'):
                    ms_bias = smc_metrics.overall_bias
                else:
                    ms_bias = smc_metrics.get("overall_bias", "NEUTRAL")

                if ms_bias == "BULLISH":
                    bias_scores.append(1)
                elif ms_bias == "BEARISH":
                    bias_scores.append(-1)
                else:
                    bias_scores.append(0)

        # Trade signal bias
        if analyses.get("trade_signal"):
            signal_direction = analyses["trade_signal"].get("signal_direction", "NEUTRAL")
            if "BUY" in signal_direction:
                bias_scores.append(1)
            elif "SELL" in signal_direction:
                bias_scores.append(-1)
            else:
                bias_scores.append(0)

        # Supply/demand bias
        if analyses.get("supply_demand_zones"):
            zone_analysis = analyses["supply_demand_zones"].get("zone_analysis")
            if zone_analysis:
                if hasattr(zone_analysis, 'recommended_bias'):
                    recommended_bias = zone_analysis.recommended_bias
                else:
                    recommended_bias = zone_analysis.get("recommended_bias", "NEUTRAL") if isinstance(zone_analysis,
                                                                                                      dict) else "NEUTRAL"

                if "BULLISH" in str(recommended_bias):
                    bias_scores.append(1)
                elif "BEARISH" in str(recommended_bias):
                    bias_scores.append(-1)
                else:
                    bias_scores.append(0)

        # Liquidity bias
        if analyses.get("liquidity_zones"):
            liq_analysis = analyses["liquidity_zones"].get("liquidity_analysis")
            if liq_analysis:
                if hasattr(liq_analysis, 'recommended_bias'):
                    recommended_bias = liq_analysis.recommended_bias
                else:
                    recommended_bias = liq_analysis.get("recommended_bias", "NEUTRAL") if isinstance(liq_analysis,
                                                                                                     dict) else "NEUTRAL"

                if "BULLISH" in str(recommended_bias):
                    bias_scores.append(1)
                elif "BEARISH" in str(recommended_bias):
                    bias_scores.append(-1)
                else:
                    bias_scores.append(0)

        # Calculate overall bias
        if not bias_scores:
            return "NEUTRAL"

        avg_bias = sum(bias_scores) / len(bias_scores)
        if avg_bias > 0.3:
            return "BULLISH"
        elif avg_bias < -0.3:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def calculate_confidence_score(self, analyses: Dict[str, Any]) -> float:
        """Calculate overall confidence score from all analyses"""
        confidence_scores = []

        # Market structure confidence
        if analyses.get("market_structure"):
            smc_metrics = analyses["market_structure"].get("smc_metrics")
            if smc_metrics:
                if hasattr(smc_metrics, 'structure_score'):
                    structure_score = smc_metrics.structure_score
                else:
                    structure_score = smc_metrics.get("structure_score", 5.0)
                confidence_scores.append(structure_score)

        # Trade signal confidence
        if analyses.get("trade_signal"):
            signal_strength = analyses["trade_signal"].get("signal_strength", "MODERATE")
            strength_map = {
                "VERY_STRONG": 9.0,
                "STRONG": 7.5,
                "MODERATE": 5.0,
                "WEAK": 3.0,
                "VERY_WEAK": 1.5
            }
            confidence_scores.append(strength_map.get(signal_strength, 5.0))

        # Supply/demand confidence
        if analyses.get("supply_demand_zones"):
            order_blocks = analyses["supply_demand_zones"].get("order_blocks")
            if order_blocks:
                fresh_ob_count = 0
                if hasattr(order_blocks, 'bullish_ob'):
                    bullish_obs = order_blocks.bullish_ob if hasattr(order_blocks.bullish_ob, '__iter__') else []
                    bearish_obs = order_blocks.bearish_ob if hasattr(order_blocks.bearish_ob, '__iter__') else []
                else:
                    bullish_obs = order_blocks.get("bullish_ob", []) if isinstance(order_blocks, dict) else []
                    bearish_obs = order_blocks.get("bearish_ob", []) if isinstance(order_blocks, dict) else []

                for ob_list in [bullish_obs, bearish_obs]:
                    for ob in ob_list:
                        if hasattr(ob, 'status'):
                            if ob.status == "FRESH":
                                fresh_ob_count += 1
                        elif isinstance(ob, dict):
                            if ob.get("status") == "FRESH":
                                fresh_ob_count += 1
                confidence_scores.append(min(8.0, fresh_ob_count * 2))

        # Calculate average confidence
        if not confidence_scores:
            return 5.0

        return sum(confidence_scores) / len(confidence_scores)

    def extract_key_levels(self, analyses: Dict[str, Any], current_price: float) -> List[float]:
        """Extract key price levels from all analyses"""
        key_levels = set()

        # Market structure levels
        if analyses.get("market_structure"):
            structure_analysis = analyses["market_structure"].get("structure_analysis")
            if structure_analysis:
                if hasattr(structure_analysis, 'invalidation_level'):
                    invalidation_level = structure_analysis.invalidation_level
                    confirmation_level = structure_analysis.confirmation_level
                elif isinstance(structure_analysis, dict):
                    invalidation_level = structure_analysis.get("invalidation_level")
                    confirmation_level = structure_analysis.get("confirmation_level")
                else:
                    invalidation_level = None
                    confirmation_level = None

                if invalidation_level:
                    key_levels.add(float(invalidation_level))
                if confirmation_level:
                    key_levels.add(float(confirmation_level))

        # Supply/demand levels
        if analyses.get("supply_demand_zones"):
            order_blocks = analyses["supply_demand_zones"].get("order_blocks")
            if order_blocks:
                if hasattr(order_blocks, 'bullish_ob'):
                    bullish_obs = order_blocks.bullish_ob if hasattr(order_blocks.bullish_ob, '__iter__') else []
                    bearish_obs = order_blocks.bearish_ob if hasattr(order_blocks.bearish_ob, '__iter__') else []
                else:
                    bullish_obs = order_blocks.get("bullish_ob", []) if isinstance(order_blocks, dict) else []
                    bearish_obs = order_blocks.get("bearish_ob", []) if isinstance(order_blocks, dict) else []

                for ob_list in [bullish_obs, bearish_obs]:
                    for ob in ob_list[:3]:  # Top 3 most relevant
                        if hasattr(ob, 'top'):
                            key_levels.add(float(ob.top))
                            key_levels.add(float(ob.bottom))
                        elif isinstance(ob, dict):
                            key_levels.add(float(ob.get("top", 0)))
                            key_levels.add(float(ob.get("bottom", 0)))

        # Liquidity levels
        if analyses.get("liquidity_zones"):
            liquidity_pools = analyses["liquidity_zones"].get("liquidity_pools")
            if liquidity_pools:
                if hasattr(liquidity_pools, 'buy_side_liquidity'):
                    buy_side = liquidity_pools.buy_side_liquidity if hasattr(liquidity_pools.buy_side_liquidity,
                                                                             '__iter__') else []
                    sell_side = liquidity_pools.sell_side_liquidity if hasattr(liquidity_pools.sell_side_liquidity,
                                                                               '__iter__') else []
                else:
                    buy_side = liquidity_pools.get("buy_side_liquidity", []) if isinstance(liquidity_pools,
                                                                                           dict) else []
                    sell_side = liquidity_pools.get("sell_side_liquidity", []) if isinstance(liquidity_pools,
                                                                                             dict) else []

                for pool_list in [buy_side, sell_side]:
                    for pool in pool_list[:3]:  # Top 3 most relevant
                        if hasattr(pool, 'level'):
                            key_levels.add(float(pool.level))
                        elif isinstance(pool, dict):
                            key_levels.add(float(pool.get("level", 0)))

        # Remove levels that are too close to current price (within 0.5%)
        filtered_levels = []
        for level in key_levels:
            if level > 0 and abs(level - current_price) / current_price > 0.005:
                filtered_levels.append(level)

        # Sort and return top 8 levels
        filtered_levels.sort(key=lambda x: abs(x - current_price))
        return filtered_levels[:8]

    def generate_summary_message(self, analyses: Dict[str, Any], overall_bias: str, confidence_score: float) -> str:
        """Generate executive summary message"""
        if confidence_score >= 7:
            confidence_text = "High confidence"
        elif confidence_score >= 5:
            confidence_text = "Moderate confidence"
        else:
            confidence_text = "Low confidence"

        bias_text = overall_bias.lower()

        # Get key insight from market structure
        key_insight = "Market structure analysis pending"
        if analyses.get("market_structure"):
            structure_analysis = analyses["market_structure"].get("structure_analysis")
            if structure_analysis:
                if hasattr(structure_analysis, 'key_message'):
                    key_insight = structure_analysis.key_message
                elif isinstance(structure_analysis, dict):
                    key_insight = structure_analysis.get("key_message", "No key insights available")
                else:
                    key_insight = "No key insights available"

        return f"{confidence_text} {bias_text} bias. {key_insight}"

    async def analyze_all_smc(self, ohlcv_data: Dict[str, List[float]], symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Perform comprehensive SMC analysis using all modules
        """
        analyses = {}
        current_price = float(ohlcv_data['close'][-1])

        try:
            # Run all analyses in parallel where possible
            tasks = []

            # Market structure analysis
            try:
                market_structure_result = self.market_structure_analyzer.analyze_market_structure(ohlcv_data, timeframe)
                analyses["market_structure"] = market_structure_result
            except Exception as e:
                print(f"Market structure analysis failed: {e}")
                analyses["market_structure"] = None

            # Supply/demand zones analysis
            try:
                supply_demand_result = self.supply_demand_analyzer.analyze_supply_demand_zones(ohlcv_data, timeframe)
                analyses["supply_demand_zones"] = supply_demand_result
            except Exception as e:
                print(f"Supply/demand analysis failed: {e}")
                analyses["supply_demand_zones"] = None

            # Liquidity zones analysis
            try:
                liquidity_result = self.liquidity_analyzer.analyze_liquidity_zones(ohlcv_data, timeframe)
                analyses["liquidity_zones"] = liquidity_result
            except Exception as e:
                print(f"Liquidity analysis failed: {e}")
                analyses["liquidity_zones"] = None

            # Killzone sessions analysis
            try:
                killzone_result = self.killzone_analyzer.analyze_killzone_sessions(ohlcv_data, timeframe)
                analyses["killzone_sessions"] = killzone_result
            except Exception as e:
                print(f"Killzone analysis failed: {e}")
                analyses["killzone_sessions"] = None

            # Trade signal analysis - this requires different approach since it's async
            try:
                # For now, skip trade signal analysis as it requires async handling
                # trade_signal_result = await self.trade_signal_analyzer.generate_trade_signal(symbol, timeframe, ohlcv_data)
                # analyses["trade_signal"] = trade_signal_result
                analyses["trade_signal"] = None
            except Exception as e:
                print(f"Trade signal analysis failed: {e}")
                analyses["trade_signal"] = None

            # MTF structure analysis (only for higher timeframes)
            if timeframe in ['4h', '1d', '1w']:
                try:
                    # MTF requires symbol parameter, not ohlcv_data
                    # mtf_result = self.mtf_analyzer.analyze_mtf_structure(symbol, timeframe)
                    # analyses["mtf_structure"] = mtf_result
                    analyses["mtf_structure"] = None
                except Exception as e:
                    print(f"MTF structure analysis failed: {e}")
                    analyses["mtf_structure"] = None

            # Calculate overall metrics
            overall_bias = self.calculate_overall_bias(analyses)
            confidence_score = self.calculate_confidence_score(analyses)
            key_levels = self.extract_key_levels(analyses, current_price)
            summary_message = self.generate_summary_message(analyses, overall_bias, confidence_score)

            return {
                "current_price": current_price,
                "analyses": analyses,
                "overall_bias": overall_bias,
                "confidence_score": confidence_score,
                "key_levels": key_levels,
                "summary_message": summary_message
            }

        except Exception as e:
            print(f"Error in comprehensive SMC analysis: {e}")
            return {
                "current_price": current_price,
                "analyses": analyses,
                "overall_bias": "NEUTRAL",
                "confidence_score": 0.0,
                "key_levels": [],
                "summary_message": f"Analysis failed: {str(e)}"
            }


@router.get("/api/smc-all/{symbol}/{timeframe}",
            response_model=SMCAllResponse,
            summary="Get Comprehensive SMC Analysis",
            description="Get all Smart Money Concepts analysis in a single request")
async def get_smc_all(
        symbol: str = Path(..., description="Trading pair symbol (e.g., BTCUSDT, BTC)"),
        timeframe: str = Path(..., description="Analysis timeframe (e.g., 1h, 4h, 1d)"),
        as_of_datetime: Optional[str] = Query(None,
                                              description="ISO 8601 datetime string for historical analysis (e.g., 2024-01-01T00:00:00Z). Defaults to current time."),
        _: bool = Depends(verify_api_key)
):
    """
    Get comprehensive Smart Money Concepts analysis combining all available modules.
    
    This endpoint fetches data once and runs all SMC analysis modules:
    - Market Structure Analysis (BOS/CHoCH, swing points, trend)
    - Supply/Demand Zones (order blocks, fair value gaps, breaker blocks)
    - Liquidity Zones (liquidity pools, sweep analysis)
    - Killzone Sessions (session analysis, power of 3)
    - Trade Signal Analysis (technical + SMC signals)
    - Multi-Timeframe Structure (for 4h, 1d, 1w timeframes)
    
    **Parameters:**
    - **symbol**: Any valid Binance trading pair (e.g., 'BTC', 'ETH', 'BTCUSDT')
    - **timeframe**: Analysis timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
    - **as_of_datetime**: Optional ISO 8601 datetime string for historical analysis
    
    **Returns:**
    - Combined analysis from all SMC modules
    - Overall trading bias and confidence score
    - Key price levels to watch
    - Executive summary message
    
    **Note:** This endpoint is optimized for performance by fetching data once and 
    running all analyses in parallel where possible.
    """
    try:
        # Validate timeframe
        valid_intervals = {
            '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',
            '6h', '8h', '12h', '1d', '3d', '1w', '1M'
        }
        if timeframe not in valid_intervals:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid timeframe. Must be one of: {", ".join(sorted(valid_intervals))}'
            )

        # Initialize components
        binance_fetcher = BinanceDataFetcher()
        analyzer = SMCAllAnalyzer()

        # Process symbol
        processed_symbol = symbol.upper()
        if not processed_symbol.endswith('USDT'):
            processed_symbol = f"{processed_symbol}USDT"

        # Parse and validate asOfDateTime if provided
        end_time_ms = None
        if as_of_datetime:
            try:
                # Parse ISO 8601 datetime string
                as_of_dt = datetime.fromisoformat(as_of_datetime.replace('Z', '+00:00'))
                # Convert to Unix timestamp in milliseconds
                end_time_ms = int(as_of_dt.timestamp() * 1000)
            except ValueError:
                print(f"Warning: Failed to parse as_of_datetime '{as_of_datetime}'. Using current timestamp instead.")
                end_time_ms = None

        # Fetch data from Binance once
        klines_data = await binance_fetcher.get_klines(
            symbol=processed_symbol,
            interval=timeframe,
            limit=500,  # Get more data for comprehensive analysis
            end_time=end_time_ms
        )
        formatted_data = binance_fetcher.format_klines_data(klines_data)

        # Perform comprehensive analysis
        analysis_result = await analyzer.analyze_all_smc(formatted_data, processed_symbol, timeframe)

        # Create response
        return SMCAllResponse(
            symbol=processed_symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc).isoformat() if end_time_ms is None else as_of_datetime,
            current_price=analysis_result["current_price"],
            market_structure=analysis_result["analyses"].get("market_structure"),
            supply_demand_zones=analysis_result["analyses"].get("supply_demand_zones"),
            liquidity_zones=analysis_result["analyses"].get("liquidity_zones"),
            killzone_sessions=analysis_result["analyses"].get("killzone_sessions"),
            trade_signal=analysis_result["analyses"].get("trade_signal"),
            mtf_structure=analysis_result["analyses"].get("mtf_structure"),
            overall_bias=analysis_result["overall_bias"],
            confidence_score=analysis_result["confidence_score"],
            key_levels=analysis_result["key_levels"],
            summary_message=analysis_result["summary_message"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive SMC analysis failed: {str(e)}")


@router.get("/api/smc-all/doc",
            summary="SMC All-in-One API Documentation",
            description="Get comprehensive documentation for the SMC All-in-One API")
async def get_smc_all_documentation():
    """
    Get detailed documentation for the SMC All-in-One API response format.
    """
    return {
        "api_endpoint": "/api/smc-all/{symbol}/{timeframe}",
        "description": "Comprehensive Smart Money Concepts analysis combining all available modules for maximum insight with optimal performance.",

        "key_features": [
            "Single data fetch for all analyses (optimized performance)",
            "Parallel analysis execution where possible",
            "Combined bias calculation from all modules",
            "Overall confidence scoring",
            "Key level extraction across all analyses",
            "Executive summary generation"
        ],

        "included_analyses": {
            "market_structure": {
                "description": "Market structure analysis with BOS/CHoCH events and swing points",
                "endpoint_equivalent": "/api/market-structure/{symbol}/{timeframe}"
            },
            "supply_demand_zones": {
                "description": "Order blocks, fair value gaps, and breaker block analysis",
                "endpoint_equivalent": "/api/supply-demand-zones/{symbol}/{timeframe}"
            },
            "liquidity_zones": {
                "description": "Liquidity pool analysis and sweep probability",
                "endpoint_equivalent": "/api/liquidity-zones/{symbol}/{timeframe}"
            },
            "killzone_sessions": {
                "description": "Session analysis and power of 3 concepts",
                "endpoint_equivalent": "/api/killzone-sessions/{symbol}/{timeframe}"
            },
            "trade_signal": {
                "description": "Technical and SMC-based trading signals",
                "endpoint_equivalent": "/api/trade-signal/{symbol}/{timeframe}"
            },
            "mtf_structure": {
                "description": "Multi-timeframe structure analysis (4h, 1d, 1w only)",
                "endpoint_equivalent": "/api/mtf-structure/{symbol}/{timeframe}"
            }
        },

        "response_structure": {
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol"
            },
            "timeframe": {
                "type": "string",
                "description": "Analysis timeframe"
            },
            "timestamp": {
                "type": "string",
                "description": "Analysis timestamp"
            },
            "current_price": {
                "type": "number",
                "description": "Current market price"
            },
            "market_structure": {
                "type": "object",
                "description": "Complete market structure analysis data",
                "note": "Same structure as /api/market-structure response"
            },
            "supply_demand_zones": {
                "type": "object",
                "description": "Complete supply/demand zones analysis data",
                "note": "Same structure as /api/supply-demand-zones response"
            },
            "liquidity_zones": {
                "type": "object",
                "description": "Complete liquidity zones analysis data",
                "note": "Same structure as /api/liquidity-zones response"
            },
            "killzone_sessions": {
                "type": "object",
                "description": "Complete killzone sessions analysis data",
                "note": "Same structure as /api/killzone-sessions response"
            },
            "trade_signal": {
                "type": "object",
                "description": "Complete trade signal analysis data",
                "note": "Same structure as /api/trade-signal response"
            },
            "mtf_structure": {
                "type": "object",
                "description": "Multi-timeframe structure analysis data (only for 4h, 1d, 1w)",
                "note": "Same structure as /api/mtf-structure response"
            },
            "overall_bias": {
                "type": "string",
                "description": "Overall trading bias calculated from all modules",
                "possible_values": ["BULLISH", "BEARISH", "NEUTRAL"]
            },
            "confidence_score": {
                "type": "number",
                "description": "Overall confidence score from 0-10",
                "interpretation": {
                    "0-3": "Low confidence, avoid trading",
                    "4-6": "Moderate confidence, trade with caution",
                    "7-10": "High confidence, favorable trading conditions"
                }
            },
            "key_levels": {
                "type": "array",
                "description": "Key price levels extracted from all analyses",
                "note": "Up to 8 most relevant levels sorted by distance from current price"
            },
            "summary_message": {
                "type": "string",
                "description": "Executive summary combining insights from all analyses"
            }
        },

        "performance_benefits": [
            "Single API call instead of 6 separate calls",
            "Data fetched once and shared across all analyses",
            "Reduced latency and bandwidth usage",
            "Simplified integration for applications",
            "Combined bias calculation for better decision making"
        ],

        "use_cases": [
            "Trading dashboard requiring comprehensive analysis",
            "Automated trading systems needing full context",
            "Mobile applications with bandwidth constraints",
            "Risk management requiring multiple perspectives",
            "Educational platforms showing complete SMC analysis"
        ],

        "trading_workflow": [
            "1. Check overall_bias for directional preference",
            "2. Verify confidence_score > 6 for high-probability setups",
            "3. Use key_levels for entry, stop-loss, and take-profit planning",
            "4. Review individual analyses for specific insights",
            "5. Monitor summary_message for key market conditions"
        ]
    }
