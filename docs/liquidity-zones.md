# 流動性區域 API 文檔 (整合 smart-money-concepts)

## API 端點

```
GET /api/liquidity-zones/{symbol}/{timeframe}
```

## 參數說明

- **symbol**: 交易對 (例如: BTCUSDT)
- **timeframe**: 時間框架 (例如: 1h, 4h, 1d)

## Query Parameters

- as_of_datetime: 可選的 ISO 8601 datetime 字串 (例如: 2024-01-01T00:00:00Z)

## 實作方法

### 1. 核心函數調用

```python
# 獲取 OHLC 數據
ohlc_data = fetch_ohlc_data(symbol, timeframe, limit=200)

# 檢測擺動高低點（流動性分析的基礎）
swing_highs_lows = smc.swing_highs_lows(ohlc_data, swing_length=50)

# 識別流動性區域（等量高低點）
liquidity_zones = smc.liquidity(ohlc_data, swing_highs_lows, range_percent=0.01)

# 獲取關鍵時間框架的高低點
daily_hl = smc.previous_high_low(ohlc_data, time_frame="1D")
weekly_hl = smc.previous_high_low(ohlc_data, time_frame="1W")
monthly_hl = smc.previous_high_low(ohlc_data, time_frame="1M")

# 獲取 session 高低點（作為流動性參考）
asia_session = smc.sessions(ohlc_data, "Tokyo", "00:00", "08:00", "UTC")
london_session = smc.sessions(ohlc_data, "London", "08:00", "16:00", "UTC")
ny_session = smc.sessions(ohlc_data, "New York", "13:00", "21:00", "UTC")
```

### 2. 流動性分類邏輯

- **未觸及流動性（Untapped）**: 尚未被價格觸及的等量高/低點
- **已清算流動性（Swept）**: 已被價格穿越的流動性區域
- **部分清算（Partially Swept）**: 價格觸及但未完全穿越
- **強弱評估**: 根據觸及次數和形成時間評估

### 3. 數據處理和優先級排序

- 結合多個時間框架的關鍵位
- 計算流動性磁吸強度（基於距離和未清算時間）
- 優先顯示最近且最重要的流動性池

## 回傳 JSON 格式

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "timestamp": "2025-01-03T10:00:00Z",
  "current_price": 95000,
  "liquidity_pools": {
    "buy_side_liquidity": [
      {
        "level": 93500,
        "distance_from_current": "+1.58%",
        "type": "Equal Lows",
        "status": "UNTAPPED",
        "formation_candles": [
          -45,
          -32,
          -28
        ],
        "age_hours": 45,
        "strength": 8,
        "sweep_probability": "HIGH"
      },
      {
        "level": 92000,
        "distance_from_current": "+3.26%",
        "type": "Previous Day Low",
        "status": "UNTAPPED",
        "age_hours": 20,
        "strength": 7,
        "sweep_probability": "MEDIUM"
      }
    ],
    "sell_side_liquidity": [
      {
        "level": 96500,
        "distance_from_current": "-1.55%",
        "type": "Equal Highs",
        "status": "PARTIALLY_SWEPT",
        "formation_candles": [
          -24,
          -16,
          -12
        ],
        "sweep_candle": -8,
        "age_hours": 24,
        "strength": 9,
        "sweep_probability": "VERY_HIGH"
      },
      {
        "level": 97000,
        "distance_from_current": "-2.06%",
        "type": "Previous Week High",
        "status": "UNTAPPED",
        "age_hours": 96,
        "strength": 10,
        "sweep_probability": "HIGH"
      }
    ]
  },
  "timeframe_highs_lows": {
    "daily": {
      "previous_high": 95800,
      "previous_low": 93200,
      "current_position": "INSIDE_RANGE",
      "high_distance": "-0.84%",
      "low_distance": "+1.93%"
    },
    "weekly": {
      "previous_high": 97000,
      "previous_low": 91000,
      "current_position": "MID_RANGE",
      "high_distance": "-2.06%",
      "low_distance": "+4.40%"
    },
    "monthly": {
      "previous_high": 99500,
      "previous_low": 88000,
      "current_position": "UPPER_THIRD",
      "high_distance": "-4.52%",
      "low_distance": "+7.95%"
    }
  },
  "session_liquidity": {
    "asia": {
      "high": 94800,
      "low": 94200,
      "high_swept": false,
      "low_swept": true,
      "session_range": 600
    },
    "london": {
      "high": 95200,
      "low": 94500,
      "high_swept": false,
      "low_swept": false,
      "session_range": 700
    },
    "new_york": {
      "high": 95500,
      "low": 94800,
      "high_swept": true,
      "low_swept": false,
      "session_range": 700
    }
  },
  "liquidity_analysis": {
    "primary_target": {
      "direction": "SELL_SIDE",
      "level": 96500,
      "reason": "Strong equal highs with 3 touches, high probability sweep"
    },
    "secondary_target": {
      "direction": "BUY_SIDE",
      "level": 93500,
      "reason": "Untapped equal lows, potential bounce zone"
    },
    "liquidity_imbalance": "SELL_SIDE_HEAVY",
    "recommended_bias": "BULLISH_SWEEP_LIKELY",
    "key_message": "Price sitting between untapped liquidity. Sell-side sweep at 96500 likely before reversal."
  },
  "liquidity_metrics": {
    "total_buy_side_pools": 4,
    "total_sell_side_pools": 3,
    "nearest_untapped_above": 96500,
    "nearest_untapped_below": 93500,
    "liquidity_void_zones": [
      {
        "range": [
          93800,
          94200
        ],
        "description": "Thin liquidity zone, fast moves expected"
      }
    ]
  }
}
```

## 欄位說明

### liquidity_pools

- 使用 `smc.liquidity()` 識別的流動性區域
- **type**: "Equal Highs/Lows" 來自 liquidity 函數，其他類型來自 previous_high_low
- **formation_candles**: 形成等量高/低的K線索引
- **strength**: 1-10 評分，基於觸及次數和時間
- **sweep_probability**: 基於市場結構和距離計算

### timeframe_highs_lows

- 直接使用 `smc.previous_high_low()` 的輸出
- **current_position**: 當前價格相對於區間的位置
- 提供多時間框架的參考

### session_liquidity

- 使用 `smc.sessions()` 獲取各交易時段數據
- 追蹤時段高低點是否被清算
- 幫助識別時段型流動性獵取

### liquidity_analysis

- 綜合分析最可能的流動性目標
- **liquidity_imbalance**: 評估買賣側流動性分布
- 提供清晰的交易偏向建議

## 實作程式碼片段

```python
def analyze_liquidity_zones(ohlc_data):
    # 獲取基礎數據
    swing_hl = smc.swing_highs_lows(ohlc_data, swing_length=50)
    liquidity = smc.liquidity(ohlc_data, swing_hl, range_percent=0.01)

    # 獲取多時間框架高低點
    timeframes = ['1D', '1W', '1M']
    tf_levels = {}
    for tf in timeframes:
        tf_levels[tf] = smc.previous_high_low(ohlc_data, time_frame=tf)

    # 處理流動性數據
    current_price = ohlc_data['close'].iloc[-1]

    # 分類和評估流動性強度
    buy_side_liquidity = process_liquidity_data(
        liquidity[liquidity['Liquidity'] == 'Low'],
        current_price,
        ohlc_data
    )

    sell_side_liquidity = process_liquidity_data(
        liquidity[liquidity['Liquidity'] == 'High'],
        current_price,
        ohlc_data
    )

    return formatted_response


def calculate_sweep_probability(liquidity_level, current_price, market_structure):
    """
    計算流動性被清算的概率
    考慮因素：距離、形成時間、市場結構方向
    """
    distance = abs(liquidity_level - current_price) / current_price
    # ... 概率計算邏輯
    return probability
```

## 總結

此API提供了完整的流動性分析框架，幫助識別價格最可能移動的目標區域，為GPT提供清晰的流動性狀況視圖。