# 市場結構 API 文檔 (整合 smart-money-concepts)

## API 端點

`GET /api/market-structure/{symbol}/{timeframe}`

## 參數說明

- symbol: 交易對 (例如: BTCUSDT)
- timeframe: 時間框架 (例如: 1h, 4h, 1d)

## 實作方法

### 1. 核心函數調用

```python
from smartmoneyconcepts import smc

# 檢測擺動高低點
swing_highs_lows = smc.swing_highs_lows(ohlc_data, swing_length=50)
# 識別 BOS 和 CHoCH
bos_choch = smc.bos_choch(ohlc_data, swing_highs_lows, close_break=True)
# 計算回撤百分比（用於趨勢強度評估）
retracements = smc.retracements(ohlc_data, swing_highs_lows)
```

### 2. 趨勢判定邏輯

分析最近的 BOS/CHoCH 事件序列
如果最近 3 個 BOS 都是 "Bullish BOS" → 強勢上升趨勢
如果出現 "Bearish CHoCH" → 趨勢可能轉變
結合回撤百分比評估趨勢強度

### 3. 數據處理和簡化

將 swing_highs_lows DataFrame 轉換為關鍵支撐/阻力位
計算每個關鍵位與當前價格的相對距離
只保留最近且最重要的 2-3 個擺動點

### 回傳 JSON 格式

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "timestamp": "2025-01-03T10:00:00Z",
  "current_price": 95000,
  "market_structure": {
    "trend": "BULLISH",
    "trend_strength": 7,
    "structure_clarity": 8,
    "last_update": "2025-01-03T08:00:00Z",
    "retracement": {
      "direction": "Pullback",
      "current_percent": 38.2,
      "deepest_percent": 45.5
    }
  },
  "recent_bos_choch": [
    {
      "type": "Bullish BOS",
      "level": 94500,
      "distance_from_current": "+0.53%",
      "candle_index": -8,
      "time_ago_hours": 8,
      "broke_level": 94200
    },
    {
      "type": "Bullish BOS",
      "level": 93000,
      "distance_from_current": "+2.15%",
      "candle_index": -24,
      "time_ago_hours": 24,
      "broke_level": 92800
    },
    {
      "type": "Bearish CHoCH",
      "level": 91500,
      "distance_from_current": "+3.83%",
      "candle_index": -48,
      "time_ago_hours": 48,
      "broke_level": 91700
    }
  ],
  "swing_points": {
    "recent_highs": [
      {
        "level": 96000,
        "distance_from_current": "-1.04%",
        "candle_index": -4,
        "age_hours": 4,
        "swing_type": "HH",
        "status": "UNBROKEN_RESISTANCE"
      },
      {
        "level": 94000,
        "distance_from_current": "+1.06%",
        "candle_index": -16,
        "age_hours": 16,
        "swing_type": "LH",
        "status": "BROKEN_NOW_SUPPORT"
      }
    ],
    "recent_lows": [
      {
        "level": 94200,
        "distance_from_current": "+0.85%",
        "candle_index": -12,
        "age_hours": 12,
        "swing_type": "HL",
        "status": "PROTECTED_LOW"
      },
      {
        "level": 92500,
        "distance_from_current": "+2.70%",
        "candle_index": -36,
        "age_hours": 36,
        "swing_type": "LL",
        "status": "KEY_SUPPORT"
      }
    ]
  },
  "structure_analysis": {
    "swing_pattern": "HH-HL",
    "last_significant_move": "Bullish BOS at 94500",
    "structure_intact": true,
    "invalidation_level": 94200,
    "confirmation_level": 96000,
    "key_message": "Bullish structure maintained. Price holding above recent HL at 94200."
  },
  "smc_metrics": {
    "structure_score": 7.5,
    "momentum_score": 6.8,
    "clarity_score": 8.2,
    "overall_bias": "BULLISH"
  }
}
```

# 欄位說明

- market_structure.retracement

直接使用 smc.retracements() 的輸出
direction: "Pullback" 或 "Extension"
百分比數值幫助評估回調深度

- recent_bos_choch

直接映射 smc.bos_choch() 返回的 DataFrame
type: 保持原始輸出 ("Bullish BOS", "Bearish BOS", "Bullish CHoCH", "Bearish CHoCH")
candle_index: 相對於當前的K線索引位置
broke_level: BOS/CHoCH 突破的擺動點價格

- swing_points

從 smc.swing_highs_lows() 提取
swing_type: "HH" (Higher High), "HL" (Higher Low), "LH" (Lower High), "LL" (Lower Low)
根據相對位置判斷 swing 類型

- structure_analysis

swing_pattern: 最近兩個擺動點的模式
structure_intact: 基於 BOS/CHoCH 序列判斷
提供清晰的操作指引