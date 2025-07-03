# 供需區域 API 文檔 (整合 smart-money-concepts)

## API 端點

```
GET /api/supply-demand-zones/{symbol}/{timeframe}
```

## 參數說明

- **symbol**: 交易對 (例如: BTCUSDT)
- **timeframe**: 時間框架 (例如: 1h, 4h, 1d)

## 實作方法

### 1. 核心函數調用

```python
# 獲取 OHLC 數據
ohlc_data = fetch_ohlc_data(symbol, timeframe, limit=300)

# 檢測擺動高低點（Order Block 識別的基礎）
swing_highs_lows = smc.swing_highs_lows(ohlc_data, swing_length=50)

# 識別 Order Blocks
order_blocks = smc.ob(ohlc_data, swing_highs_lows, close_mitigation=False)

# 識別 Fair Value Gaps (FVG)
fvg = smc.fvg(ohlc_data, join_consecutive=False)

# 獲取 BOS/CHoCH 用於識別 Breaker Blocks
bos_choch = smc.bos_choch(ohlc_data, swing_highs_lows, close_break=True)
```

### 2. 供需區域分類邏輯

- **Order Block (OB)**: 機構訂單聚集區
- **Fair Value Gap (FVG)**: 價格失衡區域
- **Breaker Block**: 失效的 OB 轉換為反向區域
- **區域強度評估**: 基於成交量、反應次數、形成方式

### 3. 區域有效性判定

- **未觸及（Fresh）**: 形成後未被測試
- **已測試（Tested）**: 被觸及但產生反應
- **已緩解（Mitigated）**: 被價格完全穿越
- **區域合併**: 相近的區域合併為一個強區域

## 回傳 JSON 格式

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "timestamp": "2025-01-03T10:00:00Z",
  "current_price": 95000,
  "order_blocks": {
    "bullish_ob": [
      {
        "top": 94500,
        "bottom": 94200,
        "ob_type": "Bullish OB",
        "formation_index": -24,
        "age_hours": 24,
        "status": "TESTED_HOLDING",
        "test_count": 1,
        "volume_profile": "HIGH",
        "strength": 8,
        "distance_from_current": "+0.53%",
        "zone_height": 300,
        "confluence": [
          "Previous Resistance",
          "50% Retracement"
        ]
      },
      {
        "top": 92800,
        "bottom": 92500,
        "ob_type": "Bullish OB",
        "formation_index": -48,
        "age_hours": 48,
        "status": "FRESH",
        "test_count": 0,
        "volume_profile": "EXTREME",
        "strength": 10,
        "distance_from_current": "+2.37%",
        "zone_height": 300,
        "confluence": [
          "Weekly Support",
          "Liquidity Sweep Point"
        ]
      }
    ],
    "bearish_ob": [
      {
        "top": 96800,
        "bottom": 96500,
        "ob_type": "Bearish OB",
        "formation_index": -16,
        "age_hours": 16,
        "status": "FRESH",
        "test_count": 0,
        "volume_profile": "MODERATE",
        "strength": 7,
        "distance_from_current": "-1.58%",
        "zone_height": 300,
        "confluence": [
          "Equal Highs"
        ]
      }
    ]
  },
  "fair_value_gaps": {
    "bullish_fvg": [
      {
        "top": 94300,
        "bottom": 94100,
        "type": "Bullish FVG",
        "formation_index": -12,
        "age_hours": 12,
        "status": "PARTIALLY_FILLED",
        "mitigation_index": -8,
        "fill_percentage": 30,
        "distance_from_current": "+0.74%",
        "gap_size": 200
      }
    ],
    "bearish_fvg": [
      {
        "top": 95800,
        "bottom": 95600,
        "type": "Bearish FVG",
        "formation_index": -6,
        "age_hours": 6,
        "status": "FRESH",
        "mitigation_index": null,
        "fill_percentage": 0,
        "distance_from_current": "-0.63%",
        "gap_size": 200
      }
    ]
  },
  "breaker_blocks": [
    {
      "level": 93800,
      "type": "Bullish Breaker",
      "original_ob_type": "Bearish OB",
      "break_candle_index": -20,
      "age_hours": 20,
      "status": "ACTIVE",
      "strength": 6,
      "distance_from_current": "+1.28%",
      "description": "Former resistance turned support after BOS"
    }
  ],
  "zone_analysis": {
    "nearest_demand_zone": {
      "zone": [
        94200,
        94500
      ],
      "type": "Bullish OB",
      "strength": 8,
      "distance": "+0.53%",
      "recommendation": "Strong support zone, expect bounce"
    },
    "nearest_supply_zone": {
      "zone": [
        96500,
        96800
      ],
      "type": "Bearish OB",
      "strength": 7,
      "distance": "-1.58%",
      "recommendation": "First resistance, potential reversal zone"
    },
    "strongest_zone": {
      "zone": [
        92500,
        92800
      ],
      "type": "Bullish OB",
      "reason": "Extreme volume, untested, multiple confluences"
    },
    "zone_density": "BALANCED",
    "premium_discount": "EQUILIBRIUM",
    "key_message": "Price in equilibrium between supply and demand. Watch for reaction at 94200-94500 support."
  },
  "confluence_matrix": {
    "high_probability_zones": [
      {
        "zone": [
          94200,
          94500
        ],
        "confluences": [
          "Bullish OB",
          "Tested Support",
          "Above FVG"
        ],
        "score": 8.5
      },
      {
        "zone": [
          96500,
          96800
        ],
        "confluences": [
          "Bearish OB",
          "Equal Highs",
          "Liquidity"
        ],
        "score": 7.8
      }
    ]
  },
  "supply_demand_metrics": {
    "total_demand_zones": 4,
    "total_supply_zones": 2,
    "fresh_zones_count": 3,
    "mitigated_today": 1,
    "zone_imbalance": "DEMAND_HEAVY",
    "avg_zone_height": 285,
    "strongest_zone_distance": "+2.37%"
  }
}
```

## 欄位說明

### order_blocks

- 直接映射 `smc.ob()` 的輸出
- **volume_profile**: 分析 OB 形成時的成交量特徵
- **confluence**: 與其他技術水平的匯合點
- **zone_height**: OB 區域的價格範圍

### fair_value_gaps

- 使用 `smc.fvg()` 識別的價格失衡
- **fill_percentage**: FVG 被填補的程度
- **mitigation_index**: 被緩解的K線索引

### breaker_blocks

- 基於 BOS/CHoCH 後的失效 OB
- 需要結合 `smc.ob()` 和 `smc.bos_choch()` 數據
- 追蹤原始 OB 類型和突破信息

### zone_analysis

- **premium_discount**: 基於區間分析的市場定位
- 提供最近和最強區域的快速參考
- 包含可操作的建議

## 實作程式碼片段

```python
def analyze_supply_demand_zones(ohlc_data):
    # 獲取供需區域數據
    swing_hl = smc.swing_highs_lows(ohlc_data, swing_length=50)
    order_blocks = smc.ob(ohlc_data, swing_hl, close_mitigation=False)
    fvg = smc.fvg(ohlc_data, join_consecutive=False)
    bos_choch = smc.bos_choch(ohlc_data, swing_hl, close_break=True)

    current_price = ohlc_data['close'].iloc[-1]

    # 處理 Order Blocks
    bullish_obs = process_order_blocks(
        order_blocks[order_blocks['OB'] == 'Bullish OB'],
        current_price,
        ohlc_data
    )

    bearish_obs = process_order_blocks(
        order_blocks[order_blocks['OB'] == 'Bearish OB'],
        current_price,
        ohlc_data
    )

    # 識別 Breaker Blocks
    breaker_blocks = identify_breaker_blocks(
        order_blocks,
        bos_choch,
        current_price
    )

    # 計算區域強度和匯合點
    zone_strength = calculate_zone_strength(
        order_blocks,
        fvg,
        ohlc_data['volume']
    )

    return formatted_response


def identify_breaker_blocks(order_blocks, bos_choch, current_price):
    """
    識別因 BOS/CHoCH 而失效並反轉的 Order Blocks
    """
    breakers = []
    for _, bos in bos_choch.iterrows():
        # 檢查被突破的 OB
        broken_obs = order_blocks[
            (order_blocks['Top'] < bos['Level']) &
            (order_blocks.index < bos.name)
            ]
        # 將失效的 OB 轉換為 Breaker Block
        # ... 處理邏輯
    return breakers


def calculate_zone_strength(ob, fvg, volume_data):
    """
    基於多個因素計算區域強度
    - 成交量
    - 區域大小
    - 形成方式（快速移動 vs 盤整）
    - 測試次數
    """
    # ... 強度計算邏輯
    return strength_scores
```

## 總結

這個API提供了全面的供需區域分析，結合Order Blocks、Fair Value Gaps和Breaker Blocks，為GPT提供清晰的市場供需動態視圖，幫助識別高概率的反轉和支撐阻力區域。