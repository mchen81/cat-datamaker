# 多時間框架（MTF）結構 API 文檔

## API 端點

```
GET /api/mtf-structure/{symbol}
```

## 參數說明

- **symbol**: 交易對 (例如: BTCUSDT)
- 無需指定時間框架，API 會自動返回多個時間框架的數據

## 實作方法

### 1. 時間框架層級

- **高級時間框架 (HTF)**: 月線、週線 - 主要趨勢方向
- **中級時間框架 (MTF)**: 日線、4小時 - 交易偏向
- **低級時間框架 (LTF)**: 1小時、15分鐘 - 入場時機

### 2. 分析重點

- 每個時間框架的市場結構狀態
- 時間框架之間的對齊程度
- 關鍵級別在多個時間框架的重要性
- 從高到低的級聯分析

### 3. 數據同步策略

- 確保所有時間框架數據的時間一致性
- 識別跨時間框架的匯合點
- 簡化輸出，避免信息過載

## 回傳 JSON 格式

```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2025-01-03T14:00:00Z",
  "current_price": 95000,
  "timeframe_analysis": {
    "monthly": {
      "timeframe": "1M",
      "trend": "BULLISH",
      "structure_status": "INTACT",
      "last_major_move": "BULLISH_BOS",
      "key_levels": {
        "major_resistance": 102000,
        "major_support": 85000,
        "current_position": "MIDDLE_RANGE"
      },
      "strength_score": 8,
      "bias": "STRONG_BULLISH",
      "phase": "EXPANSION"
    },
    "weekly": {
      "timeframe": "1W",
      "trend": "BULLISH",
      "structure_status": "INTACT",
      "last_major_move": "BULLISH_BOS",
      "key_levels": {
        "resistance": 98000,
        "support": 92000,
        "pivot": 95000,
        "current_position": "AT_PIVOT"
      },
      "strength_score": 7,
      "bias": "BULLISH",
      "phase": "PULLBACK",
      "confluence_with_monthly": true
    },
    "daily": {
      "timeframe": "1D",
      "trend": "BULLISH",
      "structure_status": "QUESTIONING",
      "last_major_move": "BEARISH_CHoCH_ATTEMPT",
      "key_levels": {
        "resistance": 96500,
        "support": 93500,
        "pivot": 95000,
        "current_position": "NEAR_PIVOT"
      },
      "strength_score": 6,
      "bias": "NEUTRAL_BULLISH",
      "phase": "CONSOLIDATION",
      "notes": "Potential CHoCH, watching for confirmation"
    },
    "h4": {
      "timeframe": "4H",
      "trend": "RANGING",
      "structure_status": "BROKEN",
      "last_major_move": "BEARISH_CHoCH",
      "key_levels": {
        "range_high": 95500,
        "range_low": 94500,
        "current_position": "MID_RANGE"
      },
      "strength_score": 5,
      "bias": "NEUTRAL",
      "phase": "ACCUMULATION",
      "notes": "Building cause after CHoCH"
    },
    "h1": {
      "timeframe": "1H",
      "trend": "BEARISH",
      "structure_status": "INTACT",
      "last_major_move": "BEARISH_BOS",
      "key_levels": {
        "resistance": 95200,
        "support": 94800,
        "current_position": "NEAR_RESISTANCE"
      },
      "strength_score": 4,
      "bias": "BEARISH",
      "phase": "RETRACEMENT",
      "entry_quality": "POOR"
    }
  },
  "mtf_alignment": {
    "alignment_score": 6.5,
    "alignment_status": "PARTIAL",
    "aligned_timeframes": [
      "1M",
      "1W",
      "1D"
    ],
    "conflicting_timeframes": [
      "4H",
      "1H"
    ],
    "dominant_bias": "BULLISH",
    "conflict_resolution": "HIGHER_TF_PRIORITY",
    "trading_recommendation": "WAIT_FOR_LTF_ALIGNMENT"
  },
  "key_level_confluence": {
    "major_confluence_zones": [
      {
        "zone": [
          94500,
          94800
        ],
        "timeframes_present": [
          "1D",
          "4H",
          "1H"
        ],
        "type": "SUPPORT",
        "strength": 9,
        "description": "Multi-timeframe support confluence"
      },
      {
        "zone": [
          96500,
          96800
        ],
        "timeframes_present": [
          "1D",
          "4H"
        ],
        "type": "RESISTANCE",
        "strength": 8,
        "description": "Daily and 4H resistance alignment"
      }
    ],
    "single_tf_levels": [
      {
        "level": 98000,
        "timeframe": "1W",
        "importance": "HIGH"
      }
    ]
  },
  "mtf_structure_summary": {
    "primary_trend": "BULLISH",
    "trading_timeframe_trend": "RANGING",
    "entry_timeframe_trend": "BEARISH",
    "structural_integrity": {
      "htf": "STRONG",
      "mtf": "MODERATE",
      "ltf": "WEAK"
    },
    "best_trading_approach": "PATIENT_ACCUMULATION",
    "ideal_entry_scenario": "Wait for 4H bullish structure break"
  },
  "cascade_analysis": {
    "monthly_to_weekly": {
      "alignment": "CONFIRMED",
      "weekly_respecting_monthly": true
    },
    "weekly_to_daily": {
      "alignment": "CONFIRMED",
      "daily_respecting_weekly": true
    },
    "daily_to_4h": {
      "alignment": "DIVERGING",
      "potential_shift": true,
      "watch_level": 95500
    },
    "4h_to_1h": {
      "alignment": "TRANSITIONING",
      "accumulation_phase": true
    }
  },
  "trading_zones": {
    "optimal_buy_zone": {
      "range": [
        94500,
        94800
      ],
      "timeframe_support": [
        "1D",
        "4H"
      ],
      "risk_reward": "EXCELLENT",
      "entry_criteria": "4H bullish structure shift"
    },
    "optimal_sell_zone": {
      "range": [
        96500,
        97000
      ],
      "timeframe_resistance": [
        "1W",
        "1D"
      ],
      "risk_reward": "GOOD",
      "entry_criteria": "1H bearish structure at resistance"
    },
    "no_trade_zone": {
      "range": [
        94900,
        95100
      ],
      "reason": "Mid-range, no edge"
    }
  },
  "mtf_bias_matrix": {
    "current_bias": "BULLISH_WITH_CAUTION",
    "confidence": 7,
    "key_message": "HTF bullish but LTF showing weakness. Wait for 4H structure repair before long entries.",
    "invalidation_scenarios": [
      "Daily close below 93500",
      "Weekly close below 92000"
    ],
    "confirmation_scenarios": [
      "4H reclaims 95500",
      "1H forms higher low above 94800"
    ]
  },
  "timeframe_transitions": {
    "next_htf_decision": {
      "timeframe": "1W",
      "time_to_close": "3 days",
      "critical_level": 92000
    },
    "next_mtf_decision": {
      "timeframe": "1D",
      "time_to_close": "10 hours",
      "critical_level": 94500
    },
    "immediate_focus": {
      "timeframe": "4H",
      "next_candle": "2 hours",
      "watch_for": "Break above 95500"
    }
  }
}
```

## 欄位說明

### timeframe_analysis

- 每個時間框架的獨立分析
- **structure_status**: INTACT | QUESTIONING | BROKEN
- **phase**: 當前市場階段（趨勢/盤整/回調等）
- **strength_score**: 該時間框架的趨勢強度

### mtf_alignment

- 衡量多個時間框架的一致性
- **alignment_score**: 0-10 的對齊程度
- 識別衝突和解決方案

### key_level_confluence

- 跨時間框架的關鍵位匯合
- 評估每個區域的強度
- 幫助識別高概率反轉區

### cascade_analysis

- 分析趨勢如何從高時間框架傳遞到低時間框架
- 識別潛在的結構轉變點
- 預警可能的趨勢變化

### trading_zones

- 基於 MTF 分析的具體交易區域
- 包含入場標準和風險評估
- 明確標示無交易區域

## 實作概念

```python
# MTF 分析邏輯示例（偽代碼）

def analyze_mtf_structure(symbol):
    """分析多時間框架結構"""
    timeframes = ['1M', '1W', '1D', '4H', '1H']
    tf_data = {}

    for tf in timeframes:
        # 獲取每個時間框架的數據
        structure = analyze_single_timeframe(symbol, tf)
        tf_data[tf] = structure

    # 計算對齊度
    alignment = calculate_alignment(tf_data)

    # 識別匯合區
    confluences = find_confluences(tf_data)

    # 級聯分析
    cascade = analyze_cascade_effect(tf_data)

    return format_mtf_response(tf_data, alignment, confluences, cascade)


def calculate_alignment(tf_data):
    """計算時間框架對齊度"""
    # 比較各時間框架的趨勢方向
    # 返回對齊分數和狀態
    pass


def analyze_cascade_effect(tf_data):
    """分析級聯效應"""
    # 檢查高時間框架對低時間框架的影響
    # 識別潛在的結構傳導
    pass
```

## 設計原則

### 層級清晰

- 從高到低的結構化分析
- 明確每個時間框架的角色

### 衝突解決

- 當時間框架不一致時，提供清晰指導
- 優先考慮高時間框架

### 實用性

- 提供具體的交易區域
- 包含時間要素（下一個關鍵時間點）

### 簡化決策

- 綜合評分和明確建議
- 避免信息過載

## 總結

這個 API 設計提供了完整的多時間框架視角，幫助 GPT 理解不同時間維度的市場結構，並據此做出更準確的交易決策。通過結構化的數據呈現，使複雜的
MTF 分析變得易於處理。