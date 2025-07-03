# Kill Zone 時段 API 文檔 (整合 smart-money-concepts)

## API 端點

```
GET /api/killzone-sessions/{symbol}/{timeframe}
```

## 參數說明

- **symbol**: 交易對 (例如: BTCUSDT)
- **timeframe**: 時間框架 (建議使用: 15m, 1h)

## Query Parameters

- as_of_datetime: 可選的 ISO 8601 datetime 字串 (例如: 2024-01-01T00:00:00Z)

## 實作方法

### 1. 核心函數調用

```python
# 獲取 OHLC 數據 (需要更多數據以涵蓋多個交易時段)
ohlc_data = fetch_ohlc_data(symbol, timeframe, limit=500)

# 獲取各主要交易時段數據
asia_session = smc.sessions(ohlc_data, "Tokyo", "00:00", "08:00", "UTC")
london_session = smc.sessions(ohlc_data, "London", "08:00", "16:00", "UTC")
ny_session = smc.sessions(ohlc_data, "New York", "13:00", "21:00", "UTC")

# 獲取重疊時段 (高波動期)
london_ny_overlap = smc.sessions(ohlc_data, session=None,
                                 start_time="13:00", end_time="16:00",
                                 time_zone="UTC")

# 獲取週開盤和日開盤數據
weekly_open = smc.previous_high_low(ohlc_data, time_frame="1W")
daily_open = smc.previous_high_low(ohlc_data, time_frame="1D")

# Kill Zone 特定時段 (ICT 概念)
asia_killzone = smc.sessions(ohlc_data, session=None,
                             start_time="08:00", end_time="12:00",
                             time_zone="UTC")  # 亞洲 Kill Zone

london_killzone = smc.sessions(ohlc_data, session=None,
                               start_time="07:00", end_time="10:00",
                               time_zone="UTC")  # 倫敦 Kill Zone

ny_killzone = smc.sessions(ohlc_data, session=None,
                           start_time="12:00", end_time="15:00",
                           time_zone="UTC")  # 紐約 Kill Zone
```

### 2. Power of 3 概念分析

- **累積 (Accumulation)**: 亞洲時段的區間震盪
- **操縱 (Manipulation)**: 倫敦時段的假突破
- **分配 (Distribution)**: 紐約時段的真實走勢

### 3. 時段特徵分析

- 計算每個時段的平均波動率
- 識別時段高低點的後續表現
- 追蹤流動性掃除模式

## 回傳 JSON 格式

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "timestamp": "2025-01-03T14:00:00Z",
  "current_price": 95000,
  "current_session": "NEW_YORK",
  "weekly_reference": {
    "weekly_open": 93500,
    "distance_from_open": "+1.60%",
    "current_week_high": 96200,
    "current_week_low": 92800,
    "weekly_bias": "BULLISH",
    "days_into_week": 3
  },
  "daily_reference": {
    "daily_open": 94200,
    "distance_from_open": "+0.85%",
    "daily_high": 95300,
    "daily_low": 93900,
    "daily_range": 1400,
    "range_position": "UPPER_75_PERCENT"
  },
  "session_data": {
    "asia": {
      "session_high": 94500,
      "session_low": 94000,
      "session_range": 500,
      "opening_price": 94100,
      "closing_price": 94400,
      "direction": "BULLISH",
      "high_time": "04:30",
      "low_time": "01:15",
      "volume_profile": "LOW",
      "range_characteristic": "TIGHT",
      "liquidity_taken": "NONE"
    },
    "london": {
      "session_high": 95200,
      "session_low": 94300,
      "session_range": 900,
      "opening_price": 94400,
      "closing_price": 94800,
      "direction": "BULLISH",
      "high_time": "10:45",
      "low_time": "08:15",
      "volume_profile": "HIGH",
      "range_characteristic": "EXPANSION",
      "liquidity_taken": "ASIA_HIGH",
      "manipulation_detected": true,
      "manipulation_type": "STOP_HUNT_LOW"
    },
    "new_york": {
      "session_high": 95300,
      "session_low": 94700,
      "session_range": 600,
      "opening_price": 94800,
      "current_price": 95000,
      "direction": "BULLISH",
      "high_time": "13:30",
      "low_time": "12:15",
      "volume_profile": "VERY_HIGH",
      "range_characteristic": "TRENDING",
      "liquidity_taken": "LONDON_HIGH",
      "session_active": true,
      "time_remaining_hours": 3.5
    }
  },
  "killzone_analysis": {
    "asia_killzone": {
      "time_window": "08:00-12:00 UTC",
      "high": 94450,
      "low": 94050,
      "range": 400,
      "type": "ACCUMULATION",
      "key_levels_formed": [
        94200,
        94350
      ],
      "recommendation": "Range formation, wait for breakout"
    },
    "london_killzone": {
      "time_window": "07:00-10:00 UTC",
      "high": 95100,
      "low": 94200,
      "range": 900,
      "type": "MANIPULATION",
      "false_breakout": true,
      "sweep_direction": "BEARISH",
      "reversal_level": 94300,
      "recommendation": "Stop hunt complete, bullish continuation likely"
    },
    "ny_killzone": {
      "time_window": "12:00-15:00 UTC",
      "high": 95300,
      "low": 94700,
      "range": 600,
      "type": "DISTRIBUTION",
      "trend_continuation": true,
      "target_projection": 95800,
      "recommendation": "True direction confirmed, follow trend"
    }
  },
  "power_of_3": {
    "current_phase": "DISTRIBUTION",
    "accumulation": {
      "session": "ASIA",
      "range": [
        94000,
        94500
      ],
      "completed": true
    },
    "manipulation": {
      "session": "LONDON",
      "sweep_level": 94200,
      "direction": "BEAR_TRAP",
      "completed": true
    },
    "distribution": {
      "session": "NEW_YORK",
      "trend_direction": "BULLISH",
      "in_progress": true,
      "target": 95800
    },
    "po3_confidence": 8.5,
    "pattern_clarity": "HIGH"
  },
  "session_overlaps": {
    "london_ny_overlap": {
      "time_window": "13:00-16:00 UTC",
      "high": 95300,
      "low": 94600,
      "volatility": "EXTREME",
      "directional_bias": "BULLISH",
      "volume_spike": true
    }
  },
  "trading_recommendations": {
    "current_opportunity": "NEW_YORK_CONTINUATION",
    "entry_zone": [
      94800,
      94900
    ],
    "stop_loss": 94600,
    "targets": [
      95300,
      95500,
      95800
    ],
    "session_bias": {
      "intraday": "BULLISH",
      "key_message": "London manipulation complete. NY showing true bullish direction. Look for pullbacks to 94800-94900 for longs."
    },
    "next_key_time": "16:00 UTC",
    "next_event": "London Close"
  },
  "session_statistics": {
    "most_volatile_session": "LONDON",
    "highest_volume_session": "NEW_YORK",
    "avg_daily_range": 1200,
    "current_day_range": 1400,
    "range_expansion": true,
    "session_rhythm": "NORMAL"
  }
}
```

## 欄位說明

### session_data

- 使用 `smc.sessions()` 獲取各時段數據
- **range_characteristic**: TIGHT | NORMAL | EXPANSION
- **manipulation_detected**: 檢測假突破行為
- **liquidity_taken**: 記錄哪個時段的高低點被掃除

### killzone_analysis

- ICT 特定的高概率時間窗口
- 每個 Kill Zone 的行為類型判定
- 提供具體的交易建議

### power_of_3

- 完整的 PO3 週期分析
- 追蹤每個階段的完成狀態
- 提供當前所處階段和預期

### trading_recommendations

- 基於時段分析的具體交易計劃
- 結合多個時段的信息
- 提供明確的入場和出場水平

## 實作程式碼片段

```python
def analyze_killzone_sessions(ohlc_data):
    # 獲取所有時段數據
    sessions = {
        'asia': smc.sessions(ohlc_data, "Tokyo", "00:00", "08:00", "UTC"),
        'london': smc.sessions(ohlc_data, "London", "08:00", "16:00", "UTC"),
        'new_york': smc.sessions(ohlc_data, "New York", "13:00", "21:00", "UTC")
    }

    # 分析 Kill Zones
    killzones = analyze_killzones(ohlc_data)

    # Power of 3 分析
    po3_analysis = analyze_power_of_3(sessions, ohlc_data)

    # 檢測操縱行為
    manipulation = detect_session_manipulation(sessions, ohlc_data)

    return formatted_response


def analyze_power_of_3(sessions, ohlc_data):
    """
    分析 Power of 3 模式
    - 亞洲累積
    - 倫敦操縱
    - 紐約分配
    """
    asia_range = sessions['asia']['High'].max() - sessions['asia']['Low'].min()

    # 檢查倫敦是否突破亞洲區間
    london_break = check_range_break(
        sessions['london'],
        sessions['asia']
    )

    # 確認紐約的真實方向
    ny_direction = confirm_true_direction(
        sessions['new_york'],
        london_break
    )

    return {
        'current_phase': determine_current_phase(),
        'pattern_validity': validate_po3_pattern(),
        'confidence': calculate_po3_confidence()
    }


def detect_session_manipulation(sessions, ohlc_data):
    """
    檢測時段操縱行為
    - 假突破
    - 流動性掃除
    - 停損獵取
    """
    manipulations = []

    # 檢查倫敦對亞洲高低點的掃除
    if sessions['london']['Low'].min() < sessions['asia']['Low'].min():
        # 可能的看跌陷阱
        reversal = check_reversal_after_sweep(ohlc_data)
        if reversal:
            manipulations.append({
                'type': 'BEAR_TRAP',
                'session': 'LONDON',
                'swept_level': sessions['asia']['Low'].min()
            })

    return manipulations
```

## 總結

這個 API 設計充分利用了 smart-money-concepts 的時段功能，並加入了 ICT 特有的 Kill Zone 和 Power of 3 概念分析。數據結構為
GPT 提供了時段動態的完整圖景，有助於識別日內交易機會。