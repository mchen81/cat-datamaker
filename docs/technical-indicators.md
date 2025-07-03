# 技術指標 API 文檔

## API 端點

```
GET /api/technical-indicators/{symbol}/{timeframe}
```

## 參數說明

- **symbol**: 交易對 (例如: BTCUSDT)
- **timeframe**: 時間框架 (例如: 1h, 4h, 1d)

## Query Parameters

- as_of_datetime: 可選的 ISO 8601 datetime 字串 (例如: 2024-01-01T00:00:00Z)

## 實作方法

### 1. 指標選擇原則

- 只保留與 SMC/ICT 策略互補的指標
- 將數值轉換為狀態描述，減少 GPT 處理數字的負擔
- 著重於背離、極值和轉折信號
- 避免重複或相似的指標

### 2. 核心指標列表

- **RSI (14)**: 動量和背離檢測
- **成交量分析**: 驗證價格行為的有效性
- **ATR (14)**: 波動率和止損計算
- **市場強度指數**: 綜合多個指標的評分
- **動量指標**: 價格變化速度

### 3. 數據簡化策略

- 不提供具體數值，只提供狀態
- 使用描述性標籤而非數字
- 整合相關指標為綜合評分

## 回傳 JSON 格式

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "timestamp": "2025-01-03T14:00:00Z",
  "current_price": 95000,
  "momentum_indicators": {
    "rsi": {
      "status": "NEUTRAL",
      "value_range": "40-50",
      "trend": "RISING",
      "divergence": {
        "detected": true,
        "type": "BULLISH_DIVERGENCE",
        "strength": "MODERATE",
        "description": "Price made lower low, RSI made higher low"
      },
      "key_levels": {
        "oversold": false,
        "overbought": false,
        "previous_extreme": "OVERSOLD_3_BARS_AGO"
      }
    },
    "momentum_oscillator": {
      "status": "BULLISH",
      "strength": "MODERATE",
      "acceleration": "INCREASING",
      "cross_signal": "NONE"
    }
  },
  "volume_analysis": {
    "current_volume_profile": "ABOVE_AVERAGE",
    "volume_trend": "INCREASING",
    "volume_confirmation": true,
    "unusual_activity": {
      "detected": true,
      "type": "VOLUME_SPIKE",
      "magnitude": "2.5X_AVERAGE",
      "interpretation": "Strong buying interest"
    },
    "volume_patterns": {
      "accumulation": true,
      "distribution": false,
      "pattern": "RISING_PRICE_RISING_VOLUME"
    },
    "smart_money_flow": {
      "direction": "INFLOW",
      "strength": "STRONG",
      "persistence": "3_CONSECUTIVE_BARS"
    }
  },
  "volatility_metrics": {
    "atr_status": "NORMAL",
    "volatility_trend": "DECREASING",
    "current_range": "MEDIUM",
    "range_expansion": false,
    "suggested_stop_distance": "1.2%",
    "volatility_regime": "CONSOLIDATION"
  },
  "market_strength": {
    "overall_score": 7.2,
    "trend_strength": "MODERATE_BULLISH",
    "buy_pressure": "HIGH",
    "sell_pressure": "LOW",
    "strength_components": {
      "price_action": 8,
      "volume": 7,
      "momentum": 6.5,
      "volatility": 7
    },
    "market_phase": "EARLY_TREND"
  },
  "confluences": {
    "bullish_signals": [
      "RSI bullish divergence",
      "Volume spike on up move",
      "Buy pressure increasing"
    ],
    "bearish_signals": [
      "Approaching resistance zone"
    ],
    "neutral_factors": [
      "Volatility contracting",
      "RSI in neutral zone"
    ],
    "signal_alignment": "BULLISH_BIAS",
    "confidence_score": 7.5
  },
  "indicator_summary": {
    "primary_signal": "BULLISH",
    "signal_strength": "MODERATE",
    "key_observation": "Bullish divergence with volume confirmation suggests potential upward move",
    "caution_notes": "Wait for volatility expansion to confirm breakout",
    "recommended_action": "LOOK_FOR_LONG_ENTRIES",
    "invalidation_scenario": "Loss of volume support or RSI breakdown below 40"
  },
  "trading_conditions": {
    "market_state": "TRENDING",
    "trend_maturity": "EARLY",
    "optimal_strategy": "TREND_FOLLOWING",
    "risk_environment": "NORMAL",
    "session_alignment": true
  }
}
```

## 欄位說明

### momentum_indicators

- **rsi.status**: OVERSOLD | NEUTRAL | OVERBOUGHT
- **divergence**: 價格與指標的背離檢測
- 不顯示具體 RSI 數值，只顯示區間和狀態

### volume_analysis

- **volume_profile**: 相對於平均成交量的描述
- **unusual_activity**: 異常成交量檢測
- **smart_money_flow**: 機構資金流向推測

### volatility_metrics

- **atr_status**: LOW | NORMAL | HIGH | EXTREME
- **suggested_stop_distance**: 基於 ATR 的止損建議（百分比）
- **volatility_regime**: 市場狀態判定

### market_strength

- 綜合評分系統 (0-10)
- 分解為多個組成部分
- 提供市場階段判定

### confluences

- 整合所有指標信號
- 分類為看漲、看跌、中性
- 提供整體信心評分

## 實作程式碼片段

```python
def analyze_technical_indicators(ohlc_data):
    """
    分析技術指標並返回簡化的狀態描述
    """
    # 計算基礎指標
    rsi = calculate_rsi(ohlc_data, period=14)
    volume_analysis = analyze_volume_patterns(ohlc_data)
    atr = calculate_atr(ohlc_data, period=14)

    # 檢測背離
    divergences = detect_divergences(ohlc_data, rsi)

    # 成交量分析
    volume_profile = analyze_volume_profile(ohlc_data)

    # 綜合市場強度
    market_strength = calculate_market_strength(
        ohlc_data, rsi, volume_analysis, atr
    )

    return format_indicators_response(
        rsi, volume_analysis, atr, divergences, market_strength
    )


def detect_divergences(price_data, indicator_data):
    """
    檢測價格與指標之間的背離
    """
    # 識別價格和指標的擺動點
    price_swings = find_swing_points(price_data)
    indicator_swings = find_swing_points(indicator_data)

    # 比較趨勢方向
    divergences = []
    # ... 背離檢測邏輯

    return divergences


def analyze_volume_profile(ohlc_data):
    """
    分析成交量模式和異常活動
    """
    volume = ohlc_data['volume']
    price = ohlc_data['close']

    # 計算平均成交量
    avg_volume = volume.rolling(20).mean()

    # 檢測成交量異常
    volume_spikes = volume > avg_volume * 2

    # 分析價量關係
    price_volume_correlation = analyze_price_volume_relationship(price, volume)

    return {
        'profile': determine_volume_profile(volume, avg_volume),
        'spikes': volume_spikes.sum(),
        'trend': determine_volume_trend(volume),
        'price_confirmation': price_volume_correlation
    }


def calculate_market_strength(ohlc_data, rsi, volume_analysis, atr):
    """
    計算綜合市場強度評分
    """
    # 價格行為評分
    price_score = calculate_price_action_score(ohlc_data)

    # 動量評分
    momentum_score = calculate_momentum_score(rsi)

    # 成交量評分
    volume_score = calculate_volume_score(volume_analysis)

    # 波動率評分
    volatility_score = calculate_volatility_score(atr)

    # 加權綜合評分
    overall_score = (
            price_score * 0.3 +
            momentum_score * 0.25 +
            volume_score * 0.25 +
            volatility_score * 0.2
    )

    return {
        'overall_score': overall_score,
        'components': {
            'price_action': price_score,
            'momentum': momentum_score,
            'volume': volume_score,
            'volatility': volatility_score
        }
    }
```

## 設計原則

### 簡化輸出

- 避免數字過載，專注於狀態和趨勢
- 使用描述性標籤提高可讀性

### 背離檢測

- 重點關注價格與指標的背離信號
- 提供背離的強度和可信度評估

### 成交量確認

- 強調成交量對價格行為的確認作用
- 識別異常成交量活動

### 綜合評分

- 將多個指標整合為單一評分
- 提供清晰的市場強度判斷

## 總結

這個技術指標API設計專注於提供與SMC/ICT分析互補的指標信息，通過簡化輸出和狀態描述，使GPT能夠更有效地處理和解釋技術指標信號，輔助市場結構分析做出更準確的交易決策。