# 交易信號 API 文檔 (Trade Signal API)

## API 端點

```
GET /api/trade-signal/{symbol}/{timeframe}
```

## 參數說明

- symbol: 交易對 (例如: BTCUSDT)
- timeframe: 主要交易時間框架 (建議: 1h, 4h)

## Query Parameters

- as_of_datetime: 可選的 ISO 8601 datetime 字串 (例如: 2024-01-01T00:00:00Z)

## 設計理念

這個 API 作為前置過濾器，基於 SMC/ICT 核心原則快速評估市場狀況，只在高概率機會出現時才觸發完整的 LLM 分析，從而大幅降低
Token 消耗。

---

### 核心決策邏輯

#### 1. SMC/ICT 高概率設置識別

- **A+ 設置**：多個條件匯合的高勝率機會
- **標準設置**：符合基本 SMC 結構的機會
- **無效設置**：市場混亂或低概率情況

#### 2. 快速評估標準

- MTF 結構對齊度
- 關鍵區域接近度
- 時段和流動性狀態
- 動量確認

#### 3. 風險管理自動化

- 基於市場結構的止損位
- 多目標止盈策略
- 風險回報比評估

---

## 回傳 JSON 格式

### 情況 1：高概率買入信號

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "timestamp": "2025-01-03T14:00:00Z",
  "current_price": 95000,
  "signal": "BUY",
  "signal_strength": "STRONG",
  "confidence": 8.5,
  "entry_zone": {
    "optimal": 94800,
    "range": [
      94700,
      94900
    ],
    "current_distance": "+0.21%"
  },
  "take_profit_prices": [
    {
      "level": 95500,
      "rationale": "4H resistance & 0.5 FVG fill",
      "probability": "HIGH",
      "rr_ratio": 1.67
    },
    {
      "level": 96200,
      "rationale": "Daily resistance & liquidity zone",
      "probability": "MEDIUM",
      "rr_ratio": 3.0
    },
    {
      "level": 97000,
      "rationale": "Weekly resistance & major liquidity",
      "probability": "LOW",
      "rr_ratio": 4.75
    }
  ],
  "stop_loss_prices": [
    {
      "level": 94500,
      "type": "TIGHT",
      "rationale": "Below 4H swing low"
    },
    {
      "level": 94200,
      "type": "STANDARD",
      "rationale": "Below daily OB"
    },
    {
      "level": 93800,
      "type": "WIDE",
      "rationale": "Major structure break"
    }
  ],
  "reason": "Price swept London low into daily bullish OB with 4H BOS confirmation. HTF aligned bullish, approaching NY session distribution phase.",
  "setup_quality": {
    "smc_score": 8.5,
    "risk_reward": "EXCELLENT",
    "timing": "OPTIMAL",
    "invalidation_clear": true
  },
  "trigger_llm": true,
  "llm_context": "HIGH_PROBABILITY_LONG"
}
```

### 情況 2：等待信號

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "timestamp": "2025-01-03T14:00:00Z",
  "current_price": 95000,
  "signal": "WAIT",
  "signal_strength": "NONE",
  "confidence": 0,
  "market_state": "CONSOLIDATION",
  "wait_reason": "Mid-range position, no clear setup",
  "next_opportunities": [
    {
      "scenario": "BULLISH",
      "trigger_level": 95500,
      "condition": "Break and retest of 4H resistance"
    },
    {
      "scenario": "BEARISH",
      "trigger_level": 94500,
      "condition": "Break below 4H support with volume"
    }
  ],
  "monitoring_levels": {
    "resistance": 95500,
    "support": 94500,
    "current_range": "NEUTRAL_ZONE"
  },
  "reason": "Price in no-trade zone between key levels. No SMC setup present. Wait for structure break.",
  "trigger_llm": false,
  "check_again_in": "1 hour"
}
```

### 情況 3：賣出信號

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "timestamp": "2025-01-03T14:00:00Z",
  "current_price": 95000,
  "signal": "SELL",
  "signal_strength": "MODERATE",
  "confidence": 7.2,
  "entry_zone": {
    "optimal": 95200,
    "range": [
      95100,
      95300
    ],
    "current_distance": "-0.21%"
  },
  "take_profit_prices": [
    {
      "level": 94500,
      "rationale": "4H support & 0.5 retracement",
      "probability": "HIGH",
      "rr_ratio": 1.4
    },
    {
      "level": 94000,
      "rationale": "Daily bullish OB",
      "probability": "MEDIUM",
      "rr_ratio": 2.4
    },
    {
      "level": 93500,
      "rationale": "Weekly support & major OB",
      "probability": "LOW",
      "rr_ratio": 3.4
    }
  ],
  "stop_loss_prices": [
    {
      "level": 95500,
      "type": "TIGHT",
      "rationale": "Above 4H swing high"
    },
    {
      "level": 95800,
      "type": "STANDARD",
      "rationale": "Above daily resistance"
    }
  ],
  "reason": "Failed break of daily resistance with bearish CHoCH on 4H. Liquidity swept above, expecting reversal.",
  "setup_quality": {
    "smc_score": 7.2,
    "risk_reward": "GOOD",
    "timing": "GOOD",
    "invalidation_clear": true
  },
  "trigger_llm": true,
  "llm_context": "MODERATE_PROBABILITY_SHORT"
}
```

---

## 欄位說明

### signal

- **BUY**：符合 SMC 買入條件
- **SELL**：符合 SMC 賣出條件
- **WAIT**：無明確機會，節省 Token

### signal_strength

- **STRONG**：A+ 設置，多重匯合
- **MODERATE**：標準設置
- **WEAK**：邊緣設置（通常不觸發 LLM）

### take_profit_prices

- 基於 SMC 結構的多個目標
- 包含達成概率評估
- 提供風險回報比

### stop_loss_prices

- 多種止損選項（緊密/標準/寬鬆）
- 基於市場結構而非固定點數

### trigger_llm

- **true**：觸發完整 LLM 分析
- **false**：不消耗 Token，等待更好機會

---

## SMC/ICT 決策規則

### 買入信號觸發條件（需滿足至少 3 個）：

- HTF（日線以上）結構看漲
- 價格接近優質需求區（OB/FVG）
- 近期出現流動性掃除
- 處於紐約時段或倫敦時段
- 4H 或 1H 出現 BOS
- 成交量確認
- RSI 背離支持

### 賣出信號觸發條件（需滿足至少 3 個）：

- HTF 結構看跌或到達主要阻力
- 價格接近優質供應區
- 賣側流動性被掃除
- 時段高點形成
- 出現 CHoCH 或 BOS 失敗
- 成交量衰竭
- 超買狀態

### 等待信號情況：

- 價格在區間中部
- 結構不明確
- 時間框架衝突嚴重
- 無明顯 SMC 設置
- 盤整階段
- 重大新聞事件前

---

## 實作概念

```python
# 決策邏輯示例（偽代碼）
def evaluate_trade_signal(market_data):
    """
    快速評估是否存在交易機會
    """
    # 1. 檢查 MTF 對齊
    mtf_score = check_mtf_alignment()

    # 2. 評估價格位置
    price_location = evaluate_price_location()

    # 3. SMC 設置檢查
    smc_setup = check_smc_patterns()

    # 4. 時段分析
    session_quality = analyze_session_timing()

    # 5. 綜合評分
    total_score = calculate_setup_score()

    if total_score >= 7:
        return generate_trade_signal()
    else:
        return generate_wait_signal()


def calculate_smc_targets(signal_type, current_price):
    """
    基於 SMC 結構計算目標價位
    """
    # 識別關鍵結構級別
    # 計算 FVG、OB、流動性目標
    # 返回多個目標和止損位
```

---

## 優勢

### Token 效率

- 只在高概率機會時觸發 LLM
- 大部分時間返回簡單的 WAIT 信號

### 風險管理

- 自動提供止損止盈建議
- 基於結構而非任意水平

### 決策品質

- 遵循嚴格的 SMC/ICT 原則
- 避免過度交易

### 可擴展性

- 可根據歷史表現調整觸發閾值

易於添加新的過濾條件