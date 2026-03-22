# Feature Set V1

## Sequence Features

- open
- high
- low
- close
- volume
- trade_value
- return_1m
- log_return_1m
- return_5m
- return_15m
- body_ratio
- upper_wick_ratio
- lower_wick_ratio
- range_ratio
- volume_zscore_30m
- volume_zscore_60m
- volume_zscore_120m
- volatility_15m
- volatility_60m
- dist_to_15m_high
- dist_to_15m_low
- dist_to_30m_high
- dist_to_30m_low
- dist_to_60m_high
- dist_to_60m_low
- dist_to_240m_high
- dist_to_240m_low
- dist_to_4h_high
- dist_to_4h_low

## Context Features

- dist_to_daily_high
- dist_to_daily_low
- dist_to_52w_high
- dist_to_52w_low
- days_since_52w_high
- days_since_52w_low
- position_in_52w_range
- acc_trade_volume_24h
- acc_trade_price_24h
- change_rate
- signed_change_rate
- market_state
- market_warning
- is_trading_suspended
- ask_bid

## Portfolio Features

- cash
- btc_holding
- avg_entry_price
- position_ratio
- total_equity
- unrealized_pnl
- realized_pnl

## Notes

- This V1 set is based on all items classified as required + recommended.
- Trade-derived microstructure features are intentionally excluded from V1.
- Orderbook-derived features are intentionally excluded from V1.
- Portfolio features are environment-managed, not part of the raw processed market table.
