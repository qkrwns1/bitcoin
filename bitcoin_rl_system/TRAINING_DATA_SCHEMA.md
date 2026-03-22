# Training Data Schema

## Goal

Define the first-pass RL market data structure for:

- 1-minute decision steps
- PPO as RL core
- Transformer on the sequence block only
- context and portfolio features handled separately

## Core Rule

One market row represents one timestamp.

The environment will construct:

- sequence block: recent `N` rows
- context block: current row context fields
- portfolio block: internal environment state

## Recommended First Table

Use one unified processed feature table first.

Example table name:

- `rl_market_frame_1m`

## Primary Key

- `ts`
- `market`

## Required Base Columns

- `ts`
- `market`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `trade_value`

These are the minimum columns required for the environment and sequence building.

## Sequence Block Columns

These columns are intended to be consumed as a rolling sequence for the
Transformer branch.

- `open`
- `high`
- `low`
- `close`
- `volume`
- `trade_value`
- `return_1m`
- `log_return_1m`
- `return_5m`
- `return_15m`
- `body_ratio`
- `upper_wick_ratio`
- `lower_wick_ratio`
- `range_ratio`
- `volume_zscore_30m`
- `volume_zscore_120m`
- `volatility_15m`
- `volatility_60m`
- `dist_to_30m_high`
- `dist_to_30m_low`
- `dist_to_60m_high`
- `dist_to_60m_low`
- `dist_to_240m_high`
- `dist_to_240m_low`

## Context Block Columns

These columns are intended to be consumed only from the current row and passed
through a non-sequence branch such as an MLP.

- `dist_to_daily_high`
- `dist_to_daily_low`
- `dist_to_52w_high`
- `dist_to_52w_low`
- `days_since_52w_high`
- `days_since_52w_low`
- `acc_trade_volume_24h`
- `acc_trade_price_24h`
- `change_rate`
- `signed_change_rate`
- `market_state`
- `market_warning`
- `is_trading_suspended`
- `ask_bid`

## Portfolio Block Columns

These are NOT stored in the market feature table as raw market inputs.
They are maintained by the environment and appended to the final observation.

- `cash`
- `btc_holding`
- `avg_entry_price`
- `position_ratio`
- `total_equity`
- `unrealized_pnl`
- `realized_pnl`

## Derived Feature Notes

### Candle structure

- `body_ratio = abs(close - open) / open`
- `upper_wick_ratio = (high - max(open, close)) / open`
- `lower_wick_ratio = (min(open, close) - low) / open`
- `range_ratio = (high - low) / open`

### Distance features

Distance features should be normalized relative distances, not raw price gaps.

Examples:

- `dist_to_30m_high = close / rolling_high_30m - 1`
- `dist_to_30m_low = close / rolling_low_30m - 1`
- `dist_to_52w_high = close / highest_52_week_price - 1`
- `dist_to_52w_low = close / lowest_52_week_price - 1`

### Leakage rule

Every derived feature must use only information available up to time `t`.

No future values may be used in:

- rolling highs
- rolling lows
- z-scores
- volatility
- breakout features

## Observation Construction

At step `t`, the environment builds:

1. sequence block
- rows `[t-N+1, ..., t]`
- columns from the sequence block list

2. context block
- row `t`
- columns from the context block list

3. portfolio block
- environment internal values at time `t`

## Example Output Shapes

If:

- sequence length = `360`
- sequence feature count = `24`
- context feature count = `14`
- portfolio feature count = `7`

Then:

- sequence tensor shape = `[360, 24]`
- context vector shape = `[14]`
- portfolio vector shape = `[7]`

## Recommended Build Order

1. Build 1-minute OHLCV table
2. Add sequence block derived features
3. Add context block derived features from Upbit fields
4. Save as processed training frame
5. Let the environment create rolling observations from this table

## First Practical Scope

Do NOT try to include every possible raw field in the first model.

Use:

- stable candle structure features
- rolling high/low distance features
- 52-week and 24h context fields from Upbit
- portfolio state inside the environment

This is enough for the first PPO + Transformer-assisted system.
