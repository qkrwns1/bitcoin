# Bitcoin RL System

This folder is a project-specific scaffold derived from the high-level structure
of `quant-rl-trading-agent`, but redesigned for the Bitcoin / Upbit use case.

## Design Goals

- RL core: PPO
- Action: `target_position_ratio` in `[0, 1]`
- Execution rule: observe up to `t`, execute at `t+1 open`
- Reward: `net_equity_(t+1) - net_equity_t - holding_penalty_t`
- Step frequency: 1 minute
- Observation structure:
  - sequence block
  - context block
  - portfolio block
- Transformer is used only as a sequence encoder, not as the entire RL model

## Folder Mapping

- `main.py`
  - pipeline entry point
  - train / evaluate orchestration
- `data_handler.py`
  - load processed minute-level data from local storage
  - generate RL-ready feature frames
- `trading_environment.py`
  - custom Gymnasium environment
  - target position ratio execution and reward logic
- `rl_agent.py`
  - PPO agent wrapper
  - feature extractor definition
  - Transformer attachment point
- `ARCHITECTURE.md`
  - detailed explanation of model wiring and training usage

## Planned Observation Layout

1. Sequence block
- recent 1-minute candles / structure features
- returns
- wick / body structure
- volatility-like features
- volume features

2. Context block
- 52-week high / low related features
- 24h accumulated trade value / volume
- daily / rolling high-low structure
- market state fields

3. Portfolio block
- current cash
- BTC holdings
- average entry price
- current position ratio
- total equity
- unrealized pnl

## Planned Model Layout

```text
sequence block -> Transformer encoder
context block -> MLP
portfolio block -> MLP
concat/fuse -> shared latent
shared latent -> PPO actor head
shared latent -> PPO critic head
```

## Status

This folder is currently a scaffold for design alignment, not a finished
training implementation.
