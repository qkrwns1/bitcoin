# Architecture Notes

## Reference Source

The structure in this folder follows the modular split of:

- environment
- agent
- data handler
- pipeline entry point

from `quant-rl-trading-agent`, but the internals are redefined for Bitcoin and
Upbit-based 1-minute trading.

## What We Keep From The Reference

- PPO as the RL core
- environment / agent / data separation
- market vs portfolio feature separation
- training pipeline concept

## What We Replace

- stock-oriented data loading
- discrete long/short action design
- flattened observation-only design
- stock-specific feature assumptions

## Action Design

- `action = target_position_ratio`
- range: `[0, 1]`
- meaning:
  - `0.0`: all cash
  - `1.0`: fully allocated to BTC

The environment should automatically rebalance the current portfolio toward the
target ratio.

## Execution Timing

- state uses information up to time `t`
- action is decided at time `t`
- trade is executed at `t+1 open`

This is the current anti-leakage assumption.

## Reward Design

```text
reward_t = net_equity_(t+1) - net_equity_t - holding_penalty_t
```

Where:

```text
holding_penalty_t = total_equity_t * (0.0005 / 1440)
```

because the current design uses a 1-minute step.

## Transformer Attachment Point

The Transformer should only process the sequence block.

```text
recent minute-level sequence -> transformer encoder -> sequence embedding
context features -> context MLP -> context embedding
portfolio features -> portfolio MLP -> portfolio embedding
all embeddings -> fused latent -> PPO policy/value heads
```

This means the Transformer is an auxiliary representation module inside the PPO
policy network, not a replacement for the entire RL system.

## Training Code Usage Plan

1. `data_handler.py`
- load minute-level processed market frames
- split train / validation / test by time
- expose sequence/context feature columns

2. `trading_environment.py`
- build observation for each step
- maintain portfolio state
- apply fees and reward logic

3. `rl_agent.py`
- define custom feature extractor for PPO
- attach Transformer to sequence branch
- attach MLP to context and portfolio branches

4. `main.py`
- instantiate data handler
- create environments
- train PPO
- evaluate on later unseen periods
