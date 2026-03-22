"""Custom Gymnasium environment scaffold for Bitcoin RL."""

from dataclasses import dataclass


@dataclass(slots=True)
class EnvironmentConfig:
    initial_cash: float = 1_000_000.0
    fee_rate: float = 0.0005
    step_minutes: int = 1
    action_low: float = 0.0
    action_high: float = 1.0


class BitcoinTradingEnvironment:
    """Planned environment rules.

    - observe data up to time t
    - choose target_position_ratio in [0, 1]
    - execute at t+1 open
    - reward = net_equity_(t+1) - net_equity_t - holding_penalty_t
    """

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config

    def reset(self) -> None:
        raise NotImplementedError("Environment reset is not implemented yet.")

    def step(self, action: float) -> None:
        raise NotImplementedError("Environment step logic is not implemented yet.")
