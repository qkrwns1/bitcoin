"""PPO agent scaffold with explicit Transformer attachment point."""

from dataclasses import dataclass


@dataclass(slots=True)
class AgentConfig:
    algorithm: str = "PPO"
    transformer_enabled: bool = True
    transformer_layers: int = 2
    transformer_hidden_dim: int = 128
    context_hidden_dim: int = 64
    portfolio_hidden_dim: int = 64


class BitcoinFeatureExtractor:
    """Planned feature extractor wiring.

    sequence block -> Transformer encoder
    context block -> MLP
    portfolio block -> MLP
    fuse -> shared latent for actor / critic
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config


class BitcoinRLAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def build_model(self) -> None:
        raise NotImplementedError("PPO model construction is not implemented yet.")

    def train(self) -> None:
        raise NotImplementedError("Training loop is not implemented yet.")

    def evaluate(self) -> None:
        raise NotImplementedError("Evaluation logic is not implemented yet.")
