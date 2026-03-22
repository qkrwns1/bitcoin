"""Pipeline entry point for the Bitcoin RL system.

This file mirrors the role of `main.py` in the reference repository, but is
redefined for:
- Upbit / Bitcoin data
- 1-minute step trading
- PPO with target_position_ratio action
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("Bitcoin RL system scaffold")
    print(f"Root: {PROJECT_ROOT}")
    print("Training pipeline implementation is pending.")


if __name__ == "__main__":
    main()
