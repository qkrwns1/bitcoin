"""백테스트: 2020-01-01 ~ 현재, 초기자본 100만원.

Phase 2 모델(phase2_model.zip)이 있으면 우선 사용, 없으면 Phase 1(final_model.zip) 사용.

실행: python -m bitcoin_rl_system.backtest [--split full|train|val|test] [--model phase2|phase1]
출력:
  - 콘솔: 수익률 요약 (총 수익, CAGR, MDD, 샤프)
  - bitcoin_rl_system/backtest_result.csv : 스텝별 포트폴리오 기록
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from bitcoin_rl_system.data_handler import BitcoinDataHandler, DataConfig
from bitcoin_rl_system.trading_environment import BitcoinTradingEnvironment, EnvironmentConfig

_HERE = Path(__file__).parent
INITIAL_CASH = 1_000_000.0   # 100만 원


# ── VecNormalize 수동 적용 ────────────────────────────────────────────

def _load_normalizer(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _normalize(obs: np.ndarray, vn) -> np.ndarray:
    return np.clip(
        (obs - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8),
        -10.0, 10.0,
    ).astype(np.float32)


# ── 성과 지표 ─────────────────────────────────────────────────────────

def _calc_metrics(equity: np.ndarray, timestamps: pd.Series, initial: float) -> dict:
    total_return = (equity[-1] - initial) / initial

    # CAGR
    start_ts = pd.to_datetime(timestamps.iloc[0])
    end_ts   = pd.to_datetime(timestamps.iloc[-1])
    years    = max((end_ts - start_ts).days / 365.25, 1/365)
    cagr     = (equity[-1] / initial) ** (1 / years) - 1

    # MDD
    peak    = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    mdd     = float(drawdown.min())

    # 샤프 (5분봉 기준 연환산: sqrt(252*24*12))
    returns  = np.diff(equity) / equity[:-1]
    ann_factor = np.sqrt(252 * 24 * 12)
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * ann_factor)

    # BTC 매수보유 수익률
    btc_start = float(equity[0])   # 시작 시 BTC로만 매수했다면 (근사치)

    return {
        "initial_krw":    initial,
        "final_krw":      float(equity[-1]),
        "total_return":   total_return,
        "cagr":           cagr,
        "mdd":            mdd,
        "sharpe":         sharpe,
        "total_trades":   None,   # 아래서 채움
        "period_days":    (end_ts - start_ts).days,
    }


# ── 메인 ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  choices=["full", "train", "val", "test"], default="full",
                        help="어떤 데이터셋 구간으로 테스트할지 (기본: full=전체)")
    parser.add_argument("--model",  choices=["phase2", "phase1", "auto"],     default="auto",
                        help="사용할 모델 (기본: auto=phase2 우선)")
    parser.add_argument("--out",    default=str(_HERE / "backtest_result.csv"),
                        help="결과 CSV 저장 경로")
    args = parser.parse_args()

    # ── 모델 선택 ────────────────────────────────────────────────
    phase2_model_path  = _HERE / "checkpoints" / "phase2_model"
    phase2_vecnorm     = _HERE / "checkpoints" / "phase2_vec_normalize.pkl"
    phase1_model_path  = _HERE / "checkpoints" / "final_model"
    phase1_vecnorm     = _HERE / "checkpoints" / "vec_normalize.pkl"

    if args.model == "phase2" or (
        args.model == "auto" and phase2_model_path.with_suffix(".zip").exists()
    ):
        model_path  = phase2_model_path
        vecnorm_path = phase2_vecnorm
        use_belief  = True
        print("[backtest] Phase 2 모델 사용")
    else:
        model_path  = phase1_model_path
        vecnorm_path = phase1_vecnorm
        use_belief  = False
        print("[backtest] Phase 1 모델 사용")

    # ── 데이터 로드 ──────────────────────────────────────────────
    print("데이터 로드...")
    cfg     = DataConfig()
    handler = BitcoinDataHandler(cfg)
    handler.load_processed_frames()
    layout  = handler.build_feature_layout()
    seq_len = cfg.sequence_length

    if args.split == "full":
        market_frame = handler.market_frame
    else:
        market_frame = handler.split_by_time()[args.split]

    print(f"  기간: {market_frame['ts'].iloc[0]}  ~  {market_frame['ts'].iloc[-1]}")
    print(f"  봉 수: {len(market_frame):,}개  ({len(market_frame)*5/60/24:.1f}일)")

    # ── 환경 생성 ────────────────────────────────────────────────
    env_cfg = EnvironmentConfig(
        initial_cash=INITIAL_CASH,
        sequence_length=seq_len,
    )

    if use_belief:
        from bitcoin_rl_system.belief_environment import BeliefAugmentedEnv, make_belief_env
        env = make_belief_env(
            market_frame=market_frame,
            sequence_features=layout["sequence"],
            context_features=layout["context"],
            portfolio_features=layout["portfolio"],
            env_config=env_cfg,
            seq_len=seq_len,
        )
    else:
        env = BitcoinTradingEnvironment(
            market_frame=market_frame,
            sequence_features=layout["sequence"],
            context_features=layout["context"],
            portfolio_features=layout["portfolio"],
            config=env_cfg,
        )

    # ── 모델 & VecNorm 로드 ───────────────────────────────────────
    print(f"모델 로드: {model_path}.zip")
    model = PPO.load(str(model_path))
    vn    = _load_normalizer(vecnorm_path)
    print("  로드 완료")

    # ── 백테스트 루프 ────────────────────────────────────────────
    print("백테스트 시작...")
    obs, _ = env.reset()
    done   = False

    records    = []
    n_trades   = 0
    step_count = 0
    REPORT_INTERVAL = 100_000

    while not done:
        obs_norm = _normalize(obs, vn)
        action, _ = model.predict(obs_norm[np.newaxis], deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action[0]))
        done = terminated or truncated

        records.append({
            "ts":             info["ts"],
            "close":          info["close"],
            "action":         info["action"],
            "target_ratio":   info["target_ratio"],
            "equity":         info["next_equity"],
            "cash":           info["cash"],
            "btc_holding":    info["btc_holding"],
            "avg_entry_price":info["avg_entry_price"],
            "realized_pnl":   info["realized_pnl"],
            "trade":          int(info["trade_executed"]),
        })

        if info["trade_executed"]:
            n_trades += 1

        step_count += 1
        if step_count % REPORT_INTERVAL == 0:
            eq = info["next_equity"]
            ret = (eq - INITIAL_CASH) / INITIAL_CASH * 100
            print(f"  {info['ts']}  equity={eq:,.0f}원  수익률={ret:+.1f}%  거래횟수={n_trades}")

    # ── 결과 집계 ────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"])

    equity_arr = df["equity"].to_numpy()
    metrics    = _calc_metrics(equity_arr, df["ts"], INITIAL_CASH)
    metrics["total_trades"] = n_trades

    # BTC 매수보유 수익률 계산
    btc_start  = float(df["close"].iloc[0])
    btc_end    = float(df["close"].iloc[-1])
    bh_return  = (btc_end - btc_start) / btc_start

    # ── 출력 ─────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  백테스트 결과 ({args.split} / {'Phase 2' if use_belief else 'Phase 1'})")
    print("="*55)
    print(f"  기간           : {metrics['period_days']}일")
    print(f"  초기자본       : {metrics['initial_krw']:>15,.0f} 원")
    print(f"  최종자산       : {metrics['final_krw']:>15,.0f} 원")
    print(f"  총 수익률      : {metrics['total_return']*100:>+14.2f} %")
    print(f"  CAGR           : {metrics['cagr']*100:>+14.2f} %")
    print(f"  MDD            : {metrics['mdd']*100:>14.2f} %")
    print(f"  샤프 비율      : {metrics['sharpe']:>14.3f}")
    print(f"  총 거래 횟수   : {n_trades:>14,} 회")
    print("-"*55)
    print(f"  BTC 매수보유   : {bh_return*100:>+14.2f} %  ({btc_start:,.0f} → {btc_end:,.0f})")
    print("="*55)

    # ── CSV 저장 ──────────────────────────────────────────────────
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
