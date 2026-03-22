# 서버 세팅 가이드 (RTX 4070 / CUDA 12.2)

## 1. 소스 코드 올리기

```bash
# 서버에서
git clone <repo-url>
cd bitcoin
```

또는 scp로 직접 전송:
```bash
scp -r ./bitcoin_rl_system user@server:/path/to/project/
```

---

## 2. 가상환경 생성 및 패키지 설치

```bash
python3.11 -m venv venv
source venv/bin/activate

# PyTorch (CUDA 12.2 → cu121 호환) — 반드시 requirements.txt보다 먼저
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 나머지 패키지
pip install -r requirements.txt
```

CUDA 버전 확인:
```bash
nvidia-smi   # 우측 상단 CUDA Version 확인
# 12.1 / 12.2이면: cu121
# 12.4이면:        cu124
```

---

## 3. 디렉토리 구조 확인

소스 코드를 올리면 아래 구조여야 합니다:

```
project/
├── bitcoin_rl_system/
│   ├── __init__.py
│   ├── fetch_upbit_history.py
│   ├── build_training_frame.py
│   ├── data_handler.py
│   ├── trading_environment.py
│   ├── rl_agent.py
│   └── main.py
├── data analysis/           ← 없어도 됨, 자동 생성
│   └── data/
│       ├── raw/
│       └── processed/
└── requirements.txt
```

---

## 4. 데이터 수집

```bash
# 일봉 + 1분봉 수집 (약 30~40분 소요, API rate limit 때문)
python -m bitcoin_rl_system.fetch_upbit_history
```

완료 시 아래 파일 생성됨:
- `data analysis/data/raw/candles_1m/KRW-BTC_minutes_1m.parquet` (525,070행)
- `data analysis/data/raw/candles_days/KRW-BTC_days.parquet`

---

## 5. Training Frame 빌드

```bash
python -m bitcoin_rl_system.build_training_frame
```

완료 시:
- `data analysis/data/processed/rl/rl_market_frame_1m.parquet` (44컬럼)

---

## 6. 파이프라인 검증 (스모크 테스트)

```bash
python -m bitcoin_rl_system.main
```

아래처럼 나오면 정상:
```
Bitcoin RL system
rows: 525070, columns: 44
Observation dim: 9742
Using cuda device    ← GPU 인식 확인
PPO model build complete (with VecNormalize).
Smoke test complete.
```

`Using cpu device`가 뜨면 PyTorch CUDA 설치 문제 → 2단계 재확인

---

## 7. GPU 인식 직접 확인

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# True
# NVIDIA GeForce RTX 4070
```
