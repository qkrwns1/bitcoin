"""Step 2 of Phase 2: FAISS index build.

추출된 embedding으로 빠른 유사 패턴 검색 인덱스를 만든다.

실행: python -m bitcoin_rl_system.build_faiss
출력:
  bitcoin_rl_system/faiss/index.faiss   - IVFFlat 검색 인덱스
  bitcoin_rl_system/faiss/index_info.pkl - 인덱스 메타 (dim, nlist 등)

다음 단계: python -m bitcoin_rl_system.verify_faiss  (검색 테스트)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

_HERE    = Path(__file__).parent
FAISS_DIR = _HERE / "faiss"
EMB_PATH  = FAISS_DIR / "embeddings.npy"
META_PATH = FAISS_DIR / "metadata.parquet"
IDX_PATH  = FAISS_DIR / "index.faiss"
INFO_PATH = FAISS_DIR / "index_info.pkl"

# IVFFlat: nlist 클수록 정확하지만 학습 느림. 경험칙: sqrt(N)
NLIST = 1024
NPROBE = 64   # 검색 시 탐색할 cluster 수 (정확도↑ vs 속도↓)


def main() -> None:
    # ── 로드 ────────────────────────────────────────────────────
    print(f"embedding 로드: {EMB_PATH}")
    embeddings = np.load(str(EMB_PATH)).astype(np.float32)
    meta       = pd.read_parquet(META_PATH)
    N, D = embeddings.shape
    print(f"  shape: {N:,} × {D}  ({embeddings.nbytes/1e6:.0f} MB)")

    # ── L2 정규화 (cosine similarity용) ──────────────────────────
    print("L2 정규화...")
    faiss.normalize_L2(embeddings)

    # ── 인덱스 학습 & 추가 ───────────────────────────────────────
    print(f"IndexIVFFlat 학습 (nlist={NLIST})...")
    quantizer = faiss.IndexFlatIP(D)        # inner product (cosine after normalization)
    index     = faiss.IndexIVFFlat(quantizer, D, NLIST, faiss.METRIC_INNER_PRODUCT)

    index.train(embeddings)
    print(f"  학습 완료. 벡터 추가 중...")
    index.add(embeddings)
    index.nprobe = NPROBE
    print(f"  총 {index.ntotal:,}개 벡터 추가됨")

    # ── GPU 사용 가능하면 GPU 인덱스로 변환 후 다시 CPU로 저장 ──
    # (저장은 항상 CPU)
    try:
        res   = faiss.StandardGpuResources()
        g_idx = faiss.index_cpu_to_gpu(res, 0, index)
        print(f"  GPU 인덱스 테스트: OK")
        # 저장은 CPU 버전으로
    except Exception:
        print("  GPU 없음, CPU 인덱스 사용")

    # ── 저장 ────────────────────────────────────────────────────
    print(f"인덱스 저장: {IDX_PATH}")
    faiss.write_index(index, str(IDX_PATH))

    info = {"dim": D, "nlist": NLIST, "nprobe": NPROBE, "n_vectors": N}
    with open(INFO_PATH, "wb") as f:
        pickle.dump(info, f)

    print(f"\n완료!")
    print(f"  index.faiss : {IDX_PATH.stat().st_size/1e6:.1f} MB")
    print(f"  벡터 수      : {N:,}")
    print(f"  차원         : {D}")

    # ── 빠른 검색 테스트 ─────────────────────────────────────────
    print("\n검색 테스트 (최근 10개 bar)...")
    index.nprobe = NPROBE
    q = embeddings[-10:]
    D_scores, I = index.search(q, 6)   # top-6 (첫번째는 자기 자신)

    future_cols = ["return_1h", "return_5h", "return_1d"]
    for qi in range(min(3, len(q))):
        neighbors  = I[qi][1:6]          # 자기 자신 제외
        scores     = D_scores[qi][1:6]
        avg_1h     = meta["return_1h"].iloc[neighbors].mean()
        confidence = float(scores.mean())
        ts         = meta["ts"].iloc[-10 + qi]
        print(f"  [{ts}]  top-5 avg_return_1h={avg_1h*100:.2f}%  confidence={confidence:.3f}")

    print(f"\n다음 단계: Phase 2 환경 확장 및 fine-tuning")


if __name__ == "__main__":
    main()
