# 전체 파이프라인 실행 가이드 (데스크탑용)

학습용 머신에서 raw AIS CSV → 학습 완료된 ONNX 모델까지의 전 과정.
노트북(7530U)에서는 `SAMPLE_MMSI`를 1000~2000으로 줄여 빠른 테스트에만 사용.
데스크탑에서는 풀스케일로 돌리는 것을 권장.

---

## 0. 사전 준비

### Python 환경
- Python 3.10 권장 (3.13은 torch 2.x DLL 이슈 발생 가능)
- 의존성:
  ```bash
  pip install torch numpy tqdm onnx onnxruntime matplotlib scikit-learn
  ```
  - GPU 머신이면 CUDA 대응 torch 설치: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
  - `scikit-learn`은 `iforest`/`ocsvm` 모델에만 필요

### 디렉터리 구조
```
ml/
├── data/                       # 원본 AIS CSV (gitignored)
│   └── ais-YYYY-MM-DD.csv
├── output/                     # 학습 산출물 (gitignored)
│   ├── scaler_<model>.json
│   └── <model>/
│       ├── model_<model>.onnx
│       └── threshold_<model>.txt
├── grid_stats.json             # 격자 통계 (gitignored)
├── ais-YYYY-MM-DD_preprocessed.csv   # 전처리 결과 (gitignored)
└── *.py                        # 스크립트
```

### 풀스케일 학습 시 노트북 vs 데스크탑 설정 차이

| 항목 | 노트북(7530U, CPU) | 데스크탑(GPU) |
|---|---|---|
| `SAMPLE_MMSI` (train_benchmark.py L127) | 1000~2000 | 10000 (기본) |
| `epochs` | 10~15 | 30~50 (기본) |
| `batch_size` | 256~512 | 256 (기본) |
| 학습 모델 | dcdetect만 | all (9개) |

---

## 1. 격자 통계 생성 (`build_grid_stats.py`)

raw AIS CSV에서 0.05° 격자별 평균 SOG, 표준편차, 표본 수를 계산해
`grid_stats.json` 생성. preprocess와 eval에서 컨텍스트 피처 계산에 사용.

```bash
cd ml/
python build_grid_stats.py data/ais-2025-05-15.csv

# 여러 파일 합치기
python build_grid_stats.py data/*.csv
python build_grid_stats.py data/
```

**기대 출력:** `grid_stats.json` (0.5~1MB, 신뢰 격자 1만 개 내외)

**소요 시간:** 단일 파일(약 850MB) 기준 약 2분 (단일 스레드, stdlib만 사용)

---

## 2. 데이터 전처리 (`preprocess.py`)

CSV를 읽어 결측치 채우고 12개 운동학 피처 + 3개 격자 컨텍스트 피처
(`sog_z_score`, `cell_density_log`, `is_rare_cell`) 계산. 총 15피처 출력.

```bash
# 단일 파일
python preprocess.py data/ais-2025-05-15.csv

# 여러 파일
python preprocess.py data/*.csv
python preprocess.py data/
```

**기대 출력:**
- `ais-YYYY-MM-DD_preprocessed.csv` — 전처리된 시퀀스 (행당 1 AIS 신호)
- `ais-YYYY-MM-DD_skip_log.csv` — 제외된 MMSI 사유

**소요 시간:** 8.8M 행 → 약 25분 (정렬 + 피처 계산 + I/O)

**주의:**
- `grid_stats.json`이 없으면 12피처만 출력 → 학습/평가 코드(15피처 기대)와 불일치
- 시퀀스 길이 < 10이거나 위치 점프 감지된 MMSI는 자동 제외

---

## 3. 비지도 학습 (`train_benchmark.py`)

정상 AIS 시퀀스로 재구성 기반 이상 탐지 모델 학습.

```bash
# 단일 모델 (기본 권장: dcdetect)
python train_benchmark.py --model dcdetect --input ais-2025-05-15_preprocessed.csv

# 풀스케일 (데스크탑 권장)
python train_benchmark.py --model all --input ais-2025-05-15_preprocessed.csv

# 옵션
python train_benchmark.py --model tranad \
    --input ais-2025-05-15_preprocessed.csv \
    --epochs 50 --lr 0.0005 --batch_size 256 --patience 7
```

**모델 9종:**
| 키 | 모델 | 메모 |
|---|---|---|
| `usad` | USAD | KDD 2020, adversarial AE |
| `tranad` | TranAD | VLDB 2022, 앙상블 보조 모델 |
| `conv1d` | Conv1D AE | 빠르고 ONNX 안정 |
| `lstm` | LSTM AE | seq=10에서는 약점 |
| `tcn` | TCN AE | dilated conv |
| `anomtrans` | AnomalyTransformer | NeurIPS 2022, KL 기반 |
| `dcdetect` | **DCdetector** | **KDD 2023, 앙상블 주력 (가중치 0.7)** |
| `iforest` | IForest-guided AE | sklearn 필요 |
| `ocsvm` | OCSVM-guided AE | sklearn 필요, 큰 데이터에서 느림 |

**산출물:**
- `model_<name>.onnx` — ONNX 모델 (입력 `x`, shape `(1, 10, 15)`)
- `scaler_<name>.json` — MinMax 스케일러 (피처별 min/max)
- `threshold_<name>.txt` — 이상 판정 임계값 (검증 데이터 95퍼센타일 MSE)

**소요 시간 (CPU 기준, SAMPLE_MMSI=2000, 15 epochs, batch 512):**
- DCdetect: 약 7분
- TranAD: 약 15~20분 (Transformer 2개 디코더)
- 전체 9개: 약 1~2시간

**GPU 기준 (RTX 3060/4060):**
- DCdetect: 약 1~2분
- 전체 9개: 약 15~30분

---

## 4. 학습 산출물 정리

`eval_anomaly.py`는 다음 경로를 기대:
```
output/scaler_<model>.json
output/<model>/model_<model>.onnx
output/<model>/threshold_<model>.txt
```

학습 후 자동 정리:
```powershell
# PowerShell
foreach ($m in @("usad","tranad","conv1d","lstm","tcn","anomtrans","dcdetect","iforest","ocsvm")) {
    if (Test-Path "model_$m.onnx") {
        New-Item -ItemType Directory -Path "output/$m" -Force | Out-Null
        Move-Item "model_$m.onnx" "output/$m/" -Force
        Move-Item "threshold_$m.txt" "output/$m/" -Force
        Copy-Item "scaler_$m.json" "output/" -Force
    }
}
```

```bash
# bash
for m in usad tranad conv1d lstm tcn anomtrans dcdetect iforest ocsvm; do
    [ -f "model_$m.onnx" ] || continue
    mkdir -p "output/$m"
    mv "model_$m.onnx" "threshold_$m.txt" "output/$m/"
    cp "scaler_$m.json" "output/"
done
```

---

## 5. 평가 (`eval_anomaly.py`)

15개 공격 시나리오(D1-LowSlow, E5-Shadow, F1-FeatSmooth 등)에 대해
탐지율 측정.

```bash
# Windows: 콘솔 인코딩 cp949 → UTF-8 강제
set PYTHONIOENCODING=utf-8

# 단일 모델
python eval_anomaly.py --model dcdetect

# 앙상블 (DCdetect 0.7 + TranAD 0.3)
python eval_anomaly.py --ensemble dcdetect,tranad --weights 0.7,0.3
```

**산출물:**
- `output/<model>/eval_result_<model>.txt` — 시나리오별 탐지율
- `output/<model>/*.png` — 시각화 (matplotlib 설치 시)
- 앙상블: `output/ensemble/eval_result_*.txt`

**소요 시간:** 약 2~5분

---

## 6. 지도 학습 (`train_supervised.py`) — 선택

레이블이 있는 데이터(정상 + 합성 이상)로 분류기 학습.
비지도 학습 모델과 별도로 평가.

```bash
# 단일 모델
python train_supervised.py --model patchtst

# 전체 (5종: patchtst, itrans, tsmixer, moderntcn, mamba)
python train_supervised.py --model all

# 데이터 양 조절
python train_supervised.py --model patchtst --n_anom 1000 --n_normal 20000
python train_supervised.py --model all --max_mmsi 200
```

**입력:** `data/*.csv` (전처리된 CSV 자동 탐색)
**산출물:** `output/sup_<name>/model_sup_<name>.onnx`, `output/scaler_sup.json`

평가:
```bash
python eval_anomaly.py --model sup_patchtst
```

---

## 7. 풀파이프라인 한 줄 실행 (데스크탑용 예시)

```bash
cd ml/

# 1. 격자 통계
python build_grid_stats.py data/

# 2. 전처리
python preprocess.py data/

# 3. 학습 (전체 9개 모델)
python train_benchmark.py --model all --input ais_preprocessed.csv

# 4. 산출물 정리 (Linux/Mac)
for m in usad tranad conv1d lstm tcn anomtrans dcdetect iforest ocsvm; do
    [ -f "model_$m.onnx" ] || continue
    mkdir -p "output/$m"
    mv "model_$m.onnx" "threshold_$m.txt" "output/$m/"
    cp "scaler_$m.json" "output/"
done

# 5. 평가
for m in dcdetect tranad anomtrans conv1d tcn; do
    python eval_anomaly.py --model "$m"
done

# 6. 앙상블 평가 (주력)
python eval_anomaly.py --ensemble dcdetect,tranad --weights 0.7,0.3
```

---

## 8. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `ImportError: DLL load failed while importing _C` (torch) | Python 3.13 + torch 2.x | Python 3.10 사용 또는 `pip install torch==2.5.0` |
| `embed_dim must be divisible by num_heads` | DCdetector 채널 어텐션 | train_benchmark.py에서 헤드 수가 n_feat의 약수가 되도록 자동 조정 (이미 적용됨) |
| `UnicodeEncodeError: 'cp949' codec can't encode` | Windows 콘솔 인코딩 | `set PYTHONIOENCODING=utf-8` 후 실행 |
| `FileNotFoundError: output/scaler_*.json` | 학습 산출물 위치 불일치 | 위 "4. 학습 산출물 정리" 스크립트 실행 |
| 학습이 너무 느림 | CPU 단독 | `SAMPLE_MMSI` 축소, `--epochs` 줄임, GPU 머신 사용 |
| `grid_stats.json` 없음 → 12피처만 출력 | build_grid_stats.py 누락 | 1단계부터 다시 |

---

## 9. 격자 컨텍스트 피처 (참고)

이 프로젝트는 12개 운동학 피처 + 3개 격자 컨텍스트 피처 = **총 15피처**.
격자 피처가 D1-LowSlow(저속 위장 공격) 탐지의 핵심.

| 피처 | 의미 |
|---|---|
| `sog_z_score` | 해당 격자 평균 SOG 대비 z-score (멈춰서면 안 되는 항로에서 정지 시 큰 값) |
| `cell_density_log` | 격자 내 과거 신호 수 log (저밀도 = 비정상 항행 가능성) |
| `is_rare_cell` | 표본 50건 미만인 격자 진입 (1=비정상, 0=정상) |

이 피처들은 `grid_stats.json`에서 lookup 되므로, 학습 데이터의 지리적
범위와 평가/배포 환경이 일치해야 함.

본 프로젝트는 Marine Cadastre US 동해안 데이터(35.5°N, -75.5°W
Cape Hatteras 부근)로 학습. 한국 해역 배포 시 한국 AIS 데이터로
`build_grid_stats.py` 재실행 필요.
