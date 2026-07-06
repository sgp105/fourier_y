# Fourier Y

Wafer map chip-level `.csv` 또는 comma-separated `.txt` 파일에서 wafer별 annulus Fourier y-value를 계산하는 공유용 repository입니다.

## Files

- `wafer_fourier_y.ipynb`: 사용자가 입력 파일, radius, target harmonic 등만 넣고 실행하는 notebook입니다.
- `fourier_y.py`: 실제 CSV/TXT 로딩, encoding 처리, wafer polar normalization, Fourier 계산 로직입니다.
- `requirements.txt`: 동작 확인용 Python package version입니다.

## Auto Install

Notebook 첫 번째 셀은 아래 package가 설치되어 있지 않으면 exact version으로 자동 설치합니다.

```text
numpy==1.26.4
polars==1.14.0
matplotlib==3.9.2
```

`fourier_y.py`를 CLI로 실행할 때는 계산에 필요한 `numpy`, `polars`만 자동 설치합니다. Notebook 차트에는 `matplotlib`도 사용합니다.

## Input Columns

기본 설정 기준 필수 칼럼은 아래와 같습니다.

```text
root_lot_id
wafer_id
item_id
chip_x_pos
chip_y_pos
y_value
last_update_time
```

입력 파일은 comma-separated `.csv` 또는 `.txt`를 사용합니다. UTF-8, CP949/EUC-KR, UTF-16 계열 export를 자동으로 처리합니다.

같은 `root_lot_id + wafer_id`에 여러 snapshot이 있으면 `last_update_time`이 가장 최신인 row만 사용합니다.

사용자는 `item_id` 칼럼값도 입력합니다. 예를 들어 `item_id`가 `MSR0022`인 row만 계산하려면 notebook에서 `ITEM_ID = "MSR0022"`로 지정합니다.

## Notebook Usage

1. `wafer_fourier_y.ipynb`를 엽니다.
2. 첫 번째 셀을 실행해서 package를 확인/자동 설치합니다.
3. `User Inputs` 셀에서 아래 값을 수정합니다.

```python
INPUT_FILE = Path("input.txt")
OUTPUT_CSV = Path("fourier_y_output.csv")
ITEM_ID = "MSR0022"
WAFER_TO_PLOT = ("ABCDE", 10)

INNER_RADIUS = 0.6
OUTER_RADIUS = 1.0
TARGET_HARMONIC = 16
ANGULAR_BINS = 384
MIN_RING_CHIPS = 20
```

4. `Run` 셀을 실행합니다.
5. 결과 CSV는 `OUTPUT_CSV` 경로에 저장됩니다.
6. 분석에 실제 사용된 chip-level Polars DataFrame은 notebook 변수 `analysis_df`에 남습니다.
7. `WAFER_TO_PLOT`에 지정한 wafer의 `mean_y_value in annulus vs theta [rad]` 차트와 harmonic amplitude spectrum을 확인합니다.

## CLI Usage

Notebook 없이 command line에서 바로 실행할 수도 있습니다.

```bash
python3 fourier_y.py input.txt -o fourier_y_output.csv
```

옵션 예시:

```bash
python3 fourier_y.py input.csv \
  -o output.csv \
  --item-id MSR0022 \
  --inner-radius 0.6 \
  --outer-radius 1.0 \
  --target-harmonic 16 \
  --angular-bins 384
```

## Calculation

Annulus 영역 안에서 theta 방향 signal을 만듭니다.

```text
y(theta_j) = mean(y_i | theta_i in bin_j, r_i in [r_in, r_out])
```

평균 offset, 즉 DC 성분을 제거한 뒤 target harmonic amplitude를 계산합니다.

```text
A_k = 2 * | mean((y(theta_j) - mean(y)) * exp(-i * k * theta_j)) |
```

기본 target은 `k=16`입니다.

```text
fourier_y_value = A_target
signal_to_noise = A_target / (std(y(theta)) + 1e-12)
snr_weighted_fourier_y = fourier_y_value * signal_to_noise
```

`snr_weighted_fourier_y`는 16주기 성분이 강하고 전체 angular variation 대비 선명한 wafer를 위로 올리는 실무 score입니다.

## Output Columns

주요 출력 칼럼:

- `fourier_y_value`: target harmonic amplitude
- `signal_to_noise`: `fourier_y_value / ring_std_y`
- `snr_weighted_fourier_y`: `fourier_y_value * signal_to_noise`
- `dominant_harmonic`: 계산에 사용된 target harmonic
- `dominant_phase_rad`: target harmonic phase
- `ring_mean_y`: annulus theta signal 평균
- `ring_std_y`: annulus theta signal 표준편차
- `ring_chip_count`: annulus 계산에 사용된 chip 수
- `ring_coverage`: 전체 유효 chip 중 annulus chip 비율

Notebook 변수:

- `result`: wafer-level 결과 Polars DataFrame
- `analysis_df`: `item_id` 필터와 latest snapshot 필터가 적용되고 `_theta`, `_radius_norm`이 추가된 chip-level Polars DataFrame
- `theta_signal_df`: `WAFER_TO_PLOT`에 해당하는 wafer의 annulus theta-bin 평균 y-value Polars DataFrame
- `harmonic_spectrum_df`: harmonic 1부터 `ANGULAR_BINS / 2`까지의 amplitude Polars DataFrame
