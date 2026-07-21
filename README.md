# Fourier Y

Wafer map chip-level `.csv` 또는 `.txt` 파일에서 wafer별 annulus Fourier y-value를 계산하는 공유용 repository입니다. 입력 구분자는 comma 또는 tab을 자동으로 시도합니다.

## Files

- `wafer_fourier_y.ipynb`: 사용자가 입력 파일, radius, target harmonic 등만 넣고 실행하는 notebook입니다.
- `fourier_y.py`: 실제 CSV/TXT 로딩, encoding 처리, wafer polar normalization, Fourier 계산 로직입니다.
- `spoke_fourier_analysis.ipynb`: `bin_no` 기반 spoke 형태 불량을 low-frequency 집중도와 sinc spectrum 형태로 찾는 notebook입니다.
- `spoke_fourier.py`: spoke 분석용 CSV/TXT 로딩, wafer normalization, Fourier spectrum scoring 로직입니다.
- `test_spoke_fourier.py`: radial spoke가 full ring보다 높은 점수를 받는지 확인하는 회귀 테스트입니다.
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

입력 파일은 `.csv` 또는 `.txt`를 사용합니다. 기본은 comma-separated 형식이며, tab, semicolon, pipe delimiter도 순서대로 시도합니다. 입력 파일은 bytes를 먼저 디코드한 뒤 필요한 칼럼만 읽으므로 UTF-8, CP949/EUC-KR, UTF-16 계열 export를 자동으로 처리합니다.

행 끝에 delimiter가 추가로 붙어 row별 칼럼 수가 달라지는 경우에는 초과 칼럼을 잘라내고 필요한 칼럼만 읽습니다.

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

## Spoke Fourier Analysis

`spoke_fourier_analysis.ipynb`는 중앙에서 바깥 방향으로 좁은 각도에 길게 뻗는 spoke 형태 불량을 찾기 위한 별도 workflow입니다. 기존 `fourier_y.py`와 `wafer_fourier_y.ipynb`는 그대로 유지됩니다.

### Spoke Input Columns

기본 raw data에는 아래 역할을 하는 칼럼이 필요합니다. 실제 파일의 칼럼명이 다르면 notebook의 `Column Mapping`에서 raw 칼럼명을 지정합니다. raw data에 다른 칼럼이 많아도 필요한 칼럼만 projection해서 읽습니다. 입력 파일은 bytes를 먼저 디코드한 뒤 Python CSV parser로 필요한 칼럼만 읽습니다.

```text
root lot id
wafer id
chip x position
chip y position
bin number
```

### Spoke Notebook Usage

사용자는 보통 아래 값만 수정합니다.

```python
INPUT_FILE = Path("input.txt")
OUTPUT_CSV = Path("spoke_fourier_output.csv")

# Column Mapping: raw file 안의 실제 칼럼명을 지정합니다.
ROOT_LOT_ID_COL = "root_lot_id"
WAFER_ID_COL = "wafer_id"
CHIP_X_COL = "chip_x_pos"
CHIP_Y_COL = "chip_y_pos"
BIN_NO_COL = "bin_no"

# list 또는 scalar 모두 허용합니다.
DEFECT_BIN_NOS = [12]

ANGULAR_BINS = 360
MIN_CHIPS = 20
```

`DEFECT_BIN_NOS = 12`처럼 scalar로 넣어도 내부에서 단일 원소 list처럼 처리합니다. 입력한 `bin_no` 값만 defect로 보고, 나머지 chip도 wafer를 반지름 1인 원으로 정규화하는 데 사용합니다.

spoke workflow는 선택된 raw 칼럼을 먼저 문자열로 읽은 뒤, 좌표 칼럼만 계산 시점에 숫자로 변환합니다. 따라서 `wafer_id`가 `10` 같은 숫자형이거나 `W01` 같은 문자형이어도 처리할 수 있습니다.

검증용 wafer는 전체 분석 실행 후 `Selected Wafer Validation` 셀에서 별도로 선택합니다.

```python
WAFER_TO_PLOT = ("ABCDE", 10)
```

선택 wafer는 문자열 기준으로 비교하므로 `WAFER_TO_PLOT = ("ABCDE", 10)`과 `("ABCDE", "10")`은 같은 wafer를 찾습니다. 파일 값이 `W01`이면 `("ABCDE", "W01")`처럼 입력합니다.

선택한 wafer 출력에는 전체 chip map도 표시됩니다. `DEFECT_BIN_NOS`에 해당하는 chip은 검은색, 나머지는 흰색이며, 좌표에서 추정한 x/y pitch 크기의 직사각형으로 빈 공간 없이 배치됩니다. 시각화에 사용된 정규화 좌표는 notebook 변수 `wafer_map_df`에 남습니다.

예를 들어 raw file 칼럼명이 `LOT`, `WF`, `X`, `Y`, `BIN`이면 아래처럼 바꾸면 됩니다.

```python
ROOT_LOT_ID_COL = "LOT"
WAFER_ID_COL = "WF"
CHIP_X_COL = "X"
CHIP_Y_COL = "Y"
BIN_NO_COL = "BIN"
```

### Spoke Calculation

입력한 defect bin set을 `B`라고 하면 chip별 defect indicator는 아래와 같습니다.

```text
d_i = 1(bin_no_i in B)
```

모든 반경 영역을 사용하고 theta 방향을 360개 bin으로 나눠 angular defect-rate signal을 만듭니다.

```text
p(theta_j) = mean(d_i | theta_i in bin_j)
```

각 harmonic amplitude는 아래처럼 계산합니다.

```text
A_k = 2 * | mean((p(theta_j) - mean(p)) * exp(-i * k * theta_j)) |
```

실제 spoke에서 나타나는 강한 low-frequency energy와 sinc 형태의 부드러운 spectrum을 보상하고, 전체 frequency에 퍼지는 ring 격자 artifact는 broadband energy와 spectral roughness로 감점합니다.

```text
E_low = sum(A_k^2 for k in low-frequency band)
E_broad = sum(A_k^2 for k in broadband band)
low_freq_energy_ratio = E_low / E_total

T_k(width) = abs(sin(k * width / 2) / k)
sinc_similarity = max cosine_similarity(A, T(width))

spectral_smoothness = 1 / (1 + spectral_roughness)

spoke_fourier_signal = sqrt(E_low / 2)
    * low_freq_energy_ratio
    * sinc_similarity
    * spectral_smoothness
    * (1 - broadband_energy_ratio)
```

spoke 기인 불량률은 추정된 spoke 폭으로 가장 강한 각도 구간을 찾고, 구간 바깥의 평균 불량률을 background로 차감합니다.

```text
background_rate = outside_defect_chips / outside_chips
spoke_defect_chip_count_estimate = max(
    0,
    sector_defect_chips - background_rate * sector_chips,
)
spoke_defect_rate = spoke_defect_chip_count_estimate / total_chip_count
```

`ANGULAR_BINS = 360`이면 계산 가능한 harmonic은 `1..180`입니다. 기본 low-frequency band는 `1..72`, broadband band는 `90..180`이며 별도의 사용자 입력 없이 자동 설정됩니다. sinc template은 `1..45 degree` 폭에서 가장 잘 맞는 값을 자동 탐색합니다.

### Spoke Output Columns

- `defect_rate`: 입력한 defect bin의 wafer 전체 chip 비율
- `spoke_defect_rate`: 검출된 spoke 각도 구간에서 background를 차감한 spoke 기인 추정 불량률
- `spoke_defect_chip_count_estimate`: spoke에 기인한 것으로 추정되는 background 보정 chip 수
- `spoke_sector_defect_rate`: 검출된 spoke 각도 구간 내부의 실제 선택-bin 불량률
- `spoke_background_defect_rate`: spoke 구간 바깥의 선택-bin 평균 불량률
- `spoke_theta_center_deg`: 검출된 spoke 중심 각도
- `spoke_fourier_signal`: low-frequency 집중도, sinc 유사도, smoothness와 broadband penalty를 결합한 대표 점수
- `low_freq_fourier_signal`: low-frequency band의 Fourier energy 크기
- `low_freq_energy_ratio`: 전체 Fourier energy 중 low-frequency band 비율
- `sinc_similarity`: 실제 spectrum과 최적 sinc template의 유사도
- `estimated_spoke_width_deg`: 최적 sinc template에서 추정한 spoke 각도 폭
- `spectral_smoothness`: spectrum이 불규칙한 spike보다 부드러운 파동 형태에 가까운 정도
- `broadband_energy_ratio`: 전체 frequency에 퍼진 energy 비율
- `signal_to_noise`: low-band RMS / broadband RMS 내부 참고값
- `high_freq_fourier_signal`: 이전 출력 호환을 위해 남긴 broadband 진단값이며 정렬 기준으로 사용하지 않음
- `theta_signal_df`: 선택 wafer의 theta-bin별 defect rate Polars DataFrame
- `harmonic_spectrum_df`: 선택 wafer의 harmonic amplitude와 matched sinc template을 포함한 Polars DataFrame

CLI로도 실행할 수 있습니다.

```bash
python3 spoke_fourier.py input.csv \
  --defect-bin-nos 12,13 \
  -o spoke_fourier_output.csv \
  --group-cols LOT,WF \
  --x-col X \
  --y-col Y \
  --bin-col BIN \
  --angular-bins 360
```

회귀 테스트는 아래 명령으로 실행합니다.

```bash
python3 -m unittest -v test_spoke_fourier.py
```
