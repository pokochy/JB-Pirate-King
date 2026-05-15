"""
격자 통계 생성 스크립트

AIS CSV 파일들을 읽어 0.05도 격자별로 선박 운동학 통계
(평균 SOG, 표준편차, 표본 수)를 계산해 grid_stats.json으로 저장한다.

격자 통계는 preprocess.py와 eval_anomaly.py에서
컨텍스트 피처 3종 계산에 사용된다.

    sog_z_score      = (sog - cell_mean_sog) / (cell_std_sog + eps)
    cell_density_log = log(cell_count + 1)
    is_rare_cell     = (cell not in stats) ? 1 : 0

────────────────────────────────────────────────────────
사용:
    python build_grid_stats.py ais_preprocessed.csv
    python build_grid_stats.py data/*.csv
    python build_grid_stats.py data/

raw Marine Cadastre csv와 preprocessed csv 모두 지원.
컬럼명은 대소문자 무관 (LAT/lat/Latitude 등).
"""

import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

CELL_SIZE_DEG = 0.05
MIN_COUNT     = 50              # 이 미만의 격자는 통계 신뢰하지 않음
OUTPUT_FILE   = "grid_stats.json"

# 컬럼명 후보 (case-insensitive 매칭)
LAT_CANDIDATES = ["latitude", "lat"]
LON_CANDIDATES = ["longitude", "lon"]
SOG_CANDIDATES = ["sog"]


def resolve_input_files() -> list:
    if len(sys.argv) < 2:
        raise FileNotFoundError(
            "사용: python build_grid_stats.py <csv_or_glob_or_dir> [...]"
        )
    files = []
    for a in sys.argv[1:]:
        if os.path.isdir(a):
            files += sorted(glob.glob(os.path.join(a, "*.csv")))
        else:
            files += sorted(glob.glob(a))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError("입력 파일 없음")
    return files


def find_column(header_lower_to_orig: dict, candidates: list) -> str | None:
    for c in candidates:
        if c in header_lower_to_orig:
            return header_lower_to_orig[c]
    return None


def cell_key(lat: float, lon: float) -> str:
    lat_idx = math.floor(lat / CELL_SIZE_DEG)
    lon_idx = math.floor(lon / CELL_SIZE_DEG)
    return f"{lat_idx},{lon_idx}"


def main():
    input_files = resolve_input_files()
    print(f"[입력 {len(input_files)}개]")
    for f in input_files:
        print(f"  {f}")

    # Welford's online algorithm으로 격자별 평균/분산 누적
    n_map    = defaultdict(int)
    mean_map = defaultdict(float)
    M2_map   = defaultdict(float)
    total = 0

    for fpath in input_files:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"  [건너뜀] {fpath}: 헤더 없음")
                continue
            h_map = {k.lower().strip(): k for k in reader.fieldnames}
            lat_col = find_column(h_map, LAT_CANDIDATES)
            lon_col = find_column(h_map, LON_CANDIDATES)
            sog_col = find_column(h_map, SOG_CANDIDATES)
            if not (lat_col and lon_col and sog_col):
                print(f"  [건너뜀] {fpath}: lat/lon/sog 컬럼 부재")
                continue

            file_count = 0
            for row in reader:
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])
                    sog = float(row[sog_col])
                except (ValueError, KeyError, TypeError):
                    continue
                if not (-90 <= lat <= 90):   continue
                if not (-180 <= lon <= 180): continue
                if sog < 0 or sog > 102.2:   continue   # AIS SOG 상한

                key = cell_key(lat, lon)
                n_map[key] += 1
                delta = sog - mean_map[key]
                mean_map[key] += delta / n_map[key]
                delta2 = sog - mean_map[key]
                M2_map[key] += delta * delta2
                total += 1
                file_count += 1

                if total % 1_000_000 == 0:
                    print(f"  누적 {total:,} 행 처리...")
            print(f"  {os.path.basename(fpath)}: {file_count:,} 행")

    print(f"\n[격자 통계] 총 {total:,} 행, 격자 {len(n_map):,}개")
    if total == 0:
        raise RuntimeError("유효한 행이 없습니다. 입력 파일 확인 필요.")

    cells = {}
    for key, n in n_map.items():
        if n < MIN_COUNT:
            continue
        mean = mean_map[key]
        var  = M2_map[key] / n
        std  = math.sqrt(var)
        cells[key] = {
            "count":    n,
            "mean_sog": round(mean, 4),
            "std_sog":  round(std, 4),
        }

    kept = len(cells)
    print(f"  → 표본 {MIN_COUNT}건 이상 격자: {kept:,}개 "
          f"({kept/len(n_map):.1%}, 전체 신호의 "
          f"{sum(c['count'] for c in cells.values())/total:.1%} 포괄)")

    out = {
        "meta": {
            "cell_size_deg":    CELL_SIZE_DEG,
            "min_count":        MIN_COUNT,
            "total_signals":    total,
            "total_cells_seen": len(n_map),
            "total_cells_kept": kept,
        },
        "cells": cells,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"\n완료: {OUTPUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
