#!/usr/bin/env bash
set -euo pipefail

data_root="${1:-/data/mpe_iclr_data}"
download_root="$data_root/downloads"
mkdir -p "$download_root"

names=(
  acs_ca_2024.zip
  acs_ny_2024.zip
  acs_tx_2024.zip
  census_2018_occupation_crosswalk.xlsx
  census_2022_industry_crosswalk.xlsx
  tlc_yellow_2024_01.parquet
  tlc_yellow_2024_04.parquet
  tlc_yellow_2024_07.parquet
  tlc_yellow_2024_10.parquet
  tlc_taxi_zones.zip
  citibike_2024_01.zip
  citibike_2024_07.zip
  citibike_2025_01.zip
  bts_on_time_2024_01.zip
  bts_on_time_2024_04.zip
  bts_on_time_2024_07.zip
  bts_on_time_2024_10.zip
  faa_aviation_facilities.csv
  amazon_raw_meta_all_beauty.parquet
)

urls=(
  https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pca.zip
  https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pny.zip
  https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_ptx.zip
  https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2018-occupation-code-list-and-crosswalk.xlsx
  https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2022-Census-Industry-Code-List-with-Crosswalk.xlsx
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-04.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-07.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-10.parquet
  https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
  https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.zip
  https://s3.amazonaws.com/tripdata/202407-citibike-tripdata.zip
  https://s3.amazonaws.com/tripdata/202501-citibike-tripdata.zip
  https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip
  https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_4.zip
  https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_7.zip
  https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_10.zip
  'https://ngda-transportation-geoplatform.hub.arcgis.com/api/download/v1/items/1551114f78e34d8395fd77bf41cd8a80/csv?layers=0'
  https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw_meta_All_Beauty/full-00000-of-00001.parquet
)

valid_file() {
  local path="$1"
  case "$path" in
    *.zip|*.xlsx) unzip -tqq "$path" >/dev/null 2>&1 ;;
    *.parquet)
      /home/byunhanjoon/2026/fintabrecipe/.conda/bin/python -c \
        'import pyarrow.parquet as pq, sys; pq.ParquetFile(sys.argv[1])' "$path" \
        >/dev/null 2>&1
      ;;
    *) [[ -s "$path" ]] ;;
  esac
}

pids=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  url="${urls[$index]}"
  destination="$download_root/$name"
  (
    if valid_file "$destination"; then
      echo "present $name"
    else
      partial="$destination.part"
      if [[ -e "$destination" ]]; then
        mv "$destination" "$partial"
      fi
      echo "fetch $name"
      curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
        --output "$partial" "$url"
      if ! valid_file "$partial"; then
        echo "failed validation: $name" >&2
        exit 1
      fi
      mv "$partial" "$destination"
    fi
  ) &
  pids+=("$!")
  if (( ${#pids[@]} >= 4 )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

sha256sum "$download_root"/*
