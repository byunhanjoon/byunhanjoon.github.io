# MPE ICLR Dataset Manifest

Frozen prospectively on 2026-08-29 before any new target-bearing result was
computed or inspected. The five independent primary sources are ACS, NYC TLC,
Citi Bike, BTS, and Amazon Reviews 2023. Occupation and industry are clustered
as ACS; pickup/dropoff and origin/destination variants are likewise clustered
within their source. No task may be removed after an unfavorable result.

## Common rules

- A state is the complete value of the metric-aware field after only documented
  missing-value and official-code normalization.
- Primary tasks require at least 50 eligible rows per state. Secondary string
  tasks require 20 because the frozen benchmark reproductions are smaller.
- Controlled partitions assign eligible states, never rows, to train/validation/
  test in 60/20/20 proportions using the five frozen split seeds. Ties are
  resolved by normalized state ID. State sets must be pairwise disjoint.
- The splitter rejects and deterministically advances to the next hash ordering
  unless every test state has a finite training-metric neighbor and each part
  contains at least five states. This rule uses states and geometry only.
- A harder hierarchy/spatial-block split is additionally constructed without
  labels. Seen-state controls use a 60/20/20 row split stratified by state.
- Rows are selected by the smallest SHA-256 hash of source-stable row identity
  and global seed, never by target value. Raw archives are deleted only after a
  checksum and the processed Parquet file have been written.
- The isolated setting contains only the metric field plus intercept. The full
  setting uses exactly the ordinary covariates below for every representation.
- Regression target standardization uses training rows only. No test target is
  read until a validation-selected fit is final.

## PRIMARY EXTERNAL-METRIC sources

### ACS PUMS 2024: occupation and industry

- Source: [Census 2024 ACS PUMS documentation](https://www.census.gov/programs-surveys/acs/microdata/documentation/2024.html).
- Raw person files: `csv_pca.zip`, `csv_pny.zip`, and `csv_ptx.zip` from the
  Census 2024 one-year PUMS directory. These three populous states are frozen
  before outcomes to stay within disk limits while retaining broad support.
- Geometry authorities: the Census [industry/occupation crosswalk page](https://www.census.gov/topics/employment/industry-occupation/guidance/code-lists.html),
  `2018-occupation-code-list-and-crosswalk.xlsx`, and
  `2022-Census-Industry-Code-List-with-Crosswalk.xlsx`.
- Eligible rows: `18 <= AGEP <= 70`, employed/civilian working rows with finite
  positive `WAGP`, a valid metric-field code, and finite required covariates.
- Target: `log1p(WAGP)`.
- Full covariates: `AGEP, SCHL, WKHP, WKWN, ST, SEX, COW, ESR, MAR, RAC1P`.
  `PERNP`, `PINCP`, other earnings/income totals, allocation flags, occupation
  aliases on the industry task, and industry aliases on the occupation task are
  excluded.
- Row cap: 300,000 after eligibility filtering and deterministic hashing.

Task `acs_occupation` uses `SOCP` as the field. The official 2018 SOC levels
(major group, minor group, broad occupation, detailed occupation) form an
unweighted tree; shortest-path distance is primary. Aggregated `X/Y` codes are
retained as official nodes and never expanded using target statistics. The
ordinary full table excludes `OCCP`, occupation titles, and all ancestor fields.
The hard split holds out complete SOC minor groups whose remaining train
coverage is nonempty.

Task `acs_industry` uses `NAICSP`. The official 2022 NAICS prefix hierarchy
(sector, subsector, industry group, NAICS industry, national industry) forms the
tree; shortest-path distance is primary. Official `M/P/S/Z` aggregate codes are
represented as the documented highest unambiguous ancestor. The full table
excludes `INDP`, industry names, and ancestor fields. The hard split holds out
complete subsectors. Occupation and industry count as one ACS source for every
inferential summary.

### NYC TLC Yellow Taxi 2024: pickup and dropoff zones

- Source: [official TLC trip-record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- Raw files: official Yellow Taxi Parquet for January, April, July, and October
  2024, URL pattern
  `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-MM.parquet`.
- Zone metadata: official `taxi_zones.zip` from the TLC CloudFront `misc`
  directory. Geometry is reprojected only for centroid/adjacency computation;
  no target enters it.
- Eligible rows: valid pickup/dropoff timestamps and zone IDs, duration in
  `[60,10800]` seconds, and finite frozen covariates.
- Target: `log1p(dropoff_timestamp - pickup_timestamp)` in seconds.
- Full pre-trip covariates: pickup hour, day of week, passenger count, vendor,
  rate code, and the other endpoint zone. Actual distance, fare, tip, toll,
  payment, dropoff time components, and all post-trip amounts are excluded.
- Row cap: 300,000, balanced prospectively across the four months before hash
  sampling.

Tasks `tlc_pickup_zone` and `tlc_dropoff_zone` separately make the named endpoint
the metric field. Primary geometry is haversine distance between official zone
polygon centroids. Secondary geometry is unweighted shortest path on the
polygon-touch adjacency graph. Controlled splits use five state partitions;
the spatial-block split orders zones by frozen k-means on centroids and holds
out whole contiguous clusters. The two tasks count as one TLC source.

### Citi Bike: naturally new start stations

- Source: [official Citi Bike system-data page](https://citibikenyc.com/system-data)
  and its public `tripdata` S3 bucket.
- Frozen windows: January 2024 for training, July 2024 for validation, January
  2025 for test. Files are `YYYYMM-citibike-tripdata.zip`.
- Eligible rows: valid station IDs/coordinates/timestamps, duration in
  `[60,10800]` seconds, and complete frozen covariates.
- Target: `log1p(ended_at - started_at)` in seconds.
- Full covariates: start hour, day of week, member/casual, rideable type, and
  end-station ID. Duration-derived fields are excluded.
- Row cap: 150,000 per period, selected by stable ride-ID hash.

Task `citibike_start_station` has primary haversine geometry from the median
training-independent published coordinates for each station. A station is a
natural validation state if it appears in July but not January 2024; a natural
test state appears in January 2025 but in neither prior window. Rows at old
stations are used only for the temporal seen-state control. Primary test states
must have at least 50 January-2025 rows and a finite earlier-station neighbor.
Five controlled state holdouts on the pooled windows replicate the result.
Secondary network distance uses an unweighted station graph whose edges are
formed by observed endpoint pairs in training-period trips only, without
duration or any other target label.

### BTS on-time performance 2024: origin and destination airports

- Source: [BTS TranStats on-time table](https://transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=&gnoyr_VQ=FGJ).
- Raw files: January, April, July, and October 2024 from the official `PREZIP`
  endpoint, named
  `On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_M.zip`.
- Coordinates: the official FAA/BTS [Aviation Facilities dataset](https://catalog.data.gov/dataset/aviation-facilities),
  snapshotted at acquisition with checksum and release date.
- Eligible primary rows: completed, non-diverted, non-cancelled domestic flights
  with finite `ArrDelay`, airport coordinates, and scheduled covariates.
- Primary target: arrival delay in minutes. Secondary classification target:
  `ArrDel15`.
- Full scheduled/pre-departure covariates: month, day of week, reporting carrier,
  scheduled flight number, scheduled departure and arrival time, scheduled
  elapsed time, published route distance, and the other endpoint airport.
  Actual departure/arrival, taxi, wheels, cancellation, diversion, and delay-
  cause fields are forbidden.
- Row cap: 300,000, balanced across months before stable hash sampling.

Tasks `airline_origin_airport` and `airline_destination_airport` use haversine
distance between FAA coordinates as primary geometry. The secondary route graph
is built from scheduled origin/destination pairs present in training rows, with
no delay values. The spatial hard split holds out geographic FAA regions. Both
tasks count as one BTS source.

### Amazon Reviews 2023 product metadata: leaf category

- Source: [McAuley Lab Amazon Reviews 2023 dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
- Frozen file: `raw_meta_All_Beauty/full-00000-of-00001.parquet`, whose published
  size is 59.6 MB and SHA-256 is
  `14815cf1312a0d847364866e6876c8c73738993469067242870774c372c04387`.
- Eligible rows: finite positive price, nonempty hierarchical `categories`, and
  a stable `parent_asin`.
- Target: `log1p(price)`. No alternative target is permitted after outcomes.
- Full covariates: main category, store, average rating, rating count, title
  length, feature count, and description length. Category ancestors, category
  names outside the metric tokenizer, review text, and target-derived price
  bands are excluded.
- Row cap: 300,000 products by stable parent-ASIN hash.

Task `amazon_leaf_category` uses the final category in the published path as
the state and the union of published paths as the target-independent tree.
Primary distance is shortest path. Controlled splits require every validation/
test leaf to have a represented parent or sibling in training. The hard split
holds out complete subtrees one level below the root.

## SECONDARY STRING-METRIC panel

These reproduce three tasks from Cerda, Varoquaux, and Kegl's similarity-
encoding benchmark. They are not counted as external-metadata evidence. Each
uses five state-disjoint 60/20/20 partitions after the 20-row state threshold.
Primary distance is one minus character-trigram Jaccard similarity after lower-
casing and whitespace normalization. Jaro-Winkler and normalized Levenshtein
are secondary metrics. Similarity Encoding is the designated main baseline.

- `employee_salaries`, OpenML 42125: field `Employee Position Title`, target
  `Current Annual Salary`, with the leak-free ordinary columns documented by
  the skrub reproduction. Full dataset (about 9,228 rows), no outcome-based
  subsampling.
- `medical_charges`, OpenML 41444: field `DRG Definition`/medical procedure,
  target `Average Total Payments`, ordinary covariates provider state and
  average covered charges. Stable 100,000-row hash sample, matching the paper.
- `open_payments`, OpenML 41442: field company/manufacturer name, binary target
  research-protocol status, ordinary covariates payment amount and dispute
  flag. Full prepared OpenML data up to 100,000 rows.

Column aliases are resolved only by case/punctuation-normalized exact matching
against the names above; a genuinely absent required column makes the task
`NOT RUN — REQUIRED SOURCE SCHEMA UNAVAILABLE` and is logged, not replaced.

## OPTIONAL CONTROLLED-ACCESS

`mimic_iii` is frozen as `NOT RUN — CONTROLLED ACCESS UNAVAILABLE`. No
PhysioNet username/password or local MIMIC path was present at freeze. The
project must continue without it.

## Real nominal negative controls

The equality-metric controls are ACS `COW` (class of worker), TLC
`payment_type`, and BTS `Reporting_Airline`. They have no defensible geometry
for this program and enough states for state-disjoint partitions under the
same frequency rule. The controls use their source's frozen target and ordinary
covariates, compare lookup/support-complete categorical encodings, uniform PLE,
equality MPE, and ten random geometries, and are never reinterpreted as positive
metric evidence.

## Source-level inference

The five primary source units are exactly `ACS`, `NYC_TLC`, `CITI_BIKE`,
`BTS`, and `AMAZON_2023`. A source result is the equal-task, equal-setting mean
within that source after averaging training seeds and state splits. Multiple
fields, metrics, corruptions, seeds, and rows never create extra independent
source units.
