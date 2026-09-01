# Table 1 — Final dataset panel

State counts use frozen split 0; unavailable sources remain visible.

| Source | Task | Status | Rows | States | Metric | Reason | Train states | Val states | Test states | Median support gap |
|---|---|---|---|---|---|---|---|---|---|---|
| ACS | acs_industry | RUN | 2.993e+05 | 234.0000 | unweighted shortest path on frozen official-code prefix hierarchy | — | 140.0000 | 46.0000 | 48.0000 | 4.0000 |
| ACS | acs_occupation | RUN | 2.973e+05 | 433.0000 | unweighted shortest path on frozen official-code prefix hierarchy | — | 259.0000 | 86.0000 | 88.0000 | 6.0000 |
| BTS | airline_destination_airport | RUN | 2.936e+05 | 212.0000 | haversine distance between official FAA airport coordinates | — | 127.0000 | 42.0000 | 43.0000 | 120.8513 |
| BTS | airline_origin_airport | RUN | 2.938e+05 | 215.0000 | haversine distance between official FAA airport coordinates | — | 129.0000 | 43.0000 | 43.0000 | 128.8023 |
| AMAZON_2023 | amazon_leaf_category | NOT RUN | — | — | — | Frozen raw_meta_All_Beauty snapshot contains no nonempty categories path, so the declared external hierarchy does not exist. | — | — | — | — |
| CITI_BIKE | citibike_start_station | RUN | 4.301e+05 | 1505.0000 | haversine distance between target-independent pooled published station coordinates | — | 903.0000 | 301.0000 | 301.0000 | 0.2428 |
| STRING_BENCHMARK | employee_salaries | RUN | 7923.0000 | 92.0000 | one minus padded character-trigram Jaccard similarity | — | 55.0000 | 18.0000 | 19.0000 | 0.6585 |
| STRING_BENCHMARK | medical_charges | RUN | 1.000e+05 | 100.0000 | one minus padded character-trigram Jaccard similarity | — | 60.0000 | 20.0000 | 20.0000 | 0.3718 |
| STRING_BENCHMARK | open_payments | NOT RUN | 73558.0000 | — | — | Frozen and active OpenML snapshots omit Total Amount of Payment, which the prospective manifest declared mandatory. | — | — | — | — |
| NYC_TLC | tlc_dropoff_zone | RUN | 2.980e+05 | 167.0000 | haversine distance between official taxi-zone polygon centroids | — | 100.0000 | 33.0000 | 34.0000 | 1.3210 |
| NYC_TLC | tlc_pickup_zone | RUN | 2.976e+05 | 92.0000 | haversine distance between official taxi-zone polygon centroids | — | 55.0000 | 18.0000 | 19.0000 | 0.8391 |
