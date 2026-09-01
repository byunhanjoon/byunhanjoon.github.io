# Table 7 — Geographic and network showdown

| Task | Method | State-balanced MSE | Row-weighted MSE | Cells |
|---|---|---|---|---|
| airline_destination_airport | Coordinate Fourier | 0.8478 | 0.9562 | 10 |
| airline_destination_airport | MPE | 0.8447 | 0.9536 | 10 |
| airline_destination_airport | Nyström | 0.8446 | 0.9536 | 10 |
| airline_destination_airport | Raw coordinates | 0.8444 | 0.9534 | 10 |
| airline_destination_airport | Raw lat/lon | 0.8445 | 0.9535 | 10 |
| airline_destination_airport | Spatial RBF | 0.8445 | 0.9536 | 10 |
| airline_origin_airport | Coordinate Fourier | 1.3224 | 1.0574 | 10 |
| airline_origin_airport | MPE | 1.3192 | 1.0554 | 10 |
| airline_origin_airport | Nyström | 1.3199 | 1.0556 | 10 |
| airline_origin_airport | Raw coordinates | 1.3190 | 1.0550 | 10 |
| airline_origin_airport | Raw lat/lon | 1.3191 | 1.0552 | 10 |
| airline_origin_airport | Spatial RBF | 1.3199 | 1.0556 | 10 |
| citibike_start_station | Coordinate Fourier | 1.0185 | 0.9718 | 10 |
| citibike_start_station | MPE | 1.0163 | 0.9712 | 10 |
| citibike_start_station | Nyström | 1.0162 | 0.9707 | 10 |
| citibike_start_station | Raw coordinates | 1.0240 | 0.9687 | 10 |
| citibike_start_station | Raw lat/lon | 1.0234 | 0.9686 | 10 |
| citibike_start_station | Spatial RBF | 1.0165 | 0.9707 | 10 |
| tlc_dropoff_zone | Coordinate Fourier | 0.7792 | 0.9398 | 10 |
| tlc_dropoff_zone | Graph Laplacian | 0.9852 | 1.1812 | 10 |
| tlc_dropoff_zone | MPE | 0.6606 | 0.8513 | 10 |
| tlc_dropoff_zone | node2vec | 0.6481 | 0.8108 | 10 |
| tlc_dropoff_zone | Nyström | 0.6552 | 0.8518 | 10 |
| tlc_dropoff_zone | Raw coordinates | 0.8435 | 1.0563 | 10 |
| tlc_dropoff_zone | Raw lat/lon | 0.8448 | 1.0589 | 10 |
| tlc_dropoff_zone | Spatial RBF | 0.6540 | 0.8508 | 10 |
| tlc_pickup_zone | Coordinate Fourier | 1.0731 | 0.9093 | 10 |
| tlc_pickup_zone | Graph Laplacian | 1.6766 | 1.2001 | 10 |
| tlc_pickup_zone | MPE | 0.9758 | 0.8387 | 10 |
| tlc_pickup_zone | node2vec | 1.0264 | 0.8414 | 10 |
| tlc_pickup_zone | Nyström | 1.0188 | 0.8414 | 10 |
| tlc_pickup_zone | Raw coordinates | 1.0065 | 0.8747 | 10 |
| tlc_pickup_zone | Raw lat/lon | 0.9988 | 0.8646 | 10 |
| tlc_pickup_zone | Spatial RBF | 1.0152 | 0.8363 | 10 |
