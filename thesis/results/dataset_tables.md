# Dataset Sample Tables for Thesis

## Behavioral Feature Set — Sample (first 15 rows, seed=42)

| profile      |   is_bot |   session_duration_sec |   checkout_velocity_sec |   mouse_speed_variance |   keystroke_flight_time_ms |   impossible_travel_flag |   is_datacenter_ip |   pages_visited |
|:-------------|---------:|-----------------------:|------------------------:|-----------------------:|---------------------------:|-------------------------:|-------------------:|----------------:|
| human_normal |        0 |                   67.7 |                    72.1 |                 117.75 |                      215.2 |                        0 |                  0 |               5 |
| human_normal |        0 |                  182.6 |                    58.8 |                  70.51 |                      123.1 |                        0 |                  0 |               8 |
| human_normal |        0 |                  105.2 |                    40.4 |                  72    |                      190.9 |                        0 |                  0 |               6 |
| human_normal |        0 |                   87.1 |                    37.6 |                 100.7  |                       58.1 |                        0 |                  0 |               6 |
| human_normal |        0 |                  127.1 |                    34.4 |                 104.18 |                      213.5 |                        0 |                  0 |               1 |
| human_normal |        0 |                  179.7 |                    33.1 |                 137.43 |                      117.9 |                        0 |                  0 |               7 |
| human_normal |        0 |                  165   |                    53.2 |                 110.62 |                      179.5 |                        0 |                  0 |               2 |
| human_normal |        1 |                  385.8 |                    40.2 |                  52.17 |                      148.1 |                        0 |                  0 |               6 |
| human_normal |        0 |                  165.8 |                    56.2 |                  75.5  |                      182.1 |                        0 |                  0 |               4 |
| human_normal |        0 |                   56.4 |                    27.1 |                  88.52 |                      130.9 |                        0 |                  0 |               9 |
| human_normal |        0 |                  163.6 |                    39.5 |                  71.31 |                      140.8 |                        0 |                  0 |               5 |
| human_fast   |        0 |                   28.5 |                    10   |                 115.66 |                      190   |                        0 |                  0 |               4 |
| human_normal |        0 |                   80.6 |                    64.5 |                  93.95 |                       51.8 |                        0 |                  0 |               8 |
| human_normal |        0 |                  134.1 |                    26.8 |                 133.92 |                      169.3 |                        0 |                  0 |               4 |
| human_normal |        0 |                  134.4 |                    38.7 |                  79.53 |                      138.9 |                        0 |                  0 |               9 |

## Transactional Feature Set — Sample (first 15 rows, seed=42)

| profile      |   is_bot |   is_fraud | transaction_type   |    amount |   old_balance_origin |   new_balance_origin |   old_balance_dest |   new_balance_dest |   balance_drain_ratio |   is_full_drain |   dest_balance_unchanged |
|:-------------|---------:|-----------:|:-------------------|----------:|---------------------:|---------------------:|-------------------:|-------------------:|----------------------:|----------------:|-------------------------:|
| human_normal |        0 |          0 | CASH_OUT           |  17766.4  |                -0.99 |                 0    |            1688.28 |           19454.7  |                0      |               0 |                        0 |
| human_normal |        0 |          0 | PAYMENT            |   2039.21 |              4130.73 |              2091.52 |               0    |            2039.21 |                0.4937 |               0 |                        0 |
| human_normal |        0 |          0 | CASH_OUT           |     36.78 |                36.78 |                 0    |               0    |              15.57 |                1      |               1 |                        0 |
| human_normal |        0 |          0 | CASH_OUT           | 161988    |                 0    |                 0    |               0    |          161988    |                0      |               0 |                        0 |
| human_normal |        0 |          0 | PAYMENT            | 332943    |                -0.23 |                 0    |               0    |          332943    |                0      |               0 |                        0 |
| human_normal |        0 |          0 | PAYMENT            | 247095    |                 0    |                 0    |               0    |          247095    |                0      |               0 |                        0 |
| human_normal |        0 |          0 | CASH_OUT           | 150635    |                 0    |                 0    |               0    |          150635    |                0      |               0 |                        0 |
| human_normal |        1 |          0 | CASH_OUT           |   9100.99 |             22788.1  |             13687.1  |               0    |            9100.99 |                0.3994 |               0 |                        0 |
| human_normal |        0 |          0 | PAYMENT            |  15362.9  |                 0    |                 0    |              37.49 |           15400.3  |                0      |               0 |                        0 |
| human_normal |        0 |          1 | CASH_OUT           |   9377.31 |                 0    |                 0    |           19510.9  |           28888.2  |                0      |               0 |                        0 |
| human_normal |        0 |          0 | CASH_OUT           |      0    |                 0    |                 0    |              -0.99 |           10913.9  |                0      |               0 |                        0 |
| human_fast   |        0 |          0 | CASH_OUT           |     54.11 |                54.11 |                 0    |               0.22 |              14.71 |                1      |               1 |                        0 |
| human_normal |        0 |          0 | PAYMENT            |  13237.8  |                 0    |                 0    |            1154.75 |           14392.5  |                0      |               0 |                        0 |
| human_normal |        0 |          1 | CASH_OUT           | 120801    |                 0    |                 0    |               0    |          120801    |                0      |               0 |                        0 |
| human_normal |        0 |          0 | PAYMENT            |     45.23 |                45.23 |                 0    |               0    |              23.64 |                1      |               1 |                        0 |

## Behavioral Features — Descriptive Statistics

|       |   session_duration_sec |   checkout_velocity_sec |   mouse_speed_variance |   keystroke_flight_time_ms |   impossible_travel_flag |   is_datacenter_ip |   pages_visited |
|:------|-----------------------:|------------------------:|-----------------------:|---------------------------:|-------------------------:|-------------------:|----------------:|
| count |             10000      |              10000      |             10000      |                 10000      |               10000      |         10000      |      10000      |
| mean  |               103.554  |                 38.3806 |                82.6274 |                   138.216  |                   0.0912 |             0.0894 |          4.6028 |
| std   |                57.9505 |                 22.0752 |                36.7401 |                    56.6544 |                   0.2879 |             0.2853 |          2.3422 |
| min   |                 0.9609 |                  0.4278 |                 0.1    |                     0      |                   0      |             0      |          1      |
| 25%   |                67.3622 |                 23.3972 |                63.3988 |                    99.0509 |                   0      |             0      |          3      |
| 50%   |               101.638  |                 37.9106 |                87.9159 |                   140.669  |                   0      |             0      |          5      |
| 75%   |               137.707  |                 51.8504 |               107.426  |                   176.615  |                   0      |             0      |          6      |
| max   |               386.386  |                245.689  |               204.657  |                   358.859  |                   1      |             1      |         14      |

## Transactional Features — Descriptive Statistics

|       |           amount |   old_balance_origin |   new_balance_origin |   old_balance_dest |   new_balance_dest |   balance_drain_ratio |   is_full_drain |   dest_balance_unchanged |
|:------|-----------------:|---------------------:|---------------------:|-------------------:|-------------------:|----------------------:|----------------:|-------------------------:|
| count |  10000           |      10000           |      10000           |    10000           |    10000           |            10000      |      10000      |               10000      |
| mean  | 417011           |          1.092e+07   |          1.05782e+07 |        1.66562e+09 |        1.66571e+09 |                0.3182 |          0.0639 |                   0.0436 |
| std   |      1.58147e+07 |          5.65853e+08 |          5.65635e+08 |        5.72989e+10 |        5.72989e+10 |                2.0268 |          0.2446 |                   0.2042 |
| min   |     -0.9997      |         -1           |          0           |       -1           |       -1           |                0      |          0      |                   0      |
| 25%   |    114.742       |          0           |          0           |        0           |      803.405       |                0      |          0      |                   0      |
| 50%   |   9162.25        |          0.6089      |          0           |        0           |    16131.9         |                0.0021 |          0      |                   0      |
| 75%   | 126496           |        873.399       |        272.859       |      491.078       |   209545           |                0.4676 |          0      |                   0      |
| max   |      1.30122e+09 |          5.48115e+10 |          5.48114e+10 |        4.26316e+12 |        4.26316e+12 |              128.928  |          1      |                   1      |

## Class Distribution (seed=42, N=10,000)

| Label      |   Count | Percentage   |
|:-----------|--------:|:-------------|
| is_bot=0   |    8261 | 82.6%        |
| is_bot=1   |    1739 | 17.4%        |
| is_fraud=0 |    7304 | 73.0%        |
| is_fraud=1 |    2696 | 27.0%        |

## Profile Distribution (seed=42, N=10,000)

| Profile      |   Count | Percentage   |
|:-------------|--------:|:-------------|
| human_normal |    7225 | 72.2%        |
| human_fast   |    1275 | 12.8%        |
| bot_standard |    1050 | 10.5%        |
| bot_stealth  |     450 | 4.5%         |
