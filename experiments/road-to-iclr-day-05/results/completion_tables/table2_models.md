| model | parameter count | training budget | schema handling | seed count |
| --- | --- | --- | --- | --- |
| mlp | 17793–26625 | 20 epochs AdamW | dense one-hot blocks | 4 |
| resnet | 133889–142721 | 20 epochs AdamW | dense one-hot blocks | 4 |
| ft_transformer | 48769–66433 | 20 epochs AdamW | dense stem + feature tokens | 4 |
| tabm | 54596–59592 | 20 epochs AdamW; k=4 | dense stem + internal members | 4 |
| tabpfn | pretrained checkpoint | pretrained; 1/8 inference members | native numeric/categorical indices | 1 |
| onehot_linear | data-dependent / n.a. | converged / ridge closed-form | dense one-hot blocks | 4 |
| native_histgb | data-dependent / n.a. | 80 boosting iterations | native ordinal categories | 4 |
| catboost_native | data-dependent / n.a. | 80 boosting iterations | native categorical strings | 4 |
| xgboost | data-dependent / n.a. | 80 boosting iterations | ordinal numeric input | 4 |
| lightgbm | data-dependent / n.a. | 80 boosting iterations | declared categorical columns | 4 |
