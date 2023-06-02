# benchmark-pytorch-mmodel

    A simple performance benchmark of the basic multi-model parallelization of Linear, Conv1d and grouped-Conv1d layers in PyTorch

----

To parallelize computations of multiple Linear layers in PyTorch, one could easily implement in three ways:

- `Linear` * n_model
- `Conv1d[k=1]` * n_model
- `Conv1d[k=1, n_group=n_model]`

Which one is the most efficient? 🤔


### run

- run `python run.py` for single model
- run `python run.py --n_model 8` for multi models


### results

⚪ single model

```
==============================
>> [go_linear]
>> model define: 0.076s
     param_cnt: 131584
>> data: 4.362s
>> zero grad: 0.364s
>> forward: 2.721s
>> backward: 2.985s
>> optim: 1.453s
>> vram usage:
     max alloc: 18470 MB
     max resrv: 22528 MB
>> go_linear done in 11.980s
==============================
>> [go_conv1d]
>> model define: 0.001s
     param_cnt: 131584
>> data: 4.388s
>> zero grad: 0.380s
>> forward: 2.063s
>> backward: 2.951s
>> optim: 1.346s
>> vram usage:
     max alloc: 22052 MB
     max resrv: 43008 MB
>> go_conv1d done in 11.150s
==============================
```

⚪ multi model

```
[n_model=8]
==============================
>> [go_mlinear]
>> model define: 0.011s
     param_cnt: 1052672
>> data: 8.670s
>> zero grad: 0.595s
>> forward: 6.813s
>> backward: 8.614s
>> optim: 1.947s
>> vram usage:
     max alloc: 31280 MB
     max resrv: 57344 MB
>> go_mlinear done in 26.690s
==============================
>> [go_mconv1d]
>> model define: 0.007s
     param_cnt: 1052672
>> data: 8.867s
>> zero grad: 0.548s
>> forward: 7.878s
>> backward: 9.076s
>> optim: 1.901s
>> vram usage:
     max alloc: 31280 MB
     max resrv: 57344 MB
>> go_mconv1d done in 28.307s
==============================
>> [go_gconv1d]
>> model define: 0.006s
     param_cnt: 1052672
>> data: 9.147s
>> zero grad: 0.366s
>> forward: 1.556s
>> backward: 4.285s
>> optim: 1.416s
>> vram usage:
     max alloc: 31280 MB
     max resrv: 57344 MB
>> go_gconv1d done in 16.819s
==============================

[n_model=32]
==============================
>> [go_mlinear]
>> model define: 0.027s
     param_cnt: 4210688
>> data: 35.733s
>> zero grad: 1.938s
>> forward: 23.139s
>> backward: 25.880s
>> optim: 3.508s
>> vram usage:
     max alloc: 75200 MB
     max resrv: 94208 MB
>> go_mlinear done in 90.273s
==============================
>> [go_mconv1d]
>> model define: 0.024s
     param_cnt: 4210688
>> data: 35.871s
>> zero grad: 1.945s
>> forward: 27.272s
>> backward: 30.205s
>> optim: 3.585s
>> vram usage:
     max alloc: 75200 MB
     max resrv: 94208 MB
>> go_mconv1d done in 98.947s
==============================
>> [go_gconv1d]
>> model define: 0.017s
     param_cnt: 4210688
>> data: 35.214s
>> zero grad: 0.839s
>> forward: 3.891s
>> backward: 8.168s
>> optim: 1.781s
>> vram usage:
     max alloc: 75200 MB
     max resrv: 131072 MB
>> go_gconv1d done in 49.947s
==============================

[n_model=64]
==============================
>> [go_mlinear]
>> model define: 0.061s
     param_cnt: 8421376
>> data: 68.585s
>> zero grad: 3.833s
>> forward: 46.170s
>> backward: 49.441s
>> optim: 5.946s
>> vram usage:
     max alloc: 133760 MB
     max resrv: 161792 MB
>> go_mlinear done in 174.109s
==============================
>> [go_mconv1d]
>> model define: 0.046s
     param_cnt: 8421376
>> data: 68.736s
>> zero grad: 3.887s
>> forward: 54.072s
>> backward: 52.666s
>> optim: 5.990s
>> vram usage:
     max alloc: 133760 MB
     max resrv: 161792 MB
>> go_mconv1d done in 185.479s
==============================
>> [go_gconv1d]
>> model define: 0.038s
     param_cnt: 8421376
>> data: 69.732s
>> zero grad: 1.415s
>> forward: 5.834s
>> backward: 12.117s
>> optim: 1.886s
>> vram usage:
     max alloc: 133760 MB
     max resrv: 260096 MB
>> go_gconv1d done in 91.070s
==============================
```

----

by Armit
2023/06/02 
