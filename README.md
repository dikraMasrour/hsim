# hsim
### Run test.py for energy measurement

``` python -m main.test```

**Options:** <br>
```-model: specify model type```

    'baseline_binary': for the vanilla CNN without band selection network (all-bands) 

    'mlbs_binary': for the MLBS network (band selection + CNN) 

```-r: specify band selection ratio for MLBS type models```

    values: 0.15, 0.3, 0.5 or 0.75
