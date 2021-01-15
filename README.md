# postMRI
A unified package for MRI: image reconstruction, analysis, and diagnosis

- Acceleration module

```bash
python models/unet/train_unet.py --challenge [multicoil | singlecoil] \
--data-path [data path] --exp-dir [exp dir] \
--accelerations [acc. factor] --center-fractions [center fractions] --mask-type [mask type] \
--sumpath [summary path]
```

- Diagnosis module

```bash
python diagnosis/Train.py --data-path [data path] --exp-dir [exp dir] --gpu [gpu id]
```
