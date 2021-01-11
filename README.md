# postMRI
A unified package for MRI: image reconstruction, analysis, and diagnosis


diagnosis module:
python diagnosis/Train.py --data-path [data path] --exp-dir [exp dir] --gpu [gpu id]

acceleration module:
python models/unet/train_unet.py --challenge [multicoil | singlecoil] --data-path [data path] --exp-dir [exp dir] --accelerations [acc. factor] --center-fractions [center fractions] --mask-type [mask type: equispaced] --sumpath [summary path]
