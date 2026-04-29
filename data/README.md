# Data Directory

Place the PhysioNet 2012 dataset here:

```
data/
└── predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/
    ├── set-a/          # 4,000 patient .txt files
    ├── set-b/          # 4,000 patient .txt files
    ├── Outcomes-a.txt  # mortality labels for set-a
    └── Outcomes-b.txt  # mortality labels for set-b
```

**Download:** https://physionet.org/content/challenge-2012/1.0.0/  
No credentialed access required — freely available.

The notebook defaults to reading from the path configured in cell 1.  
Update `PHYSIONET_DIR` if your data is in a different location.
