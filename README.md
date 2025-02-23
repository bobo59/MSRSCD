# MSRSCD

MSRS-CD Dataset link:https://pan.baidu.com/s/1Bb-uIh_uRFhrZaPM3i01dQ?pwd=MSRS 

The MSRS-CD dataset significantly complements existing RSCD datasets in terms of image resolution, change types, dataset size, and change dimensions, further providing a new benchmark for RSCD. This dataset comprises 841 pairs of remote sensing images captured in southern Chinese cities from 2019 to 2023, with each image sized at 1024×1024 pixels and a spatial resolution of 0.5 meters. The dataset is divided into training, validation, and testing sets in a ratio of 7:1:2. As shown in Fig. 1, the main types of changes in the dataset include new buildings, suburban expansion, vegetation changes, and road construction.

 ![image](https://github.com/user-attachments/assets/15a68f4c-72b2-45c1-9b1c-a7bd1efdd26b)

Fig. 1. The MSRS-CD dataset example, where Image T1 and Image T2 represent two remote sensing images at different times, and GT denotes the ground truth labels.



This is the result of the quantitative analysis of some of the networks on the dataset. (More networks are waiting to be updated.)
| Methods       | years | input | Precision | Recall | F1-score | IoU  | OA   | FLOPs(G) | Param(M) | Inference time(ms) |
|---------------|-------|-------|-----------|--------|----------|------|------|----------|----------|---------------------|
| FCEF          | 2018  | 256   | 74.99     | 66.69  | 70.59    | 45.44| 91.80| 3.55     | 1.35     | 1.90                |
| BIT           | 2021  | 256   | 75.73     | 70.79  | 73.18    | 57.70| 92.34| 8.75     | 3.04     | 11.58               |
| FCCDN         | 2022  | 256   | 75.56     | 71.31  | 73.37    | 65.42| 92.36| 12.49    | 6.31     | 15.29               |
| Changeformer  | 2022  | 256   | 72.22     | 72.94  | 72.58    | 56.96| 91.86| 202.79   | 41.03    | 28.78               |
| SGSLN         | 2023  | 256   | 77.39     | 69.73  | 73.36    | 56.28| 92.52| 11.50    | 6.04     | 10.31               |
| VcT           | 2023  | 256   | 76.64     | 69.72  | 73.02    | 57.50| 92.39| 10.64    | 3.57     | 14.68               |
| EATDer        | 2023  | 256   | 66.73     | 84.17  | 74.44    | 59.29| 91.47| 23.46    | 6.61     | 21.18               |
| AANet         | 2024  | 256   | 71.94     | 77.03  | 74.40    | 59.23| 92.17| 24.21    | 15.82    | 10.63               |



