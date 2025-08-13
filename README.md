# HOI-Det

This repo provides code for hand-object contact detection using HOT3D dataset.
1. Hand-object contact data extraction from HOT3D dataset
2. Object detector (DETR) training and evaluation on HOT3D dataset
3. Contact detector training and evaluation on HOT3D dataset

## Dataset Setup

Refer to https://github.com/facebookresearch/hot3d and follow their instructions to setup the environment and download the necessary data.
It is not necessary to download HOT3D Quest data, as this repo only uses HOT3D Aria data.

Next, extract the contact data from HOT3D using the following:
```
cd hot3d/hot3d
bash extract_contact_script.sh
```
Note that you should modify 
`
BASE_PATH
`
and
`
MANO_MODEL_PATH
`
in the script according to the path you saved the data. In addition, you can change the sequence ID (currently it's P0001 to P0010) to suit your need. 

Once the process was done, it should create a new folder 
`
images/rgb/
`
within each sequence.
Additionally, you can add the 
`
--save_viz
`
flag for 
`
HOT3D_extract_images_contacts.py
`
to visualize the bbox for verification.

Last, modify 
`
train.json
`
and
`
val.json
`
accordingly based on your need.

## Train Object Detector (DETR)

Run
```
torchrun --nproc_per_node=NUM_GPUs train_detr_dist.py --data_root PATH_TO_DATA
```

