# Dataset Generation

## Download official dataset
Please download ShapeNet and ShapeNetPart datasets from [Point-BERT_repo](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md)
```
│data/
├──│ShapeNet55-34/
│  ├──shapenet_pc/
│  │  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  │  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  │  ├── .......
│  ├──ShapeNet-55/
│  │  ├── train.txt
│  │  └── test.txt
├──|ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal/
│  ├──02691156/
│  │  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  │  ├── .......
│  │── .......
│  │──train_test_split/
│  └──synsetoffset2category.txt
```
## Process data for four tasks
To process the data, run the following command:
```
# Part Segmentation task
python gen_dataset_seg.py
# Reconstruction, Denoising, Registration tasks
python gen_dataset_cd.py
```
Please check your data folder architecture like this:
```
│data/
├──│ShapeNet55-34/
├──|ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──|Train_dataset/
│  ├── ShapeNet55/
│  └── partsegmentation/
│      ├── sources/
│      └── targets/
├──|Test_dataset/
│  ├── reconstruction/
│  ├── denoising/
│  ├── registration/
│  └── partsegmentation/
├──|train_list.json
├──|test_list.json
```