# cs231n-cyclone

This project estimates intensity and size of tropical cyclones from more than 70k frames collected worldwide.<br>
Data is available at [this webpage](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/).


## Requirements

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Proposed model

## Model Training

### Overfit case
Training and validation data in the overfitting case has been prepared for you. Please go to the **maxOverfit.ipynb** file and preform interactive training there.

Notice don't run two sections in the Jupyter Notebook unless you want to re-generate the training and validation data: **Read in info and get maxes**, **Calculate mean and std in training**.

Another reminder is that the last two sections **Playground** and **Calculate channel... in rolling manner** is also for pre-processing and playing with data. They are relevenat in the training.

### Normal case
Data has not been provided for this case, because they take up 60GB on disk. You need to generate the data by yourself.

Therefore, to perform training on a specified model, you have to do these commands (assume you have fulfilled all dependency requirements):

#### Create folders
You need to have two folders in ./normal/. That is **./normal/data** and **./normal/results**.

#### Generate training and validation data
```
python generate.py
```

#### Train

A use case is to train a Resnet34, for 100 epochs, with batch size 256, regularization strength 3e-3, initial learning rate 2e-3, step linear scheduler for learning rate decay and no center crop on images.
```
python train.py --modelSize 34 --epochTrain 100 --batchSize 256 --regStrength 3e-3 --lr 2e-3 --scheduler step --centerCrop False
```

## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:
```
@inproceedings{chen2018rotation,
  title={Rotation-blended CNNs on a New Open Dataset for Tropical Cyclone Image-to-intensity Regression},
  author={Chen, Boyo and Chen, Buo-Fu and Lin, Hsuan-Tien},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={90--99},
  year={2018},
  organization={ACM}
}
```
