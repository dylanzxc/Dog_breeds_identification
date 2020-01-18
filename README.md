# Dog Breed Classification

## Project Structure
```
base.py   # main file to train the network
dataProcess.py   # pre-process input images
network.py   # network constructor and trainer
```

## How to run the code
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 base.py
```

## Folder structure
```
.
├── base.py               ## main program
├── dataProcess.py        ## image pre-process
├── dog_breed_VGG16.py
├── labels.csv            ## label of class-picture
├── network.py            ## neural network
├── README.md
├── requirements.txt
├── resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
├── sample_submission.csv
├── test                  ## data for kaggle, not usable yet
├── train                 ## train data, has 7k images
└── val                   ## validation data, has 3k images
```
