# Action-Recognition
Action Recognition from video with VIT connect

Conda setup:

Create new conda environment for action recognition:
> conda create --prefix ar python=3.8
> conda activate ar

Depending on your cuda version install the coresponding torch and cudatoolkit
> conda install pytorch=1.11.0 torchvision=0.12 torchaudio=0.11 cudatoolkit=10.2 -c pytorch -c conda-forge

Import environment.yml
> conda env update --name ar --file environment.yml --prune

you are good to go!!