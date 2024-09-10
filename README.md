# Running the code

1. Install Miniconda: https://docs.anaconda.com/miniconda/
2. Create a virtual environment. In your terminal, run the command:
```
conda create -n aloha python=3.10
```

3. Activate the virtual environment.
```
conda activate aloha
```
4. Install all packages.
```
pip install -r requirements.txt
```
5. Setup is complete.

Make sure you are in the overall folder and not in a subfolder.

- Run the training script to train the demonstrator:
```
python -m src.training
```
- Run the dagger script to train the imitator (edit dagger.py if the trained demonstrator is in an undexpected folder. Should be in ./old_models/passive/model.pth)
```
python -m src.dagger
```
- To visualize the model's trajectories (make sure to uncomment lines 49-59, will be changed soon):
```
python -m src.visualize
```
