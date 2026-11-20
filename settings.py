from pathlib import Path

H5_PATH = r"C:\Users\26664984\Documents\Masters\hdf5_training_data\iteration2\hydrofoil_rev2.h5" # packed dataset, written by pack_h5.py
OUT_DIR = r"C:\Users\26664984\Documents\Masters\Model_training\cosine\Normalisation\surfaceloss_fixed" # where train.py writes a run
RUN_DIR = Path(r"C:\Users\26664984\Documents\Masters\Model_training\cosine\Normalisation\surfaceloss_fixed") # the finished run read by evaluate.py, forces.py and plot.py

RHO = 1025.0 
