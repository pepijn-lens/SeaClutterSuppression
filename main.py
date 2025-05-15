from radar import generate_bursts
import torch
import pyarrow.parquet as pq
import pandas as pd
from helper import plot_doppler
from radar import PulsedRadar
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    n_bursts = 3000
    n_targets = 6

    for target in range(n_targets):
        # Generate radar bursts
        if target == 0:
            generate_bursts(device, n_bursts=500, num_targets=target, dir="data/bursts_classification")
        else:
            generate_bursts(device, n_bursts=n_bursts, num_targets=target, dir="data/bursts_classification")

    print("Radar bursts generated successfully.")