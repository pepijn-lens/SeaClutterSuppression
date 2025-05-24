import numpy as np
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from datetime import datetime, timedelta
import torch 
import os
import json
from tqdm import trange

from helper import calculate_resolution, plot_doppler

import matplotlib.pyplot as plt

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Speed of light (m/s)
c0 = 299792458
INDICES = np.arange(97, 161)


class TargetSimulator:
    def __init__(self, num_steps=20, state_vectors=[[1000, 20, 1000, 20]], randomness=[0.1, 0.1]):
        self.num_steps = num_steps
        self.state_vectors = state_vectors  # Now a list of initial states, one per target
        self.start_time = datetime.now()
        self.randomness = randomness  # Keep randomness the same for all targets for simplicity
        self.transition_model = CombinedLinearGaussianTransitionModel([
            ConstantVelocity(randomness[0]),
            ConstantVelocity(randomness[1])
        ])
        self.truths = self._generate_ground_truths()  # List of GroundTruthPaths

    def _generate_ground_truths(self):
        truths = []

        for initial_state in self.state_vectors:
            timesteps = [self.start_time]
            truth = GroundTruthPath([GroundTruthState(initial_state, timestamp=timesteps[0])])

            for k in range(1, self.num_steps + 1):
                timesteps.append(self.start_time + timedelta(seconds=k))
                new_state = self.transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1))
                truth.append(GroundTruthState(new_state, timestamp=timesteps[k]))

            truths.append(truth)

        return truths


class PulsedRadar:
    # Voor een range resolution van 1 meter, moet de bandbreedte 150 MHz zijn (dus de sampling rate 150*2 MHz). 
    # Voor een velocity resolution van 0.5 m/s, moet de pulse repetition interval 66.7 microseconds zijn.
    def __init__(self, BW=50e6, fs=100e6, tau=10e-6, PRI=50e-6, fc=10e9, n_pulses=256, noise=1.0, snr=20, rcs_variation=False, device='cpu'):
        self.fs = fs  # Sampling frequency
        self.tau = tau  # Pulse duration 
        self.BW = BW    # Bandwidth
        self.fc = fc  # Carrier frequency
        self.noise = noise
        self.snr=snr
        self.rcs_variation=rcs_variation

        self.PRI = PRI  # Pulse Repetition Interval 
        self.n_pulses = n_pulses  # Number of pulses
        self.T = self.n_pulses * self.PRI  # Total measurement duration

        self.device=device

        self.pulse_compression_gain = self.tau * self.BW * self.n_pulses # Pulse compression gain
        self.Pspec = 10**(self.snr / 10)  # Reference power (linear scale)
        self.Rspec = 750 # Reference range (m)
        
        # Time steps for one pulse
        self.t = torch.linspace(0, self.PRI, int(self.fs*self.PRI), dtype=torch.float32, device=self.device)        
    
    def s_tx(self, t_delay=0):
        t = self.t - t_delay
        k = self.BW/self.tau
        s = torch.exp(1j * torch.pi * k * torch.pow(t-self.tau/2,2))
        s[(t < 0) | (t >= self.tau)] = 0
        return s

    def rx_signals(self, r_list, v_list):
        # r_list and v_list are lists of lists: one list per target
        RX_signals = torch.zeros((len(r_list), self.n_pulses, self.t.size(0)), dtype=torch.complex64, device=self.device)

        for target_idx in range(len(r_list)):
            if self.rcs_variation:
                RCS_target = np.random.uniform(0.25, 1)  # m^2
                Power = self.Pspec / self.pulse_compression_gain * RCS_target
            else:
                Power = self.Pspec / self.pulse_compression_gain

            r = r_list[target_idx]
            v = v_list[target_idx]
            
            # Calculate mean target range for scaling (or use first pulse)
            # Rtarget = r[0]
            # Power = (self.Rspec**4 / (np.array(Rtarget)**4)) * self.Pspec / self.pulse_compression_gain * RCS_target/self.RCS_ref
            for pulse_idx in range(self.n_pulses):
                t_pulse = pulse_idx * self.PRI

                doppler_freq = 2 * v * self.fc / c0
                delay_time = 2 * r / c0

                RX_signal = self.s_tx(t_delay=delay_time)
                doppler_phase = torch.exp(1j * 2 * torch.pi * doppler_freq * torch.tensor(t_pulse, dtype=torch.float32, device=self.device))

                RX_signals[target_idx, pulse_idx, :] = RX_signal *doppler_phase * np.sqrt(Power)

        # Sum over targets to get a 2D matrix
        RX_signals = RX_signals.sum(dim=0)

        noise = torch.exp(1j * 2 * np.pi * torch.rand(RX_signals.size(), device=self.device)) * self.noise * torch.normal(0, 1, RX_signals.size(), device=self.device)

        return RX_signals + noise

    def range_doppler(self, r, v):
        num_samples = int(self.fs * self.tau)
        num_pulses = int(self.n_pulses)

        fast_time_window = torch.hamming_window(num_samples).to(self.device)
        # fast_time_window = fast_time_window/torch.sqrt(torch.sum(fast_time_window)) *np.sqrt(num_samples) # Normalize the window
        slow_time_window = torch.hamming_window(num_pulses, periodic=False).to(self.device)
        # slow_time_window = slow_time_window/torch.sqrt(torch.sum(slow_time_window)) *np.sqrt(num_pulses)

        s_tx = self.s_tx()
        s_tx[:num_samples] *= fast_time_window
        s_rx = self.rx_signals(r, v)

        S_tx = torch.fft.fft(s_tx)
        S_rx = torch.fft.fft(s_rx, dim=1)

        correlated = torch.fft.ifft(S_rx * torch.conj(S_tx)[None, :], dim=1)
        correlated *= slow_time_window[:, None]

        range_doppler = torch.fft.fftshift(torch.fft.fft(correlated, dim=0), dim=0)

        return range_doppler
    

def generate_single_burst(device, num_targets, shape=(64, 512), max_position=[500, 27.5]):
    radar = PulsedRadar(noise=1, rcs_variation=True, snr=20, device=device)

    state_vectors = [
        [np.random.randint(-max_position[0], max_position[0]),
         np.random.randint(-max_position[1], max_position[1]),
         np.random.randint(-max_position[0], max_position[0]),
         np.random.randint(-max_position[1], max_position[1])]
        for _ in range(num_targets)
    ]

    r_list = []
    v_list = []

    for target in state_vectors:
        x, vx, y, vy = target[0], target[1], target[2], target[3]
        r = np.sqrt(x**2 + y**2)
        v_radial = (vx * x + vy * y) / r if r > 0 else 0

        r_list.append(r)
        v_list.append(v_radial)

    rd_map = radar.range_doppler(r_list, v_list)
    rd_map = rd_map[INDICES, :shape[1]]  # shape: (64, 512)

    real = rd_map.real
    imag = rd_map.imag

    magnitude = 20 * torch.log10(torch.sqrt(real**2 + imag**2) + 1e-10)
    magnitude = (magnitude - magnitude.mean()) / magnitude.std()

    return magnitude.unsqueeze(0), num_targets, torch.tensor(r_list), torch.tensor(v_list)  # image: (1, 64, 512), ranges: (num_targets), velocities: (num_targets)

def burst_worker(num_targets, device):
    return generate_single_burst(device=device, num_targets=num_targets)

def create_dataset_parallel(n_targets, bursts_per_class, save_path="data/name.pt", device="cpu", num_workers=8):
    all_data = []
    all_labels = []
    all_ranges = []
    all_velocities = []

    total_tasks = (n_targets) * bursts_per_class
    futures = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for num_targets in range(n_targets):
            for _ in range(bursts_per_class):
                futures.append(executor.submit(burst_worker, num_targets, device))

        print(f"🔄 Generating {total_tasks} bursts in parallel...")
        for future in trange(total_tasks, desc="Progress"):
            result = futures[future].result()  # Get completed in order (instead of as_completed)
            # Inside the for future in trange loop
            burst_data, labels, ranges, velocities = result
            all_data.append(burst_data)
            all_labels.append(labels)
            all_ranges.append(ranges)
            all_velocities.append(velocities)

    # Convert to tensors
    data_tensor = torch.stack(all_data)
    label_tensor = torch.tensor(all_labels)

    # Add padding to ranges and velocities with nan
    max_length = max(len(r) for r in all_ranges)
    all_ranges = [torch.cat([r, torch.full((max_length - len(r),), float('nan'))]) for r in all_ranges]
    all_velocities = [torch.cat([v, torch.full((max_length - len(v),), float('nan'))]) for v in all_velocities]

    ranges_tensor = torch.stack(all_ranges)
    velocities_tensor = torch.stack(all_velocities)

    # Wrap in TensorDataset and save
    dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor, ranges_tensor, velocities_tensor)
    torch.save(dataset, save_path)

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate dataset
    n_targets = 6
    bursts_per_class = 5000
    create_dataset_parallel(n_targets, bursts_per_class, save_path='data/20dB_RCS.pt', device=device, num_workers=10)





# class RadarDataGenerator:
#     def __init__(self, radar, target_sim):
#         self.radar = radar
#         self.target_sim = target_sim

#     def generate_data(self, num_targets=2):
#         range_doppler_maps = []
#         ground_truths = []

#         for step in range(self.target_sim.num_steps):
#             r_list = []
#             v_list = []

#             for target_idx in range(num_targets):
#                 state = self.target_sim.truths[target_idx][step]
#                 x, vx, y, vy = state.state_vector

#                 r = np.sqrt(x**2 + y**2)
#                 v_radial = (vx * x + vy * y) / r if r > 0 else 0

#                 r_list.append(r)
#                 v_list.append(v_radial)

#             rd_map = self.radar.range_doppler(r_list, v_list)
#             range_doppler_maps.append(rd_map)

#             step_truth = [[r_list[target_idx], v_list[target_idx]] for target_idx in range(num_targets)]
#             ground_truths.append(step_truth)

#         ground_truths = np.array(ground_truths)

#         return range_doppler_maps, ground_truths


# def generate_tracks(device, n_tracks=5, n_bursts=20, max_position=[500, 27.5], num_targets=2):
#     indices = calculate_resolution(PulsedRadar(), min_dopp=-37, max_dopp=38)

#     for track in range(n_tracks):
#         if track % 50 == 0:
#             print(f"Generating data for track {track}")

#         radar = PulsedRadar(noise=1, device=device)

#         # Generate random starting states for all targets
#         state_vectors = []
#         for _ in range(num_targets):
#             random_state = [
#                 np.random.randint(-max_position[0], max_position[0]), np.random.randint(-max_position[1], max_position[1]),
#                 np.random.randint(-max_position[0], max_position[0]), np.random.randint(-max_position[1], max_position[1])
#             ]
#             state_vectors.append(random_state)

#         # Pass list of states to TargetSimulator
#         target_sim = TargetSimulator(num_steps=n_bursts, state_vectors=state_vectors, randomness=[0.1, 0.1])

#         data_generator = RadarDataGenerator(radar, target_sim)
#         range_doppler_maps, ground_truths = data_generator.generate_data(num_targets=num_targets)

#         for step, rd_map in enumerate(range_doppler_maps):
#             # Slice directly on GPU
#             rd_map = rd_map[indices, :512]  # shape: (64, 512), complex dtype
#             plot_doppler(radar,20*np.log10(np.abs(rd_map.cpu().numpy())))

#         track_dir = f"/nas-tmp/P_Lens/tracks/track_{track}"
#         os.makedirs(track_dir, exist_ok=True)

#         gt_dir = f"/nas-tmp/P_Lens/tracks/ground_truth"
#         os.makedirs(gt_dir, exist_ok=True)

#         gt_df = pd.DataFrame({
#             'range': ground_truths[:, :, 0].reshape(-1),
#             'velocity': ground_truths[:, :, 1].reshape(-1),
#             'target_id': np.repeat(np.arange(num_targets), ground_truths.shape[0])
#         })
#         gt_df.to_parquet(f"{gt_dir}/ground_truths_{track}.parquet")

#         for step, rd_map in enumerate(range_doppler_maps):
#             rd_map_np = rd_map.cpu().numpy() if hasattr(rd_map, 'cpu') else np.array(rd_map)
#             rd_map_np = rd_map_np[indices, :4096]

#             df = pd.DataFrame({
#                 'real': rd_map_np.real.flatten(),
#                 'imag': rd_map_np.imag.flatten(),
#             })

#             pq.write_table(pa.Table.from_pandas(df), f"{track_dir}/burst_{step}.parquet")
