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

    def get_range_velocity(self):
        """Return per-target ranges and velocities arrays."""
        all_ranges = []
        all_velocities = []

        for truth in self.truths:
            ranges = []
            velocities = []
            for state in truth:
                x, vx, y, vy = state.state_vector
                r = np.sqrt(x**2 + y**2)
                v_radial = (vx * x + vy * y) / r if r > 0 else 0
                ranges.append(r)
                velocities.append(v_radial)
            all_ranges.append(np.array(ranges))
            all_velocities.append(np.array(velocities))

        return all_ranges, all_velocities

class PulsedRadar:
    # Voor een range resolution van 1 meter, moet de bandbreedte 150 MHz zijn (dus de sampling rate 150*2 MHz). 
    # Voor een velocity resolution van 0.5 m/s, moet de pulse repetition interval 66.7 microseconds zijn.
    def __init__(self, BW=50e6, fs=100e6, tau=10e-6, PRI=50e-6, fc=10e9, n_pulses=256, noise=1.0, device='cpu'):
        self.fs = fs  # Sampling frequency
        self.tau = tau  # Pulse duration 
        self.BW = BW    # Bandwidth
        self.fc = fc  # Carrier frequency
        self.noise = noise

        self.PRI = PRI  # Pulse Repetition Interval 
        self.n_pulses = n_pulses  # Number of pulses
        self.T = self.n_pulses * self.PRI  # Total measurement duration

        self.device=device

        self.pulse_compression_gain = self.tau * self.BW * self.n_pulses # Pulse compression gain
        self.Pspec = 10**(40 / 10)  # Reference power (linear scale)
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
        num_targets = len(r_list)
        RX_signals = torch.zeros((num_targets, self.n_pulses, self.t.size(0)), dtype=torch.complex64, device=self.device)

        for target_idx in range(num_targets):
            RCS_target = np.random.uniform(0.1, 1)  # m^2
            r = r_list[target_idx]
            v = v_list[target_idx]
            
            # Calculate mean target range for scaling (or use first pulse)
            # Rtarget = r[0]
            # Power = (self.Rspec**4 / (np.array(Rtarget)**4)) * self.Pspec / self.pulse_compression_gain * RCS_target/self.RCS_ref
            Power = self.Pspec / self.pulse_compression_gain * RCS_target
            for pulse_idx in range(self.n_pulses):
                t_pulse = pulse_idx * self.PRI

                doppler_freq = 2 * v[pulse_idx] * self.fc / c0
                delay_time = 2 * r[pulse_idx] / c0

                RX_signal = self.s_tx(t_delay=delay_time)
                doppler_phase = torch.exp(1j * 2 * torch.pi * doppler_freq * torch.tensor(t_pulse, dtype=torch.float32, device=self.device))

                RX_signals[target_idx, pulse_idx, :] = RX_signal *doppler_phase * np.sqrt(Power)

        # Sum over targets to get a 2D matrix
        RX_signals = RX_signals.sum(dim=0)

        noise = torch.exp(1j * 2 * np.pi * torch.rand(RX_signals.size(), device=self.device)) * self.noise * torch.normal(0, 1, RX_signals.size(), device=self.device)

        return RX_signals + noise


    def range_doppler(self, range_list, velocity_list):
        num_samples = int(self.fs * self.tau)
        num_pulses = int(self.n_pulses)

        fast_time_window = torch.hamming_window(num_samples).to(self.device)
        # fast_time_window = fast_time_window/torch.sqrt(torch.sum(fast_time_window)) *np.sqrt(num_samples) # Normalize the window
        slow_time_window = torch.hamming_window(num_pulses, periodic=False).to(self.device)
        # slow_time_window = slow_time_window/torch.sqrt(torch.sum(slow_time_window)) *np.sqrt(num_pulses)

        s_tx = self.s_tx()
        s_tx[:num_samples] *= fast_time_window
        s_rx = self.rx_signals(range_list, velocity_list)

        S_tx = torch.fft.fft(s_tx)
        S_rx = torch.fft.fft(s_rx, dim=1)

        correlated = torch.fft.ifft(S_rx * torch.conj(S_tx)[None, :], dim=1)

        correlated *= slow_time_window[:, None]

        range_doppler = torch.fft.fftshift(torch.fft.fft(correlated, dim=0), dim=0)

        return range_doppler
    
class RadarDataGenerator:
    def __init__(self, radar, target_sim):
        self.radar = radar
        self.target_sim = target_sim
        self.burst_duration = self.radar.n_pulses * self.radar.tau

    def interpolate_per_pulse(self, state1, state2, n_pulses):
        # Linearly interpolate between two consecutive states
        x1, vx1, y1, vy1 = state1.state_vector
        x2, vx2, y2, vy2 = state2.state_vector - ((state2.state_vector - state1.state_vector) * (1 - self.radar.T))

        interp_ranges = []
        interp_velocities = []

        for i in range(n_pulses):
            alpha = i / n_pulses
            x = (1 - alpha) * x1 + alpha * x2
            y = (1 - alpha) * y1 + alpha * y2
            vx = (1 - alpha) * vx1 + alpha * vx2
            vy = (1 - alpha) * vy1 + alpha * vy2

            r = np.sqrt(x**2 + y**2)
            v_radial = (vx * x + vy * y) / r if r > 0 else 0
            interp_ranges.append(r)
            interp_velocities.append(v_radial)

        return interp_ranges, interp_velocities

    def generate_data(self, num_targets=2):
        range_doppler_maps = []
        ground_truths = []

        for step in range(self.target_sim.num_steps):
            r_list = []
            v_list = []

            for target_idx in range(num_targets):
                state1 = self.target_sim.truths[target_idx][step]
                state2 = self.target_sim.truths[target_idx][step+1]

                r_array, v_array = self.interpolate_per_pulse(state1, state2, self.radar.n_pulses)

                r_list.append(r_array)
                v_list.append(v_array)

            rd_map = self.radar.range_doppler(r_list, v_list)

            range_doppler_maps.append(rd_map)

            step_truth = [[r_list[target_idx][0], v_list[target_idx][0]] for target_idx in range(num_targets)]
            ground_truths.append(step_truth)

        ground_truths = np.array(ground_truths)

        return range_doppler_maps, ground_truths


def generate_tracks(device, n_tracks=5, n_bursts=20, max_position=[500, 27.5], num_targets=2):
    indices = calculate_resolution(PulsedRadar(), min_dopp=-37, max_dopp=38)

    for track in range(n_tracks):
        if track % 50 == 0:
            print(f"Generating data for track {track}")

        radar = PulsedRadar(noise=1, device=device)

        # Generate random starting states for all targets
        state_vectors = []
        for _ in range(num_targets):
            random_state = [
                np.random.randint(-max_position[0], max_position[0]), np.random.randint(-max_position[1], max_position[1]),
                np.random.randint(-max_position[0], max_position[0]), np.random.randint(-max_position[1], max_position[1])
            ]
            state_vectors.append(random_state)

        # Pass list of states to TargetSimulator
        target_sim = TargetSimulator(num_steps=n_bursts, state_vectors=state_vectors, randomness=[0.1, 0.1])

        data_generator = RadarDataGenerator(radar, target_sim)
        range_doppler_maps, ground_truths = data_generator.generate_data(num_targets=num_targets)

        for step, rd_map in enumerate(range_doppler_maps):
            # Slice directly on GPU
            rd_map = rd_map[indices, :512]  # shape: (64, 512), complex dtype
            plot_doppler(radar,20*np.log10(np.abs(rd_map.cpu().numpy())))

        track_dir = f"/nas-tmp/P_Lens/tracks/track_{track}"
        os.makedirs(track_dir, exist_ok=True)

        gt_dir = f"/nas-tmp/P_Lens/tracks/ground_truth"
        os.makedirs(gt_dir, exist_ok=True)

        gt_df = pd.DataFrame({
            'range': ground_truths[:, :, 0].reshape(-1),
            'velocity': ground_truths[:, :, 1].reshape(-1),
            'target_id': np.repeat(np.arange(num_targets), ground_truths.shape[0])
        })
        gt_df.to_parquet(f"{gt_dir}/ground_truths_{track}.parquet")

        for step, rd_map in enumerate(range_doppler_maps):
            rd_map_np = rd_map.cpu().numpy() if hasattr(rd_map, 'cpu') else np.array(rd_map)
            rd_map_np = rd_map_np[indices, :4096]

            df = pd.DataFrame({
                'real': rd_map_np.real.flatten(),
                'imag': rd_map_np.imag.flatten(),
            })

            pq.write_table(pa.Table.from_pandas(df), f"{track_dir}/burst_{step}.parquet")
            # plot_doppler(rd_map_np)


def generate_single_burst(device, num_targets, shape=(64, 512), max_position=[500, 27.5]):
    radar = PulsedRadar(noise=1, device=device)
    indices = calculate_resolution(radar, min_dopp=-37, max_dopp=38)

    # Generate one burst
    state_vectors = [
        [np.random.randint(-max_position[0], max_position[0]),
         np.random.randint(-max_position[1], max_position[1]),
         np.random.randint(-max_position[0], max_position[0]),
         np.random.randint(-max_position[1], max_position[1])]
        for _ in range(num_targets)
    ]

    target_sim = TargetSimulator(num_steps=1, state_vectors=state_vectors, randomness=[0.1, 0.1])
    data_generator = RadarDataGenerator(radar, target_sim)
    range_doppler_maps, _ = data_generator.generate_data(num_targets=num_targets)

    rd_map = range_doppler_maps[0]
    rd_map = rd_map[indices, :512]  # shape: (64, 512)

    real = rd_map.real
    imag = rd_map.imag
    tensor_data = torch.stack((real, imag), dim=-1)

    magnitude = 20 * torch.log10(torch.sqrt(tensor_data[..., 0]**2 + tensor_data[..., 1]**2) + 1e-10)
    magnitude = (magnitude - magnitude.mean()) / magnitude.std()

    return magnitude.cpu().unsqueeze(0), num_targets  # Ensure it's on CPU for saving

def burst_worker(num_targets, device):
    return generate_single_burst(device=device, num_targets=num_targets)

def create_dataset_parallel(n_targets, bursts_per_class, save_path="data/preprocessed_dataset.pt", device="cpu", num_workers=8):
    all_data = []
    all_labels = []

    total_tasks = n_targets * bursts_per_class
    futures = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for num_targets in range(n_targets):
            for _ in range(bursts_per_class):
                futures.append(executor.submit(burst_worker, num_targets, device))

        print(f"🔄 Generating {total_tasks} bursts in parallel...")
        for future in trange(total_tasks, desc="Progress"):
            result = futures[future].result()  # Get completed in order (instead of as_completed)
            burst_data, label = result
            all_data.append(burst_data)
            all_labels.append(label)

    torch.save((all_data, all_labels), save_path)
    print(f"✅ Dataset saved to: {save_path}")



if __name__ == "__main__":
    create_dataset_parallel(5, 500, save_path="data/test.pt")
    # create_dataset(5, 500, save_path="data/test.pt")