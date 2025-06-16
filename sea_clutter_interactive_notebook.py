import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    import os
    from typing import Tuple, List
    import time
    from tqdm import tqdm

    # Import our custom modules
    import sea_clutter
    import models
    from src.generate_data import generate_segmentation_dataset_with_sequences, generate_single_frame_with_targets, simulate_sequence_with_realistic_targets_and_masks
    from src.simulation import simulate_sequence_with_realistic_targets
    from src.unet_training import train_model, comprehensive_model_analysis, DiceLoss, CombinedLoss, dice_coeff

    print("✅ All modules imported successfully!")
    return (
        comprehensive_model_analysis,
        mo,
        np,
        os,
        plt,
        random,
        sea_clutter,
        simulate_sequence_with_realistic_targets,
        simulate_sequence_with_realistic_targets_and_masks,
        time,
        torch,
        tqdm,
        train_model,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # 🌊 Interactive Sea Clutter Simulation & U-Net Training Notebook

    This notebook provides three main functionalities: \t
    1. **Parameter Adjustment & Real-time Visualization** - Adjust radar and clutter parameters and see sea clutter simulation \t
    2. **Dataset Generation** - Generate synthetic sea clutter datasets for training \t
    3. **U-Net Training** - Train deep learning models on the generated data
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""## 🎛️ **Section 1: Real-time Sea Clutter Visualization**""")
    return


@app.cell
def _(mo):
    mo.md("""### Radar Parameters""")
    return


@app.cell
def _(mo):
    # Radar parameter controls
    radar_prf = mo.ui.slider(start=1000, stop=10000, value=5000, step=100, label="PRF (Hz)")
    radar_n_pulses = mo.ui.slider(start=64, stop=256, value=128, step=64, label="Number of Pulses")
    radar_n_ranges = mo.ui.slider(start=64, stop=256, value=128, step=64, label="Number of Range Bins")
    radar_range_res = mo.ui.slider(start=0.5, stop=5.0, value=1.0, step=0.1, label="Range Resolution (m)")

    mo.hstack([radar_prf, radar_n_pulses, radar_n_ranges, radar_range_res])
    return radar_n_pulses, radar_n_ranges, radar_prf, radar_range_res


@app.cell
def _(mo):
    mo.md("""### Clutter Parameters""")
    return


@app.cell
def _(mo):
    # Clutter parameter controls
    clutter_mean_power = mo.ui.slider(start=-30, stop=30, value=15, step=1, label="Mean Clutter Power (dB)")
    clutter_shape_param = mo.ui.slider(start=0.01, stop=1.0, value=0.1, step=0.01, label="Shape Parameter")
    clutter_ar_coeff = mo.ui.slider(start=0.7, stop=0.99, value=0.85, step=0.01, label="AR Coefficient")
    clutter_bragg_offset = mo.ui.slider(start=0, stop=100, value=45, step=1, label="Bragg Offset (Hz)")
    clutter_bragg_width = mo.ui.slider(start=1, stop=10, value=4, step=0.5, label="Bragg Width (Hz)")
    clutter_bragg_power = mo.ui.slider(start=0, stop=20, value=8, step=0.5, label="Bragg Power (dB)")
    clutter_wave_speed = mo.ui.slider(start=0, stop=15, value=6, step=0.5, label="Wave Speed (m/s)")

    mo.vstack([
        mo.hstack([clutter_mean_power, clutter_shape_param]),
        mo.hstack([clutter_ar_coeff, clutter_bragg_offset]),
        mo.hstack([clutter_bragg_width, clutter_bragg_power, clutter_wave_speed])
    ])
    return (
        clutter_ar_coeff,
        clutter_bragg_offset,
        clutter_bragg_power,
        clutter_bragg_width,
        clutter_mean_power,
        clutter_shape_param,
        clutter_wave_speed,
    )


@app.cell
def _(mo):
    mo.md("""### Target Parameters""")
    return


@app.cell
def _(mo):
    # Target parameter controls
    target_n_targets = mo.ui.slider(start=0, stop=15, value=3, label="Number of Targets")
    target_type_select = mo.ui.dropdown(
        options=["SPEEDBOAT", "CARGO_SHIP"],
        value="SPEEDBOAT",
        label="Target Type"
    )

    mo.hstack([target_n_targets, target_type_select])
    return target_n_targets, target_type_select


@app.cell
def _(mo):
    mo.md("""### Signal & Noise Power (SNR Control)""")
    return


@app.cell
def _(mo):
    # SNR control parameters
    target_signal_power = mo.ui.slider(start=-10, stop=30, value=15, step=1, label="Target Signal Power (dB)")
    noise_power = mo.ui.slider(start=-20, stop=10, value=-5, step=1, label="Noise Power (dB)")

    mo.hstack([target_signal_power, noise_power])
    return noise_power, target_signal_power





@app.cell
def _(mo):
    mo.md("""### Sequence Parameters (for Multi-Frame Generation)""")
    return


@app.cell
def _(mo):
    # Sequence parameter controls
    seq_frame_rate = mo.ui.slider(start=0.5, stop=10.0, value=2.0, step=0.5, label="Frame Rate (Hz)")
    seq_total_time = mo.ui.slider(start=1.0, stop=20.0, value=5.0, step=0.5, label="Total Sequence Time (s)")

    mo.hstack([seq_frame_rate, seq_total_time])
    return seq_frame_rate, seq_total_time


@app.cell
def _(
    clutter_ar_coeff,
    clutter_bragg_offset,
    clutter_bragg_power,
    clutter_bragg_width,
    clutter_mean_power,
    clutter_shape_param,
    clutter_wave_speed,
    mo,
    noise_power,
    np,
    plt,
    radar_n_pulses,
    radar_n_ranges,
    radar_prf,
    radar_range_res,
    random,
    sea_clutter,
    seq_frame_rate,
    seq_total_time,
    simulate_sequence_with_realistic_targets,
    target_n_targets,
    target_signal_power,
    target_type_select,
):
    # Real-time sea clutter visualization
    # Initialize sequence_data as None for all cases
    sequence_data = None

    print("🔄 Generating new simulation...")

    # Create radar and clutter parameters from sliders
    sim_rp = sea_clutter.RadarParams(
        prf=radar_prf.value,
        n_pulses=radar_n_pulses.value,
        n_ranges=radar_n_ranges.value,
        range_resolution=radar_range_res.value
    )

    sim_cp = sea_clutter.ClutterParams(
        mean_power_db=clutter_mean_power.value,
        shape_param=clutter_shape_param.value,
        ar_coeff=clutter_ar_coeff.value,
        bragg_offset_hz=clutter_bragg_offset.value if clutter_bragg_offset.value > 0 else None,
        bragg_width_hz=clutter_bragg_width.value,
        bragg_power_rel=clutter_bragg_power.value,
        wave_speed_mps=clutter_wave_speed.value
    )

    # Create sequence parameters
    sim_sp = sea_clutter.SequenceParams()
    sim_sp.frame_rate_hz = seq_frame_rate.value
    sim_sp.n_frames = max(1, int(seq_total_time.value * seq_frame_rate.value))

    # Determine if we should show single frame or sequence
    show_sequence = sim_sp.n_frames > 1

    if show_sequence:
        # Multi-frame sequence visualization
        print(f"🎬 Generating {sim_sp.n_frames} frame sequence at {sim_sp.frame_rate_hz} Hz")
        
        # Create targets for sequence
        sim_targets = []
        if target_n_targets.value > 0:
            min_range = 10
            max_range = sim_rp.n_ranges - 10
            sim_target_type = getattr(sea_clutter.TargetType, target_type_select.value)

            for sim_target_idx in range(target_n_targets.value):
                sim_tgt = sea_clutter.create_realistic_target(
                    sim_target_type, 
                    random.randint(min_range, max_range), 
                    sim_rp
                )
                sim_targets.append(sim_tgt)

        # Generate sequence using simulation module with SNR controls
        sim_rdm_list = simulate_sequence_with_realistic_targets(
            sim_rp, sim_cp, sim_sp, sim_targets, random_roll=True, 
            thermal_noise_db=noise_power.value, target_signal_power=target_signal_power.value
        )

        # Store sequence data for frame slider
        sequence_data = {
            'rdm_list': sim_rdm_list,
            'rdm_db_list': [20 * np.log10(np.abs(rdm) + 1e-10) for rdm in sim_rdm_list],
            'rp': sim_rp,
            'cp': sim_cp,
            'sp': sim_sp,
            'n_targets': target_n_targets.value,
            'target_type': target_type_select.value,
            'noise_power': noise_power.value,
            'target_signal_power': target_signal_power.value
        }
        
        # Display basic info about generated sequence
        sim_info = mo.md(f"""
        **🎬 Sequence Generated Successfully:**
        - **Frames**: {len(sim_rdm_list)} frames at {sim_sp.frame_rate_hz} Hz  
        - **Duration**: {len(sim_rdm_list)/sim_sp.frame_rate_hz:.1f} seconds
        - **Targets**: {target_n_targets.value} x {target_type_select.value}
        - **SNR**: {target_signal_power.value - noise_power.value:.1f} dB
        
        *Use the interactive frame viewer below to explore the sequence.*
        """)
        
        sequence_display = sim_info

    else:
        # Single frame visualization
        sim_clutter_td, _, _ = sea_clutter.simulate_sea_clutter(sim_rp, sim_cp, thermal_noise_db=noise_power.value)

        # Add targets if specified
        if target_n_targets.value > 0:
            min_range = 10
            max_range = sim_rp.n_ranges - 10
            sim_target_type = getattr(sea_clutter.TargetType, target_type_select.value)

            for sim_target_idx in range(target_n_targets.value):
                # Create realistic target
                sim_tgt = sea_clutter.create_realistic_target(
                    sim_target_type, 
                    random.randint(min_range, max_range), 
                    sim_rp
                )

                # Override target power with user-controlled value
                sim_tgt.power = target_signal_power.value

                # Convert to simple target for adding to clutter
                sim_simple_target = sea_clutter.Target(
                    rng_idx=sim_tgt.rng_idx,
                    doppler_hz=sim_tgt.doppler_hz,
                    power=sim_tgt.power,
                    size=getattr(sim_tgt, 'size', 1)  # Include target size
                )

                # Add target to clutter data
                sea_clutter.add_target_blob(sim_clutter_td, sim_simple_target, sim_rp)

        # Compute range-Doppler map
        sim_rdm = sea_clutter.compute_range_doppler(sim_clutter_td, sim_rp, sim_cp)

        # Convert to dB and display
        sim_rdm_db = 20 * np.log10(np.abs(sim_rdm) + 1e-10)

        # Create the plot
        sim_fig, sim_ax = plt.subplots(figsize=(10, 6))

        # Calculate frequency and range axes
        sim_fd = np.fft.fftshift(np.fft.fftfreq(sim_rp.n_pulses, d=1.0 / sim_rp.prf))
        sim_velocity = sim_fd * sim_rp.carrier_wavelength / 2.0
        sim_range_axis = np.arange(sim_rp.n_ranges) * sim_rp.range_resolution

        # Use adaptive scaling
        sim_vmin = np.min(sim_rdm_db)
        sim_vmax = np.max(sim_rdm_db)

        sim_im = sim_ax.imshow(sim_rdm_db, aspect='auto', cmap='viridis', origin='lower',
                       extent=[sim_velocity.min(), sim_velocity.max(), sim_range_axis.min(), sim_range_axis.max()],
                       vmin=sim_vmin, vmax=sim_vmax)

        sim_ax.set_xlabel('Velocity (m/s)')
        sim_ax.set_ylabel('Range (m)')
        sim_ax.set_title(f'Range-Doppler Map ({target_n_targets.value} targets, Single Frame)')

        # Add colorbar
        sim_cbar = plt.colorbar(sim_im, ax=sim_ax)
        sim_cbar.set_label('Power (dB)')

        plt.tight_layout()

        # Display simulation info
        sim_info = mo.md(f"""
        **🎯 Simulation Parameters:**
        - **Targets**: {target_n_targets.value} x {target_type_select.value}
        - **Clutter Power**: {clutter_mean_power.value} dB
        - **SNR**: {target_signal_power.value - noise_power.value:.1f} dB
        - **Sea Conditions**: AR={clutter_ar_coeff.value}, Wave Speed={clutter_wave_speed.value} m/s
        - **RDM Size**: {sim_rp.n_ranges} x {sim_rp.n_pulses}
        """)

        sequence_display = mo.vstack([sim_fig, sim_info])

    sequence_display
    return (sequence_data,)


@app.cell
def _(mo, sequence_data):
    # Create frame slider for multi-frame sequences
    if sequence_data is not None:
        frame_slider = mo.ui.slider(
            start=0, 
            stop=len(sequence_data['rdm_list'])-1, 
            value=0, 
            step=1, 
            label=f"Frame (0 to {len(sequence_data['rdm_list'])-1})"
        )

        slider_display = mo.vstack([
            mo.md("**🎬 Interactive Frame Navigation:**"),
            frame_slider
        ])
    else:
        frame_slider = None
        slider_display = mo.md("*Generate a multi-frame sequence to use the interactive frame viewer.*")

    return frame_slider, slider_display


@app.cell
def _(frame_slider, mo, np, plt, sequence_data):
    # Display the selected frame from the slider
    if sequence_data is not None and frame_slider is not None:
        # Calculate consistent scaling across all frames  
        vmin = np.min(sequence_data['rdm_db_list'])
        vmax = np.max(sequence_data['rdm_db_list'])

        # Calculate axes
        fd = np.fft.fftshift(np.fft.fftfreq(sequence_data['rp'].n_pulses, d=1.0 / sequence_data['rp'].prf))
        velocity = fd * sequence_data['rp'].carrier_wavelength / 2.0
        range_axis = np.arange(sequence_data['rp'].n_ranges) * sequence_data['rp'].range_resolution

        # Create the plot for the selected frame
        fig, ax = plt.subplots(figsize=(12, 8))

        # Display the frame selected by the slider
        current_frame = frame_slider.value
        im = ax.imshow(sequence_data['rdm_db_list'][current_frame], aspect='auto', cmap='viridis',
                       extent=[velocity[0], velocity[-1], range_axis[-1], range_axis[0]],
                       vmin=vmin, vmax=vmax, interpolation='nearest')

        ax.set_xlabel('Radial Velocity (m/s)')
        ax.set_ylabel('Range (m)')
        ax.set_title(f'Frame {current_frame} - Interactive RD Map Sequence')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        plt.tight_layout()

        # Frame timing and detailed information
        frame_time = current_frame / sequence_data['sp'].frame_rate_hz
        total_time = len(sequence_data['rdm_list']) / sequence_data['sp'].frame_rate_hz

        # Display detailed info
        frame_info = mo.md(f"""
        **📊 Frame Information:**

        **Current Frame**: {current_frame} / {len(sequence_data['rdm_list'])-1}  
        **Time**: {frame_time:.2f}s / {total_time:.2f}s  
        **Targets**: {sequence_data['n_targets']} x {sequence_data['target_type']}  
        **SNR**: {sequence_data['target_signal_power'] - sequence_data['noise_power']:.1f} dB  
        **Frame Rate**: {sequence_data['sp'].frame_rate_hz} Hz  
        **RDM Size**: {sequence_data['rp'].n_ranges} x {sequence_data['rp'].n_pulses}  

        *Adjust the slider above to navigate through frames and observe target movement over time.*
        """)

        frame_display = mo.vstack([fig, frame_info])
    else:
        frame_display = mo.md("*No sequence data available for frame display.*")

    return (frame_display,)


@app.cell
def _(frame_display, mo, slider_display):
    # Display the interactive frame viewer
    mo.vstack([slider_display, frame_display])
    return


@app.cell
def _(mo):
    mo.md("""## 📊 **Section 2: Dataset Generation**""")
    return


@app.cell
def _(mo):
    mo.md("""Configure dataset generation parameters:""")
    return


@app.cell
def _(mo):
    # Dataset generation controls
    dataset_samples_per_class = mo.ui.slider(start=100, stop=2000, value=500, step=50, label="Samples per Class")
    dataset_max_targets = mo.ui.slider(start=5, stop=15, value=10, step=1, label="Maximum Targets")
    dataset_sea_state = mo.ui.slider(start=1, stop=9, value=5, step=2, label="Sea State (WMO)")
    dataset_n_frames = mo.ui.slider(start=1, stop=10, value=3, step=1, label="Frames per Sequence")

    dataset_name = mo.ui.text(value="my_sea_clutter_dataset", label="Dataset Name")

    mo.vstack([
        mo.hstack([dataset_samples_per_class, dataset_max_targets]),
        mo.hstack([dataset_sea_state, dataset_n_frames]),
        dataset_name
    ])
    return (
        dataset_max_targets,
        dataset_n_frames,
        dataset_name,
        dataset_samples_per_class,
    )


@app.cell
def _(mo):
    # Dataset type selection
    dataset_type = mo.ui.radio(
        options=["single_frame", "multi_frame"],
        value="single_frame",
        label="Dataset Type"
    )

    generate_data_button = mo.ui.button(label="🚀 Generate Dataset", kind="success")

    mo.hstack([dataset_type, generate_data_button])
    return dataset_type, generate_data_button


@app.cell
def _(
    clutter_ar_coeff,
    clutter_bragg_offset,
    clutter_bragg_power,
    clutter_bragg_width,
    clutter_mean_power,
    clutter_shape_param,
    clutter_wave_speed,
    dataset_max_targets,
    dataset_n_frames,
    dataset_name,
    dataset_samples_per_class,
    dataset_type,
    generate_data_button,
    mo,
    noise_power,
    np,
    os,
    plt,
    radar_n_pulses,
    radar_n_ranges,
    radar_prf,
    radar_range_res,
    random,
    sea_clutter,
    simulate_sequence_with_realistic_targets_and_masks,
    target_signal_power,
    target_type_select,
    time,
    torch,
    tqdm,
):
    # Dataset generation logic
    dataset_result = None
    if generate_data_button.value:
        print("🔄 Starting dataset generation...")
        gen_start_time = time.time()

        gen_save_path = f"data/{dataset_name.value}.pt"

        if dataset_type.value == "single_frame":
            print("📸 Generating single-frame dataset...")

            # Use parameters from visualization section (user-configured)
            gen_rp = sea_clutter.RadarParams(
                prf=radar_prf.value,
                n_pulses=radar_n_pulses.value,
                n_ranges=radar_n_ranges.value,
                range_resolution=radar_range_res.value
            )

            gen_cp = sea_clutter.ClutterParams(
                mean_power_db=clutter_mean_power.value,
                shape_param=clutter_shape_param.value,
                ar_coeff=clutter_ar_coeff.value,
                bragg_offset_hz=clutter_bragg_offset.value if clutter_bragg_offset.value > 0 else None,
                bragg_width_hz=clutter_bragg_width.value,
                bragg_power_rel=clutter_bragg_power.value,
                wave_speed_mps=clutter_wave_speed.value
            )

            print(f"📊 Using User-Configured Parameters:")
            print(f"  • Samples per class: {dataset_samples_per_class.value}")
            print(f"  • Max targets: {dataset_max_targets.value}")
            print(f"  • Target type: {target_type_select.value}")
            print(f"  • RDM size: {gen_rp.n_ranges} x {gen_rp.n_pulses}")
            print(f"  • PRF: {gen_rp.prf} Hz")
            print(f"  • Range resolution: {gen_rp.range_resolution} m")
            print(f"  • Clutter power: {gen_cp.mean_power_db} dB")
            print(f"  • Sea conditions: AR={gen_cp.ar_coeff}, Wave Speed={gen_cp.wave_speed_mps} m/s")

            # Storage for data and labels
            gen_all_images = []
            gen_all_labels = []

            # Get target type from user selection
            gen_target_type = getattr(sea_clutter.TargetType, target_type_select.value)

            # Generate data for each class
            for n_targets in range(dataset_max_targets.value + 1):
                print(f"🎯 Generating {dataset_samples_per_class.value} samples for {n_targets} targets...")

                for gen_sample_idx in tqdm(range(dataset_samples_per_class.value), desc=f"Class {n_targets}"):
                    # Generate sea clutter with user-controlled noise power
                    gen_clutter_td, _, _ = sea_clutter.simulate_sea_clutter(gen_rp, gen_cp, thermal_noise_db=noise_power.value)

                    # Add targets if needed using user-selected target type
                    if n_targets > 0:
                        for _ in range(n_targets):
                            gen_target = sea_clutter.create_realistic_target(
                                gen_target_type, 
                                random.randint(30, gen_rp.n_ranges - 30), 
                                gen_rp
                            )

                            # Override target power with user-controlled value
                            gen_target.power = target_signal_power.value

                            # Convert to simple target for adding to clutter
                            gen_simple_target = sea_clutter.Target(
                                rng_idx=gen_target.rng_idx,
                                doppler_hz=gen_target.doppler_hz,
                                power=gen_target.power,
                                size=getattr(gen_target, 'size', 1)  # Include target size
                            )

                            # Add target to clutter data
                            sea_clutter.add_target_blob(gen_clutter_td, gen_simple_target, gen_rp)

                    # Compute range-Doppler map
                    gen_rdm = sea_clutter.compute_range_doppler(gen_clutter_td, gen_rp, gen_cp)

                    # Convert to dB scale and normalize
                    gen_rdm_db = 20 * np.log10(np.abs(gen_rdm) + 1e-10)
                    gen_rdm_normalized = (gen_rdm_db - np.mean(gen_rdm_db)) / (np.std(gen_rdm_db) + 1e-10)

                    # Store image and label
                    gen_all_images.append(gen_rdm_normalized)
                    gen_all_labels.append(n_targets)

            # Convert to numpy arrays and then to tensors
            gen_images = np.array(gen_all_images)
            gen_labels = np.array(gen_all_labels)

            # Convert to PyTorch tensors
            gen_images_tensor = torch.from_numpy(gen_images).float()
            gen_labels_tensor = torch.from_numpy(gen_labels).long()

            # Create dataset dictionary
            dataset_result = {
                'images': gen_images_tensor,
                'labels': gen_labels_tensor,
                'metadata': {
                    'samples_per_class': dataset_samples_per_class.value,
                    'max_targets': dataset_max_targets.value,
                    'target_type': target_type_select.value,
                    'n_ranges': gen_rp.n_ranges,
                    'n_doppler_bins': gen_rp.n_pulses,
                    'range_resolution': gen_rp.range_resolution,
                    'prf': gen_rp.prf,
                    'clutter_params': {
                        'mean_power_db': gen_cp.mean_power_db,
                        'shape_param': gen_cp.shape_param,
                        'ar_coeff': gen_cp.ar_coeff,
                        'bragg_offset_hz': gen_cp.bragg_offset_hz,
                        'bragg_width_hz': gen_cp.bragg_width_hz,
                        'bragg_power_rel': gen_cp.bragg_power_rel,
                        'wave_speed_mps': gen_cp.wave_speed_mps
                    },
                    'snr_params': {
                        'target_signal_power_db': target_signal_power.value,
                        'noise_power_db': noise_power.value,
                        'snr_db': target_signal_power.value - noise_power.value
                    },
                    'class_names': [f"{targets_idx}_targets" for targets_idx in range(dataset_max_targets.value + 1)],
                    'dataset_type': 'single_frame_classification'
                }
            }

        else:  # multi_frame
            print("🎬 Generating multi-frame segmentation dataset...")

            # Use parameters from visualization section (user-configured)
            gen_rp = sea_clutter.RadarParams(
                prf=radar_prf.value,
                n_pulses=radar_n_pulses.value,
                n_ranges=radar_n_ranges.value,
                range_resolution=radar_range_res.value
            )

            gen_cp = sea_clutter.ClutterParams(
                mean_power_db=clutter_mean_power.value,
                shape_param=clutter_shape_param.value,
                ar_coeff=clutter_ar_coeff.value,
                bragg_offset_hz=clutter_bragg_offset.value if clutter_bragg_offset.value > 0 else None,
                bragg_width_hz=clutter_bragg_width.value,
                bragg_power_rel=clutter_bragg_power.value,
                wave_speed_mps=clutter_wave_speed.value
            )

            # Set sequence parameters
            gen_sp = sea_clutter.SequenceParams()
            gen_sp.n_frames = dataset_n_frames.value

            print(f"📊 Using User-Configured Parameters:")
            print(f"  • Samples per class: {dataset_samples_per_class.value}")
            print(f"  • Max targets: {dataset_max_targets.value}")
            print(f"  • Target type: {target_type_select.value}")
            print(f"  • Frames per sequence: {dataset_n_frames.value}")
            print(f"  • RDM size: {gen_rp.n_ranges} x {gen_rp.n_pulses}")
            print(f"  • PRF: {gen_rp.prf} Hz")
            print(f"  • Range resolution: {gen_rp.range_resolution} m")
            print(f"  • Clutter power: {gen_cp.mean_power_db} dB")
            print(f"  • Sea conditions: AR={gen_cp.ar_coeff}, Wave Speed={gen_cp.wave_speed_mps} m/s")

            # Generate multi-frame dataset with custom parameters
            gen_all_sequences = []
            gen_all_mask_sequences = []
            gen_all_labels = []

            # Get target type from user selection
            gen_target_type = getattr(sea_clutter.TargetType, target_type_select.value)

            # Generate data for each class
            for n_targets in range(dataset_max_targets.value + 1):
                print(f"🎯 Generating {dataset_samples_per_class.value} sequences for {n_targets} targets...")

                for gen_sample_idx in tqdm(range(dataset_samples_per_class.value), desc=f"Class {n_targets}"):
                    # Generate targets for this sequence
                    gen_targets = []
                    if n_targets > 0:
                        for _ in range(n_targets):
                            gen_target = sea_clutter.create_realistic_target(
                                gen_target_type, 
                                random.randint(30, gen_rp.n_ranges - 30), 
                                gen_rp
                            )
                            gen_targets.append(gen_target)

                    # Generate sequence of RDMs and masks with SNR controls
                    gen_rdm_list, gen_mask_list = simulate_sequence_with_realistic_targets_and_masks(
                        gen_rp, gen_cp, gen_sp, gen_targets, 
                        thermal_noise_db=noise_power.value, target_signal_power=target_signal_power.value
                    )

                    # Process each frame in the sequence
                    gen_processed_sequence = []
                    gen_processed_mask_sequence = []

                    for gen_rdm, gen_mask in zip(gen_rdm_list, gen_mask_list):
                        # Convert to dB scale and normalize
                        gen_rdm_db = 20 * np.log10(np.abs(gen_rdm) + 1e-10)
                        gen_rdm_normalized = (gen_rdm_db - np.mean(gen_rdm_db)) / (np.std(gen_rdm_db) + 1e-10)
                        gen_processed_sequence.append(gen_rdm_normalized)
                        gen_processed_mask_sequence.append(gen_mask)

                    # Stack frames into sequence arrays
                    gen_sequence = np.stack(gen_processed_sequence, axis=0)
                    gen_mask_sequence = np.stack(gen_processed_mask_sequence, axis=0)

                    # Store sequence and label
                    gen_all_sequences.append(gen_sequence)
                    gen_all_mask_sequences.append(gen_mask_sequence)
                    gen_all_labels.append(n_targets)

            # Convert to numpy arrays
            gen_sequences = np.array(gen_all_sequences)
            gen_mask_sequences = np.array(gen_all_mask_sequences)
            gen_labels = np.array(gen_all_labels)

            # Convert to PyTorch tensors
            gen_sequences_tensor = torch.from_numpy(gen_sequences).float()
            gen_mask_sequences_tensor = torch.from_numpy(gen_mask_sequences).float()
            gen_labels_tensor = torch.from_numpy(gen_labels).long()

            # Create dataset dictionary
            dataset_result = {
                'sequences': gen_sequences_tensor,
                'mask_sequences': gen_mask_sequences_tensor,
                'labels': gen_labels_tensor,
                'metadata': {
                    'samples_per_class': dataset_samples_per_class.value,
                    'max_targets': dataset_max_targets.value,
                    'target_type': target_type_select.value,
                    'n_frames': dataset_n_frames.value,
                    'n_ranges': gen_rp.n_ranges,
                    'n_doppler_bins': gen_rp.n_pulses,
                    'range_resolution': gen_rp.range_resolution,
                    'prf': gen_rp.prf,
                    'clutter_params': {
                        'mean_power_db': gen_cp.mean_power_db,
                        'shape_param': gen_cp.shape_param,
                        'ar_coeff': gen_cp.ar_coeff,
                        'bragg_offset_hz': gen_cp.bragg_offset_hz,
                        'bragg_width_hz': gen_cp.bragg_width_hz,
                        'bragg_power_rel': gen_cp.bragg_power_rel,
                        'wave_speed_mps': gen_cp.wave_speed_mps
                    },
                    'snr_params': {
                        'target_signal_power_db': target_signal_power.value,
                        'noise_power_db': noise_power.value,
                        'snr_db': target_signal_power.value - noise_power.value
                    },
                    'class_names': [f"{targets_idx}_targets" for targets_idx in range(dataset_max_targets.value + 1)],
                    'dataset_type': 'sequence_segmentation'
                }
            }

        # Save dataset (if single frame) or load info (if multi frame)
        if dataset_type.value == "single_frame":
            torch.save(dataset_result, gen_save_path)

        gen_end_time = time.time()
        gen_generation_time = gen_end_time - gen_start_time

        print(f"✅ Dataset generation completed in {gen_generation_time:.1f} seconds!")
        print(f"📁 Dataset saved to: {gen_save_path}")

        if dataset_type.value == "single_frame":
            gen_file_size = torch.load(gen_save_path)['images'].element_size() * torch.load(gen_save_path)['images'].nelement() / (1024**2)
            print(f"💾 File size: {gen_file_size:.1f} MB")
            print(f"📈 Dataset shape: {dataset_result['images'].shape}")
        else:
            gen_file_size = os.path.getsize(gen_save_path) / (1024**2)
            print(f"💾 File size: {gen_file_size:.1f} MB")
            print(f"📈 Sequences shape: {dataset_result['sequences'].shape}")
            print(f"📈 Masks shape: {dataset_result['mask_sequences'].shape}")

        # Show sample
        if dataset_type.value == "single_frame":
            gen_sample_idx = random.randint(0, len(dataset_result['images']) - 1)
            gen_sample_img = dataset_result['images'][gen_sample_idx].numpy()
            gen_sample_label = dataset_result['labels'][gen_sample_idx].item()

            gen_sample_fig, gen_sample_ax = plt.subplots(figsize=(8, 6))
            gen_sample_ax.imshow(gen_sample_img, aspect='auto', cmap='viridis', origin='lower')
            gen_sample_ax.set_title(f'Sample RDM (Label: {gen_sample_label} targets)')
            gen_sample_ax.set_xlabel('Doppler Bin')
            gen_sample_ax.set_ylabel('Range Bin')
            gen_cbar = plt.colorbar(gen_sample_ax.images[0], ax=gen_sample_ax, label='Normalized Power (dB)')
            plt.tight_layout()

        else:
            gen_sample_idx = random.randint(0, len(dataset_result['sequences']) - 1)
            gen_sample_seq = dataset_result['sequences'][gen_sample_idx].numpy()
            gen_sample_mask = dataset_result['mask_sequences'][gen_sample_idx].numpy()
            gen_sample_label = dataset_result['labels'][gen_sample_idx].item()

            gen_sample_fig, gen_sample_axes = plt.subplots(2, min(3, gen_sample_seq.shape[0]), figsize=(12, 8))
            if gen_sample_seq.shape[0] == 1:
                gen_sample_axes = gen_sample_axes.reshape(2, 1)

            for gen_vis_idx in range(min(3, gen_sample_seq.shape[0])):
                # Show RDM
                gen_sample_axes[0, gen_vis_idx].imshow(gen_sample_seq[gen_vis_idx], aspect='auto', cmap='viridis', origin='lower')
                gen_sample_axes[0, gen_vis_idx].set_title(f'Frame {gen_vis_idx+1} RDM')
                gen_sample_axes[0, gen_vis_idx].set_xlabel('Doppler Bin')
                gen_sample_axes[0, gen_vis_idx].set_ylabel('Range Bin')

                # Show mask
                gen_sample_axes[1, gen_vis_idx].imshow(gen_sample_mask[gen_vis_idx], aspect='auto', cmap='Reds', origin='lower', vmin=0, vmax=1)
                gen_sample_axes[1, gen_vis_idx].set_title(f'Frame {gen_vis_idx+1} Mask')
                gen_sample_axes[1, gen_vis_idx].set_xlabel('Doppler Bin')
                gen_sample_axes[1, gen_vis_idx].set_ylabel('Range Bin')

            plt.suptitle(f'Sample Sequence (Label: {gen_sample_label} targets)')
            plt.tight_layout()

        dataset_sample_display = gen_sample_fig
    else:
        dataset_sample_display = mo.md("Click 'Generate Dataset' to see a sample.")

    dataset_sample_display
    return


@app.cell
def _(mo):
    mo.md("""## 🤖 **Section 3: U-Net Training**""")
    return


@app.cell
def _(mo):
    mo.md("""Configure U-Net model and training parameters:""")
    return


@app.cell
def _(mo):
    # Model architecture controls - only UNet now
    train_n_channels = mo.ui.slider(start=1, stop=10, value=1, step=1, label="Input Channels")
    train_base_filters = mo.ui.slider(start=8, stop=64, value=16, step=8, label="Base Filters")

    mo.hstack([train_n_channels, train_base_filters])
    return train_base_filters, train_n_channels


@app.cell
def _(mo):
    # Training parameters
    train_epochs = mo.ui.slider(start=10, stop=100, value=30, step=5, label="Epochs")
    train_batch_size = mo.ui.slider(start=8, stop=64, value=16, step=8, label="Batch Size")
    train_learning_rate = mo.ui.slider(start=1e-5, stop=1e-2, value=1e-4, step=1e-5, label="Learning Rate")
    train_patience = mo.ui.slider(start=5, stop=20, value=10, step=1, label="Early Stopping Patience")

    mo.vstack([
        mo.hstack([train_epochs, train_batch_size]),
        mo.hstack([train_learning_rate, train_patience])
    ])
    return train_batch_size, train_epochs, train_learning_rate, train_patience


@app.cell
def _(mo):
    # Loss function parameters
    train_bce_weight = mo.ui.slider(start=0.1, stop=2.0, value=1.0, step=0.1, label="BCE Weight")
    train_dice_weight = mo.ui.slider(start=0.1, stop=2.0, value=1.0, step=0.1, label="Dice Weight")

    mo.hstack([train_bce_weight, train_dice_weight])
    return train_bce_weight, train_dice_weight


@app.cell
def _(mo, os):
    # Dataset selection for training
    train_available_datasets = ["Select dataset..."] + [f for f in os.listdir("data") if f.endswith('.pt')]
    train_dataset_dropdown = mo.ui.dropdown(
        options=train_available_datasets,
        value=train_available_datasets[0] if train_available_datasets else "No datasets found",
        label="Training Dataset"
    )

    train_model_name = mo.ui.text(value="my_trained_model", label="Model Save Name")

    train_start_button = mo.ui.button(label="🚀 Start Training", kind="success")

    mo.vstack([
        train_dataset_dropdown,
        train_model_name,
        train_start_button
    ])
    return train_dataset_dropdown, train_model_name, train_start_button


@app.cell
def _(
    comprehensive_model_analysis,
    mo,
    time,
    torch,
    train_base_filters,
    train_batch_size,
    train_bce_weight,
    train_dataset_dropdown,
    train_dice_weight,
    train_epochs,
    train_learning_rate,
    train_model,
    train_model_name,
    train_n_channels,
    train_patience,
    train_start_button,
):
    # Training execution
    training_output = None
    if train_start_button.value and train_dataset_dropdown.value != "Select dataset...":
        print("🚀 Starting model training...")
        training_start_time = time.time()

        training_dataset_path = f"data/{train_dataset_dropdown.value}"
        training_model_save_path = f"pretrained/{train_model_name.value}.pt"

        print(f"📊 Training Configuration:")
        print(f"  • Model: U-Net")
        print(f"  • Dataset: {train_dataset_dropdown.value}")
        print(f"  • Input Channels: {train_n_channels.value}")
        print(f"  • Base Filters: {train_base_filters.value}")
        print(f"  • Epochs: {train_epochs.value}")
        print(f"  • Batch Size: {train_batch_size.value}")
        print(f"  • Learning Rate: {train_learning_rate.value}")
        print(f"  • Loss Weights: BCE={train_bce_weight.value}, Dice={train_dice_weight.value}")

        try:
            # Check if dataset is for segmentation or classification
            training_dataset_check = torch.load(training_dataset_path)
            training_is_segmentation = 'mask_sequences' in training_dataset_check or 'masks' in training_dataset_check

            if training_is_segmentation:
                print("🎯 Detected segmentation dataset - proceeding with U-Net training")

                # Train the model
                train_model(
                    dataset_path=training_dataset_path,
                    n_channels=train_n_channels.value,
                    num_epochs=train_epochs.value,
                    patience=train_patience.value,
                    batch_size=train_batch_size.value,
                    lr=train_learning_rate.value,
                    model_save_path=training_model_save_path,
                    bce_weight=train_bce_weight.value,
                    dice_weight=train_dice_weight.value,
                    base_filters=train_base_filters.value
                )

                training_end_time = time.time()
                training_time = training_end_time - training_start_time

                print(f"✅ Training completed in {training_time/60:.1f} minutes!")
                print(f"💾 Model saved to: {training_model_save_path}")

                # Run comprehensive analysis
                print("📈 Running model analysis...")
                training_analysis_start = time.time()

                training_save_dir = 'single_frame' if train_n_channels.value == 1 else 'multi_frame'
                training_analysis_results = comprehensive_model_analysis(
                    model_path=training_model_save_path,
                    dataset_path=training_dataset_path,
                    n_channels=train_n_channels.value,
                    save_dir=training_save_dir,
                    base_filters=train_base_filters.value
                )

                training_analysis_time = time.time() - training_analysis_start
                print(f"📊 Analysis completed in {training_analysis_time:.1f} seconds!")
                print(f"📁 Analysis results saved to: analysis_{training_save_dir}/")

                # Display training summary
                training_output = mo.md(f"""
                ## 🎉 **Training Summary:**
                - **Total Time**: {training_time/60:.1f} minutes
                - **Model Type**: U-Net with {train_base_filters.value} base filters
                - **Final Model**: `{training_model_save_path}`
                - **Analysis**: `analysis_{training_save_dir}/`
                - **Training completed successfully!** ✅
                """)

            else:
                training_output = mo.md("""
                ❌ **Error**: Dataset appears to be for classification, not segmentation.

                Please generate a multi-frame segmentation dataset first.
                """)

        except Exception as e:
            training_output = mo.md(f"""
            ❌ **Training failed with error**: 
            ```
            {str(e)}
            ```
            Please check your dataset and parameters.
            """)

    elif train_start_button.value:
        training_output = mo.md("⚠️ **Please select a dataset first!**")
    else:
        training_output = mo.md("Configure parameters above and click '🚀 Start Training' to begin.")

    training_output
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 📋 **Quick Reference**

    ### Parameter Guidelines:

    **Sea States (WMO):**
    - 1: Calm seas (0-0.1m waves)
    - 3: Slight seas (0.5-1.25m waves)  
    - 5: Moderate seas (2.5-4m waves)
    - 7: Rough seas (4-6m waves)
    - 9: Very rough seas (7-9m waves)

    **Dataset Types:**
    - **Single Frame**: For classification tasks (counting targets)
    - **Multi Frame**: For segmentation tasks (localizing targets)

    ### Recommended Settings:
    - **Learning Rate**: 1e-4 for most cases
    - **Batch Size**: 16-32 depending on memory
    - **Loss Weights**: BCE=1.0, Dice=1.0 for balanced training
    - **Patience**: 10-15 epochs for early stopping
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 🎯 **Usage Tips**

    1. **Start with visualization**: Adjust parameters in Section 1 to understand how they affect sea clutter
    2. **Generate small datasets first**: Use fewer samples (100-200) for quick testing
    3. **Monitor training**: Watch the console output for training progress and metrics
    4. **Experiment with parameters**: Try different model architectures and hyperparameters
    5. **Check results**: Analysis plots are saved automatically after training

    **Performance Notes:**
    - Training time scales with dataset size and model complexity
    - Larger datasets generally give better performance
    - Multi-frame models can capture temporal information for better target detection
    """
    )
    return


if __name__ == "__main__":
    app.run()
