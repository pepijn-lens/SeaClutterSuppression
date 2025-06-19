import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    import os
    import time
    import gc
    import psutil

    # Import our custom modules
    import sea_clutter
    import models
    from src.generate_data import simulate_sequence_with_realistic_targets_and_masks
    from src.simulation import simulate_sequence_with_realistic_targets
    from src.unet_training import train_model
    from src.end_to_end_evaluate import comprehensive_evaluation

    # Memory management utility function
    def cleanup_memory(verbose=True):
        """Clean up memory by running garbage collection and clearing GPU cache"""
        if verbose:
            # Get memory usage before cleanup
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        if verbose:
            # Get memory usage after cleanup
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_freed = mem_before - mem_after
            print(f"🧹 Memory cleanup: {mem_before:.1f} MB → {mem_after:.1f} MB (freed {mem_freed:.1f} MB)")
        
        return mem_before, mem_after

    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    print("All modules imported successfully!")
    print(f"Current memory usage: {get_memory_usage():.1f} MB")
    
    return (
        cleanup_memory,
        comprehensive_evaluation,
        gc,
        get_memory_usage,
        mo,
        models,
        np,
        os,
        plt,
        psutil,
        random,
        sea_clutter,
        simulate_sequence_with_realistic_targets,
        simulate_sequence_with_realistic_targets_and_masks,
        time,
        torch,
        train_model,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # 🌊 Interactive Sea Clutter Simulation & U-Net Training Notebook

    This notebook provides four main functionalities: \t
    1. **Parameter Adjustment & Real-time Visualization** - Adjust radar and clutter parameters and see sea clutter simulation \t
    2. **Dataset Generation** - Generate synthetic sea clutter datasets for training \t
    3. **U-Net Training** - Train deep learning models on the generated data\t
    """
    )
    return


@app.cell
def _(cleanup_memory, get_memory_usage, mo):
    # Memory Management Section
    mo.md(f"""
    ## 💾 **Memory Management**
    
    **Current Memory Usage**: {get_memory_usage():.1f} MB
    
    Large datasets and models can consume significant memory. Use the button below to free up RAM when needed.
    """)
    return


@app.cell
def _(cleanup_memory, mo):
    # Manual memory cleanup button
    cleanup_button = mo.ui.button(
        label="🧹 Free Memory", 
        kind="warn",
        value=0,
        on_click=lambda count: count + 1
    )
    
    cleanup_button
    return cleanup_button


@app.cell
def _(cleanup_button, cleanup_memory, mo):
    # Execute memory cleanup when button is pressed
    if cleanup_button.value > 0:
        mem_before, mem_after = cleanup_memory(verbose=True)
        cleanup_result = mo.md(f"""
        ✅ **Memory cleaned up successfully!**\t
        - **Before**: {mem_before:.1f} MB\t
        - **After**: {mem_after:.1f} MB  \t
        - **Freed**: {mem_before - mem_after:.1f} MB
        """)
    else:
        cleanup_result = mo.md("Click the button above to free up memory manually.")
    
    cleanup_result
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
    radar_n_pulses = mo.ui.slider(start=64, stop=512, value=128, step=64, label="Number of Pulses")
    radar_n_ranges = mo.ui.slider(start=64, stop=512, value=128, step=64, label="Number of Range Bins")
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
    noise_power = mo.ui.slider(start=-20, stop=30, value=-5, step=1, label="Noise Power (dB)")

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
            sim_rp, sim_cp, sim_sp, sim_targets, random_roll=False, 
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
        **🎬 Sequence Generated Successfully:**\t
        - **Frames**: {len(sim_rdm_list)} frames at {sim_sp.frame_rate_hz} Hz \t 
        - **Duration**: {len(sim_rdm_list)/sim_sp.frame_rate_hz:.1f} seconds\t
        - **Targets**: {target_n_targets.value} x {target_type_select.value}\t
        - **SNR**: {target_signal_power.value - noise_power.value:.1f} dB\t

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
        sim_vmin = -20
        sim_vmax = 80

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
        vmin = -20
        vmax = 80

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
    # Dataset generation controls
    dataset_samples_per_class = mo.ui.slider(start=100, stop=2000, value=500, step=50, label="Samples per Class")
    dataset_max_targets = mo.ui.slider(start=0, stop=15, value=10, step=1, label="Maximum Targets")
    dataset_n_frames = mo.ui.slider(start=1, stop=10, value=3, step=1, label="Frames per Sequence")

    dataset_name = mo.ui.text(value="my_sea_clutter_dataset", label="Dataset Name")

    mo.vstack([
        mo.hstack([dataset_samples_per_class, dataset_max_targets]),
        mo.hstack([dataset_n_frames]),
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
    # Just the generate button, no dataset type selection
    generate_data_button = mo.ui.button(
        label="🚀 Generate Dataset", 
        kind="success",
        value=0,
        on_click=lambda count: count + 1
    )

    mo.vstack([
        mo.md("**Dataset Type:** Single Frame Segmentation (1 frame) or Multi-Frame Segmentation (>1 frames) - both include masks for target localization"),
        generate_data_button
    ])
    return (generate_data_button,)


@app.cell
def _(
    cleanup_memory,
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
    generate_data_button,
    noise_power,
    np,
    os,
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
):
    # Dataset generation logic - stored as variable instead of file
    generated_dataset = None
    dataset_size_gb = 0.0

    # Create a global variable to track the last generation button value
    import __main__
    if not hasattr(__main__, '_last_generation_button_value'):
        __main__._last_generation_button_value = 0

    # Only generate if button was clicked AND we haven't already generated for this click
    should_generate = (generate_data_button.value > 0 and 
                      generate_data_button.value != __main__._last_generation_button_value)

    if should_generate:
        # Update the tracked value to prevent re-generation
        __main__._last_generation_button_value = generate_data_button.value

        print("🔄 Starting dataset generation...")
        gen_start_time = time.time()

        # Determine dataset type based on number of frames
        is_single_frame = dataset_n_frames.value == 1

        if is_single_frame:
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

            # Set sequence parameters for single frame
            gen_sp = sea_clutter.SequenceParams()
            gen_sp.n_frames = 1

            print(f"📊 Using Parameters from Section 1 Visualization:")
            print(f"  🎯 Dataset Generation:")
            print(f"    • Samples per class: {dataset_samples_per_class.value}")
            print(f"    • Max targets: {dataset_max_targets.value}")
            print(f"    • Target type: {target_type_select.value}")
            print(f"  📡 Radar Configuration:")
            print(f"    • RDM size: {gen_rp.n_ranges} x {gen_rp.n_pulses}")
            print(f"    • PRF: {gen_rp.prf} Hz")
            print(f"    • Range resolution: {gen_rp.range_resolution} m")
            print(f"  🌊 Sea Clutter Configuration:")
            print(f"    • Clutter power: {gen_cp.mean_power_db} dB")
            print(f"    • Shape parameter: {gen_cp.shape_param}")
            print(f"    • AR coefficient: {gen_cp.ar_coeff}")
            print(f"    • Wave speed: {gen_cp.wave_speed_mps} m/s")
            print(f"    • Bragg offset: {gen_cp.bragg_offset_hz} Hz")
            print(f"  📶 Signal/Noise Configuration:")
            print(f"    • Target signal power: {target_signal_power.value} dB")
            print(f"    • Noise power: {noise_power.value} dB")
            print(f"    • SNR: {target_signal_power.value - noise_power.value:.1f} dB")

            # Storage for data, masks and labels
            gen_all_images = []
            gen_all_masks = []
            gen_all_labels = []

            # Get target type from user selection
            gen_target_type = getattr(sea_clutter.TargetType, target_type_select.value)

            # Generate data for each class
            for n_targets in range(dataset_max_targets.value + 1):
                print(f"🎯 Generating {dataset_samples_per_class.value} samples for {n_targets} targets...")

                for gen_sample_idx in range(dataset_samples_per_class.value):
                    # Generate targets for this sample
                    gen_targets = []
                    if n_targets > 0:
                        for _ in range(n_targets):
                            gen_target = sea_clutter.create_realistic_target(
                                gen_target_type, 
                                random.randint(30, gen_rp.n_ranges - 30), 
                                gen_rp
                            )
                            gen_targets.append(gen_target)

                    # Generate single frame RDM and mask with SNR controls
                    gen_rdm_list, gen_mask_list = simulate_sequence_with_realistic_targets_and_masks(
                        gen_rp, gen_cp, gen_sp, gen_targets, 
                        thermal_noise_db=noise_power.value, target_signal_power=target_signal_power.value
                    )

                    # Extract single frame from lists
                    gen_rdm = gen_rdm_list[0]
                    gen_mask = gen_mask_list[0]

                    # Convert to dB scale only (no normalization at storage time)
                    # Normalization will be applied during data loading for training/evaluation
                    gen_rdm_db = 20 * np.log10(np.abs(gen_rdm) + 1e-10)

                    # Store image, mask and label
                    gen_all_images.append(gen_rdm_db)
                    gen_all_masks.append(gen_mask)
                    gen_all_labels.append(n_targets)

            # Convert to numpy arrays and then to tensors
            gen_images = np.array(gen_all_images)
            gen_masks = np.array(gen_all_masks)
            gen_labels = np.array(gen_all_labels)

            # Convert to PyTorch tensors
            gen_images_tensor = torch.from_numpy(gen_images).float()
            gen_masks_tensor = torch.from_numpy(gen_masks).float()
            gen_labels_tensor = torch.from_numpy(gen_labels).long()

            # Create dataset dictionary - stored in memory
            generated_dataset = {
                'images': gen_images_tensor,
                'masks': gen_masks_tensor,
                'labels': gen_labels_tensor,
                'metadata': {
                    'samples_per_class': dataset_samples_per_class.value,
                    'max_targets': dataset_max_targets.value,
                    'target_type': target_type_select.value,
                    'n_frames': 1,
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
                    'data_format': 'dB_scale',  # Data is stored in dB scale, normalization applied during loading
                    'dataset_type': 'single_frame_segmentation'
                }
            }

        else:  # multi_frame
            print(f"🎬 Generating multi-frame segmentation dataset ({dataset_n_frames.value} frames)...")

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

            print(f"📊 Using Parameters from Section 1 Visualization:")
            print(f"  🎯 Dataset Generation:")
            print(f"    • Samples per class: {dataset_samples_per_class.value}")
            print(f"    • Max targets: {dataset_max_targets.value}")
            print(f"    • Target type: {target_type_select.value}")
            print(f"    • Frames per sequence: {dataset_n_frames.value}")
            print(f"  📡 Radar Configuration:")
            print(f"    • RDM size: {gen_rp.n_ranges} x {gen_rp.n_pulses}")
            print(f"    • PRF: {gen_rp.prf} Hz")
            print(f"    • Range resolution: {gen_rp.range_resolution} m")
            print(f"  🌊 Sea Clutter Configuration:")
            print(f"    • Clutter power: {gen_cp.mean_power_db} dB")
            print(f"    • Shape parameter: {gen_cp.shape_param}")
            print(f"    • AR coefficient: {gen_cp.ar_coeff}")
            print(f"    • Wave speed: {gen_cp.wave_speed_mps} m/s")
            print(f"    • Bragg offset: {gen_cp.bragg_offset_hz} Hz")
            print(f"  📶 Signal/Noise Configuration:")
            print(f"    • Target signal power: {target_signal_power.value} dB")
            print(f"    • Noise power: {noise_power.value} dB")
            print(f"    • SNR: {target_signal_power.value - noise_power.value:.1f} dB")

            # Generate multi-frame dataset with custom parameters
            gen_all_sequences = []
            gen_all_mask_sequences = []
            gen_all_labels = []

            # Get target type from user selection
            gen_target_type = getattr(sea_clutter.TargetType, target_type_select.value)

            # Generate data for each class
            for n_targets in range(dataset_max_targets.value + 1):
                print(f"🎯 Generating {dataset_samples_per_class.value} sequences for {n_targets} targets...")

                for gen_sample_idx in range(dataset_samples_per_class.value):
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
                        # Convert to dB scale only (no normalization at storage time)
                        # Normalization will be applied during data loading for training/evaluation
                        gen_rdm_db = 20 * np.log10(np.abs(gen_rdm) + 1e-10)
                        gen_processed_sequence.append(gen_rdm_db)
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

            # Create dataset dictionary - stored in memory
            generated_dataset = {
                'sequences': gen_sequences_tensor,
                'masks': gen_mask_sequences_tensor,
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
                    'data_format': 'dB_scale',  # Data is stored in dB scale, normalization applied during loading
                    'dataset_type': 'sequence_segmentation'
                }
            }

        # Calculate dataset size in GB
        total_bytes = 0
        if is_single_frame:
            total_bytes += generated_dataset['images'].element_size() * generated_dataset['images'].nelement()
            total_bytes += generated_dataset['masks'].element_size() * generated_dataset['masks'].nelement()
        else:
            total_bytes += generated_dataset['sequences'].element_size() * generated_dataset['sequences'].nelement()
            total_bytes += generated_dataset['masks'].element_size() * generated_dataset['masks'].nelement()
        
        total_bytes += generated_dataset['labels'].element_size() * generated_dataset['labels'].nelement()
        dataset_size_gb = total_bytes / (1024**3)  # Convert to GB

        gen_end_time = time.time()
        gen_generation_time = gen_end_time - gen_start_time

        print(f"✅ Dataset generation completed in {gen_generation_time:.1f} seconds!")
        print(f"� Dataset stored in memory (size: {dataset_size_gb:.3f} GB)")

        if is_single_frame:
            print(f"📈 Dataset shape: {generated_dataset['images'].shape}")
            print(f"� Masks shape: {generated_dataset['masks'].shape}")
        else:
            print(f"📈 Sequences shape: {generated_dataset['sequences'].shape}")
            print(f"📈 Masks shape: {generated_dataset['masks'].shape}")

        # Clean up temporary variables
        if is_single_frame:
            del gen_all_images, gen_all_masks, gen_all_labels
            del gen_images, gen_masks, gen_labels
            del gen_images_tensor, gen_masks_tensor, gen_labels_tensor
        else:
            del gen_all_sequences, gen_all_mask_sequences, gen_all_labels
            del gen_sequences, gen_mask_sequences, gen_labels
            del gen_sequences_tensor, gen_mask_sequences_tensor, gen_labels_tensor
        
        # Force garbage collection
        cleanup_memory(verbose=True)

    else:
        print("⏭️ No dataset generation requested.")

    print("✅ Dataset generation section completed.")
    return dataset_size_gb, generated_dataset


@app.cell
def _(mo):
    mo.md("""## 👁️ **Section 2.5: Dataset Visualization**""")
    return


@app.cell
def _(dataset_size_gb, generated_dataset, mo, os):
    # Dataset visualization controls - updated to support memory datasets
    viz_available_datasets = ["Use Generated Dataset (from memory)"] + ["Select dataset..."] + [f for f in os.listdir("data") if f.endswith('.pt')] if os.path.exists("data") else ["Use Generated Dataset (from memory)", "Select dataset..."]
    viz_dataset_dropdown = mo.ui.dropdown(
        options=viz_available_datasets,
        value=viz_available_datasets[0] if viz_available_datasets else "No datasets found",
        label="Dataset to Visualize"
    )

    viz_sample_button = mo.ui.button(
        label="🎲 Show Random Sample", 
        kind="neutral",
        value=0,
        on_click=lambda count: count + 1
    )

    # Show dataset info if generated dataset is available
    dataset_info_text = ""
    if generated_dataset is not None:
        dataset_info_text = f"**Generated Dataset Available:** {dataset_size_gb:.3f} GB in memory"
    else:
        dataset_info_text = "**No generated dataset in memory.** Generate one first or select a file dataset."

    mo.vstack([
        mo.md("**Explore your generated datasets by viewing random samples:**"),
        mo.md(dataset_info_text),
        viz_dataset_dropdown,
        viz_sample_button
    ])
    return viz_dataset_dropdown, viz_sample_button


@app.cell
def _(dataset_size_gb, generated_dataset, mo, plt, random, torch, viz_dataset_dropdown, viz_sample_button):
    # Dataset sample visualization - updated to support memory datasets
    viz_output = None

    if viz_sample_button.value > 0 and viz_dataset_dropdown.value != "Select dataset...":
        try:
            # Determine if using generated dataset or file dataset
            using_generated_viz = viz_dataset_dropdown.value == "Use Generated Dataset (from memory)"
            
            if using_generated_viz and generated_dataset is not None:
                print(f"📊 Using generated dataset from memory (size: {dataset_size_gb:.3f} GB)")
                viz_dataset = generated_dataset
            elif not using_generated_viz:
                # Load the selected dataset
                viz_dataset_path = f"data/{viz_dataset_dropdown.value}"
                viz_dataset = torch.load(viz_dataset_path)
                print(f"📂 Loaded dataset: {viz_dataset_dropdown.value}")
            else:
                print("❌ **Error**: No generated dataset available. Please generate a dataset first.")

            # Get dataset metadata
            viz_metadata = viz_dataset.get('metadata', {})
            viz_dataset_type = viz_metadata.get('dataset_type', 'unknown')
            viz_is_single_frame = 'single_frame' in viz_dataset_type

            print(f" Dataset type: {viz_dataset_type}")

            if viz_is_single_frame:
                # Single frame dataset
                viz_images = viz_dataset['images']
                viz_masks = viz_dataset['masks']
                viz_labels = viz_dataset['labels']

                # Select random sample
                viz_sample_idx = random.randint(0, len(viz_images) - 1)
                viz_sample_img = viz_images[viz_sample_idx].numpy()
                viz_sample_mask = viz_masks[viz_sample_idx].numpy()
                viz_sample_label = viz_labels[viz_sample_idx].item()

                print(f"🎯 Sample {viz_sample_idx}: {viz_sample_label} targets")
                print(f"📏 Image shape: {viz_sample_img.shape}")
                print(f"📏 Mask shape: {viz_sample_mask.shape}")

                # Create visualization
                viz_fig, viz_axes = plt.subplots(1, 2, figsize=(12, 6))

                # Show RDM
                viz_axes[0].imshow(viz_sample_img, aspect='auto', cmap='viridis', origin='lower')
                viz_axes[0].set_title(f'Sample RDM (Label: {viz_sample_label} targets)')
                viz_axes[0].set_xlabel('Doppler Bin')
                viz_axes[0].set_ylabel('Range Bin')
                viz_cbar0 = plt.colorbar(viz_axes[0].images[0], ax=viz_axes[0], label='Power (dB)')

                # Show mask
                viz_axes[1].imshow(viz_sample_mask, aspect='auto', cmap='Reds', origin='lower', vmin=0, vmax=1)
                viz_axes[1].set_title(f'Sample Mask (Label: {viz_sample_label} targets)')
                viz_axes[1].set_xlabel('Doppler Bin')
                viz_axes[1].set_ylabel('Range Bin')
                viz_cbar1 = plt.colorbar(viz_axes[1].images[0], ax=viz_axes[1], label='Target Mask')

                plt.suptitle(f'Dataset: {viz_dataset_dropdown.value} | Sample #{viz_sample_idx}')
                plt.tight_layout()

                # Display dataset info
                viz_info = mo.md(f"""
                **📊 Dataset Information:** \t
                - **Name**: {viz_dataset_dropdown.value} \t
                - **Type**: {viz_dataset_type} \t
                - **Total Samples**: {len(viz_images):,} \t
                - **Sample #{viz_sample_idx}**: {viz_sample_label} targets \t
                - **Image Shape**: {viz_sample_img.shape} \t
                - **Classes**: {viz_metadata.get('max_targets', 'unknown')} + 1 (0-{viz_metadata.get('max_targets', 'unknown')} targets) \t
                - **Target Type**: {viz_metadata.get('target_type', 'unknown')} \t

                *Click '🎲 Show Random Sample' again to see another random sample.*
                """)

                viz_output = mo.vstack([viz_fig, viz_info])

            else:
                # Multi-frame dataset
                viz_sequences = viz_dataset['sequences']
                viz_mask_sequences = viz_dataset['masks']
                viz_labels = viz_dataset['labels']

                # Select random sample
                viz_sample_idx = random.randint(0, len(viz_sequences) - 1)
                viz_sample_seq = viz_sequences[viz_sample_idx].numpy()
                viz_sample_mask = viz_mask_sequences[viz_sample_idx].numpy()
                viz_sample_label = viz_labels[viz_sample_idx].item()

                print(f"🎯 Sample {viz_sample_idx}: {viz_sample_label} targets")
                print(f"📏 Sequence shape: {viz_sample_seq.shape}")
                print(f"📏 Mask sequence shape: {viz_sample_mask.shape}")

                # Create visualization
                viz_n_frames_to_show = min(3, viz_sample_seq.shape[0])
                viz_fig, viz_axes = plt.subplots(2, viz_n_frames_to_show, figsize=(4*viz_n_frames_to_show, 8))

                if viz_n_frames_to_show == 1:
                    viz_axes = viz_axes.reshape(2, 1)

                for viz_frame_idx in range(viz_n_frames_to_show):
                    # Show RDM
                    viz_axes[0, viz_frame_idx].imshow(viz_sample_seq[viz_frame_idx], aspect='auto', cmap='viridis', origin='lower')
                    viz_axes[0, viz_frame_idx].set_title(f'Frame {viz_frame_idx+1} RDM')
                    viz_axes[0, viz_frame_idx].set_xlabel('Doppler Bin')
                    viz_axes[0, viz_frame_idx].set_ylabel('Range Bin')

                    # Show mask
                    viz_axes[1, viz_frame_idx].imshow(viz_sample_mask[viz_frame_idx], aspect='auto', cmap='Reds', origin='lower', vmin=0, vmax=1)
                    viz_axes[1, viz_frame_idx].set_title(f'Frame {viz_frame_idx+1} Mask')
                    viz_axes[1, viz_frame_idx].set_xlabel('Doppler Bin')
                    viz_axes[1, viz_frame_idx].set_ylabel('Range Bin')

                plt.suptitle(f'Dataset: {viz_dataset_dropdown.value} | Sample #{viz_sample_idx} | Label: {viz_sample_label} targets')
                plt.tight_layout()

                # Display dataset info
                viz_frame_info = f" (showing first {viz_n_frames_to_show} of {viz_sample_seq.shape[0]} frames)" if viz_sample_seq.shape[0] > viz_n_frames_to_show else ""
                viz_info = mo.md(f"""
                **📊 Dataset Information:** \t
                - **Name**: {viz_dataset_dropdown.value} \t
                - **Type**: {viz_dataset_type} \t
                - **Total Sequences**: {len(viz_sequences):,} \t
                - **Sample #{viz_sample_idx}**: {viz_sample_label} targets \t
                - **Sequence Shape**: {viz_sample_seq.shape}{viz_frame_info} \t
                - **Classes**: {viz_metadata.get('max_targets', 'unknown')} + 1 (0-{viz_metadata.get('max_targets', 'unknown')} targets) \t
                - **Target Type**: {viz_metadata.get('target_type', 'unknown')} \t
                - **Frames per Sequence**: {viz_metadata.get('n_frames', 'unknown')}\t

                *Click '🎲 Show Random Sample' again to see another random sample.*
                """)

                viz_output = mo.vstack([viz_fig, viz_info])

        except Exception as e:
            viz_output = mo.md(f"""
            ❌ **Error loading dataset**: 
            ```
            {str(e)}
            ```
            Please ensure the dataset file is valid and accessible.
            """)

    elif viz_sample_button.value > 0:
        viz_output = mo.md("⚠️ **Please select a dataset first!**")
    else:
        viz_output = mo.md("**Select a dataset above and click '🎲 Show Random Sample' to visualize data.**")

    viz_output
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
    train_n_channels = mo.ui.slider(start=1, stop=10, value=1, step=1, label="Input Channels (the number of channels need to correspond to the number of frames used for generating the dataset")
    train_base_filters = mo.ui.slider(start=8, stop=64, value=16, step=8, label="Base Filters")

    mo.hstack([train_n_channels, train_base_filters])
    return train_base_filters, train_n_channels


@app.cell
def _(mo):
    # Training parameters
    train_epochs = mo.ui.slider(start=5, stop=100, value=30, step=5, label="Epochs")
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
def _(mo, models, train_base_filters, train_n_channels):
    # Model architecture preview
    def show_model_info():
        """Show U-Net model architecture and trainable parameters"""
        try:
            # Create a temporary U-Net model to show architecture
            temp_model = models.UNet(n_channels=train_n_channels.value, base_filters=train_base_filters.value)
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
            
            # Get model parameter details
            total_params = sum(p.numel() for p in temp_model.parameters())
            
            return {
                'trainable_params': trainable_params,
                'total_params': total_params,
                'model_size_mb': (trainable_params * 4) / (1024**2),  # Assuming float32 (4 bytes per param)
                'n_channels': train_n_channels.value,
                'base_filters': train_base_filters.value
            }
        except Exception as e:
            return {'error': str(e)}
    
    model_info = show_model_info()
    
    if 'error' not in model_info:
        model_info_md = f"""
        ## 🧠 **U-Net Model Configuration**
        
        **Architecture:**
        - Input Channels: **{model_info['n_channels']}** {'(Single Frame)' if model_info['n_channels'] == 1 else '(Multi-Frame)'}\t
        - Base Filters: **{model_info['base_filters']}**\t
        
        **Parameters:**
        - Trainable Parameters: **{model_info['trainable_params']:,}**\t
        - Total Parameters: **{model_info['total_params']:,}**\t
        - Estimated Model Size: **{model_info['model_size_mb']:.1f} MB**\t
        
        *Note: Model will be created with these parameters when training starts.*
        """
    else:
        model_info_md = f"""
        ## 🧠 **U-Net Model Configuration**
        
        ⚠️ **Error getting model info:** {model_info['error']}
        """
    
    mo.md(model_info_md)
    return model_info


@app.cell
def _(mo, os):
    # Dataset selection for training - updated to support both file and memory datasets
    train_available_datasets = ["Use Generated Dataset (from memory)"] + ["Select dataset..."] + [f for f in os.listdir("data") if f.endswith('.pt')]
    train_dataset_dropdown = mo.ui.dropdown(
        options=train_available_datasets,
        value=train_available_datasets[0] if train_available_datasets else "No datasets found",
        label="Training Dataset"
    )

    train_model_name = mo.ui.text(value="my_trained_model", label="Model Save Name")

    train_start_button = mo.ui.button(
        label="🚀 Start Training", 
        kind="success",
        value=0,
        on_click=lambda count: count + 1
    )

    mo.vstack([
        train_dataset_dropdown,
        train_model_name,
        train_start_button
    ])
    return train_dataset_dropdown, train_model_name, train_start_button


@app.cell
def _(
    cleanup_memory,
    comprehensive_evaluation,
    dataset_size_gb,
    generated_dataset,
    mo,
    models,
    os,
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
    if train_start_button.value > 0 and train_dataset_dropdown.value != "Select dataset...":
        print("🚀 Starting model training...")
        training_start_time = time.time()

        # Determine if using generated dataset or file dataset
        using_generated_dataset = train_dataset_dropdown.value == "Use Generated Dataset (from memory)"
        
        if using_generated_dataset and generated_dataset is not None:
            print(f"📊 Using generated dataset from memory (size: {dataset_size_gb:.3f} GB)")
            training_dataset = generated_dataset
            training_dataset_path = None  # No file path for memory dataset
        elif not using_generated_dataset:
            training_dataset_path = f"data/{train_dataset_dropdown.value}"
            print(f"📂 Loading dataset from: {training_dataset_path}")
            training_dataset = None  # Will load from file
        else:
            print("❌ Error: No generated dataset available in memory. Please generate a dataset first.")

        training_model_save_path = f"pretrained/{train_model_name.value}.pt"

        # Ensure pretrained directory exists
        os.makedirs("pretrained", exist_ok=True)

        print(f"📊 Training Configuration:")
        print(f"  • Model: U-Net")
        if using_generated_dataset:
            print(f"  • Dataset: Generated dataset (memory)")
        else:
            print(f"  • Dataset: {train_dataset_dropdown.value}")
        print(f"  • Input Channels: {train_n_channels.value}")
        print(f"  • Base Filters: {train_base_filters.value}")
        print(f"  • Epochs: {train_epochs.value}")
        print(f"  • Batch Size: {train_batch_size.value}")
        print(f"  • Learning Rate: {train_learning_rate.value}")
        print(f"  • Loss Weights: BCE={train_bce_weight.value}, Dice={train_dice_weight.value}")

        # Show model parameters before training
        temp_model = models.UNet(n_channels=train_n_channels.value, base_filters=train_base_filters.value)
        trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        model_size_mb = (trainable_params * 4) / (1024**2)
        print(f"🧠 Model Architecture:")
        print(f"  • Trainable Parameters: {trainable_params:,}")
        print(f"  • Estimated Model Size: {model_size_mb:.1f} MB")
        del temp_model  # Clean up temporary model

        try:
            # Check if dataset is for segmentation or classification
            if using_generated_dataset:
                training_dataset_check = training_dataset
            else:
                training_dataset_check = torch.load(training_dataset_path)
            
            # Add dataset validation and debugging
            print("🔍 Dataset Validation:")
            if 'masks' in training_dataset_check:
                # Segmentation dataset
                data_key = 'images' if 'images' in training_dataset_check else 'sequences'
                print(f"  • Dataset type: Segmentation")
                print(f"  • Data shape: {training_dataset_check[data_key].shape}")
                print(f"  • Masks shape: {training_dataset_check['masks'].shape}")
                print(f"  • Labels shape: {training_dataset_check['labels'].shape}")
                
                # Validate dimensions match
                n_samples_data = training_dataset_check[data_key].shape[0]
                n_samples_masks = training_dataset_check['masks'].shape[0]
                n_samples_labels = training_dataset_check['labels'].shape[0]
                
                print(f"  • Samples in data: {n_samples_data}")
                print(f"  • Samples in masks: {n_samples_masks}")
                print(f"  • Samples in labels: {n_samples_labels}")
                
                if n_samples_data != n_samples_masks or n_samples_data != n_samples_labels:
                    print(f"❌ ERROR: Dimension mismatch detected!")
                    print(f"  • Data samples: {n_samples_data}")
                    print(f"  • Mask samples: {n_samples_masks}")
                    print(f"  • Label samples: {n_samples_labels}")
                    raise ValueError(f"Dataset has mismatched dimensions: data={n_samples_data}, masks={n_samples_masks}, labels={n_samples_labels}")
                
                print(f"  ✅ All dimensions match: {n_samples_data} samples")
            
            training_is_segmentation = 'masks' in training_dataset_check

            if training_is_segmentation:
                print("🎯 Detected segmentation dataset - proceeding with U-Net training")

                # Train the model - handle both memory and file datasets
                if using_generated_dataset:
                    # For memory datasets, we need to modify train_model to accept dataset directly
                    # For now, save temporarily and clean up after
                    temp_dataset_path = f"temp_dataset_for_training.pt"
                    print(f"💾 Saving temporary dataset for training...")
                    torch.save(training_dataset, temp_dataset_path)
                    
                    # Verify the saved dataset
                    temp_check = torch.load(temp_dataset_path)
                    data_key = 'images' if 'images' in temp_check else 'sequences'
                    print(f"✅ Temporary dataset verified:")
                    print(f"  • Data: {temp_check[data_key].shape}")
                    print(f"  • Masks: {temp_check['masks'].shape}")
                    print(f"  • Labels: {temp_check['labels'].shape}")
                    del temp_check
                    
                    train_model(
                        dataset_path=temp_dataset_path,
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
                    
                    # Clean up temporary file
                    if os.path.exists(temp_dataset_path):
                        os.remove(temp_dataset_path)
                        print(f"🧹 Removed temporary dataset file")
                else:
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
                
                # For analysis, use the appropriate dataset path
                analysis_dataset_path = training_dataset_path if not using_generated_dataset else temp_dataset_path
                if using_generated_dataset:
                    # Recreate temp file for analysis
                    torch.save(training_dataset, temp_dataset_path)
                
                results_plot = comprehensive_evaluation(
                    model_path=training_model_save_path,
                    dataset_path=analysis_dataset_path,
                    base_filter_size=train_base_filters.value,
                    marimo_var = True
                )
                
                if using_generated_dataset and os.path.exists(temp_dataset_path):
                    os.remove(temp_dataset_path)

                # Clean up memory after training and analysis
                print("🧹 Cleaning up memory after training...")
                
                # Clear loaded dataset variables to free memory
                del training_dataset_check
                
                # Force garbage collection and GPU cache clearing
                cleanup_memory(verbose=True)

                # Display training summary
                training_output = mo.vstack([mo.md(f"""
                ## 🎉 **Training Summary:**
                - **Total Time**: {training_time/60:.1f} minutes\t
                - **Model Type**: U-Net with {train_base_filters.value} base filters\t
                - **Final Model**: `{training_model_save_path}`\t
                - **Training completed successfully!** ✅
                """),
                mo.md("## 📈 **Model Evaluation Results:**"),
                    results_plot
                ])
            

            else:
                training_output = mo.md("""
                ❌ **Error**: Dataset appears to be for classification, not segmentation.

                Please generate a single-frame or multi-frame segmentation dataset first (both include masks).
                """)

        except Exception as e:
            # Clean up any loaded variables in case of error
            try:
                if 'training_dataset_check' in locals():
                    del training_dataset_check
                if 'temp_dataset_path' in locals() and os.path.exists(temp_dataset_path):
                    os.remove(temp_dataset_path)
            except:
                pass


    elif train_start_button.value > 0:
        training_output = mo.md("⚠️ **Please select a dataset first!**")
    else:
        training_output = mo.md("Configure parameters above and click '🚀 Start Training' to begin. Note that training may take a while depending on the number of epochs, samples in dataset and model complexity. When training has finished, the performance of the model is measured and shown automatically.")

    training_output
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 📋 **Quick Reference**

    ### Parameter Guidelines:

    **Sea Clutter Parameters:**\t
    - **Mean Power**: Controls overall clutter intensity (-30 to 30 dB)\t
    - **Shape Parameter**: Controls clutter distribution (0.01-1.0, lower = more spiky)\t
    - **AR Coefficient**: Controls temporal correlation (0.7-0.99, higher = more correlated)\t
    - **Bragg Components**: Simulate ocean wave reflections

    ### Recommended Settings:
    - **Learning Rate**: 1e-4 for most cases\t
    - **Batch Size**: 16-32 depending on memory\t
    - **Loss Weights**: BCE=1.0, Dice=1.0 for balanced training\t
    - **Patience**: 10-15 epochs for early stopping
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 🎯 **Usage Tips**

    1. **Design your data in Section 1**: Adjust radar, clutter, and target parameters to create the desired simulation characteristics\t
    2. **Visualize before generating**: Use the interactive visualization to understand how parameters affect sea clutter and target visibility\t
    3. **Generate datasets with tested parameters**: Section 2 automatically uses your Section 1 configuration for dataset generation\t
    4. **Explore your datasets**: Use Section 2.5 to visualize random samples from any generated dataset\t
    5. **Start with small datasets**: Use fewer samples (100-200) for quick testing and validation\t
    6. **Monitor training**: Watch the console output for training progress and metrics\t

    **Performance Notes:**\t
    - Training time scales with dataset size and model complexity \t
    - Larger datasets generally give better performance but take longer to generate \t
    - Multi-frame models can capture temporal information for better target detection \t
    - SNR significantly affects model performance - test with realistic values\t
    - Model parameters (channels, filters) must match between training and evaluation
    """
    )
    return

@app.cell
def _(mo):
    mo.md(
        """
    ## **Credits and Feedback**

    This notebook and the underlying software were created by **Pepijn Lens**.

    - **Contribute or report issues:** Github repository
    - **Contact:** pepijn.lens@tno.nl

    Contributions, suggestions, and feedback are welcome!
    """
    )
    return


if __name__ == "__main__":
    app.run()
