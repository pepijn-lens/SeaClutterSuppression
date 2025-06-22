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

    # Import our custom modules
    import sea_clutter
    import models
    from src.generate_data import simulate_sequence_with_realistic_targets_and_masks
    from src.simulation import simulate_sequence_with_realistic_targets
    from src.unet_training import train_model
    from src.end_to_end_evaluate import comprehensive_evaluation
    from src.end_to_end_helper import analyze_single_sample


    return (
        analyze_single_sample,
        comprehensive_evaluation,
        mo,
        models,
        np,
        os,
        plt,
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
    # Interactive Sea Clutter Simulation & U-Net Training Notebook

    This notebook provides four main functionalities: \t
    1. **Parameter Adjustment & Real-time Visualization** - Adjust radar and clutter parameters and see sea clutter simulation \t
    2. **Dataset Generation** - Generate synthetic sea clutter datasets for training \t
    3. **U-Net Training** - Train deep learning models on the generated data\t
    4. **Model Evaluation** - Evaluate trained models on any dataset with basic performance metrics\t
    5. **Single Sample Analysis** - Analyze individual samples to understand model predictions and performance\t
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""## **Section 1: Real-time Sea Clutter Visualization**""")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Radar Parameters

    Configure the fundamental radar system parameters that determine the measurement grid size and resolution.
    """
    )
    return


@app.cell
def _(mo):
    # Configure the basic radar operating parameters that define the measurement grid and timing
    # PRF determines the maximum unambiguous velocity, while pulses and ranges set the resolution
    radar_prf = mo.ui.slider(start=1000, stop=10000, value=5000, step=100, label="PRF (Hz)")
    radar_n_pulses = mo.ui.slider(start=64, stop=2048, value=128, step=64, label="Number of Pulses")
    radar_n_ranges = mo.ui.slider(start=64, stop=2048, value=128, step=64, label="Number of Range Bins")

    mo.hstack([radar_prf, radar_n_pulses, radar_n_ranges])
    return radar_n_pulses, radar_n_ranges, radar_prf


@app.cell
def _(mo):
    mo.md(
        """
    ### Clutter Parameters

    Control the statistical and spectral characteristics of sea clutter returns.
    """
    )
    return


@app.cell
def _(mo):
    # Control sea clutter characteristics: intensity, temporal correlation, and ocean wave effects
    # Lower shape parameter = more spiky clutter; higher AR coefficient = more temporal correlation
    clutter_mean_power = mo.ui.slider(start=-20, stop=25, value=16, step=1, label="Mean Clutter Power (dB)")
    clutter_shape_param = mo.ui.slider(start=0.01, stop=1.0, value=0.75, step=0.01, label="Shape Parameter")
    clutter_ar_coeff = mo.ui.slider(start=0.5, stop=0.99, value=0.9, step=0.01, label="AR Coefficient")
    clutter_bragg_offset = mo.ui.slider(start=0, stop=100, value=45, step=1, label="Bragg Offset (Hz)")
    clutter_bragg_width = mo.ui.slider(start=1, stop=10, value=4, step=0.5, label="Bragg Width (Hz)")
    clutter_bragg_power = mo.ui.slider(start=0, stop=20, value=0, step=0.5, label="Bragg Power (dB)")
    clutter_wave_speed = mo.ui.slider(start=0, stop=15, value=4, step=0.5, label="Wave Speed (m/s)")

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
    mo.md(
        """
    ### Target Parameters

    Define the number and type of targets present in the simulation scenario.
    """
    )
    return


@app.cell
def _(mo):
    # Define the number and type of targets to simulate in the scene
    # Different target types have realistic RCS and velocity characteristics
    target_n_targets = mo.ui.slider(start=0, stop=15, value=6, label="Number of Targets")
    target_type_select = mo.ui.dropdown(
        options=["SPEEDBOAT", "CARGO_SHIP"],
        value="SPEEDBOAT",
        label="Target Type"
    )

    mo.hstack([target_n_targets, target_type_select])
    return target_n_targets, target_type_select


@app.cell
def _(mo):
    mo.md(
        """
    ### Signal (SNR Control)

    Adjust target signal strength to control detection difficulty.
    """
    )
    return


@app.cell
def _(mo):
    # Control target signal strength - higher values create easier detection scenarios
    # This determines the Signal-to-Noise Ratio (SNR) for target visibility
    target_signal_power = mo.ui.slider(start=1, stop=30, value=20, step=1, label="Target Signal Power (dB)")
    noise_power = 0
    mo.hstack([target_signal_power])
    return noise_power, target_signal_power


@app.cell
def _(mo):
    mo.md(
        """
    ### Sequence Parameters (for Multi-Frame Generation)

    Configure temporal aspects for generating sequences of radar frames.
    """
    )
    return


@app.cell
def _(mo):
    # Configure temporal parameters for multi-frame sequences
    # Higher frame rates capture faster target movements; longer sequences show more motion patterns
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
    radar_n_pulses,
    radar_n_ranges,
    radar_prf,
    random,
    sea_clutter,
    seq_frame_rate,
    seq_total_time,
    simulate_sequence_with_realistic_targets,
    target_n_targets,
    target_signal_power,
    target_type_select,
):
    sequence_data = None

    # Create radar and clutter parameters from sliders
    sim_rp = sea_clutter.RadarParams(
        prf=radar_prf.value,
        n_pulses=radar_n_pulses.value,
        n_ranges=radar_n_ranges.value,
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
    sim_sp.n_frames = max(2, int(seq_total_time.value * seq_frame_rate.value))  # Always at least 2 frames

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
        thermal_noise_db=1, target_signal_power=target_signal_power.value
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
        'noise_power': noise_power,
        'target_signal_power': target_signal_power.value
    }

    # Display basic info about generated sequence
    sim_info = mo.md(f"""
    **Sequence Generated Successfully:**\t
    - **Frames**: {len(sim_rdm_list)} frames at {sim_sp.frame_rate_hz} Hz \t 
    - **Duration**: {len(sim_rdm_list)/sim_sp.frame_rate_hz:.1f} seconds\t
    - **Targets**: {target_n_targets.value} x {target_type_select.value}\t
    - **SNR**: {target_signal_power.value - noise_power:.1f} dB\t

    *Use the interactive frame viewer below to explore the sequence.*
    """)

    sequence_display = sim_info

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
            mo.md("**Interactive Frame Navigation:**"),
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
        # consistent scaling across all frames  
        frame_vmin = 0
        frame_vmax = 45

        # Calculate axes
        frame_fd = np.fft.fftshift(np.fft.fftfreq(sequence_data['rp'].n_pulses, d=1.0 / sequence_data['rp'].prf))
        frame_velocity = frame_fd * sequence_data['rp'].carrier_wavelength / 2.0
        frame_range_axis = np.arange(sequence_data['rp'].n_ranges) * sequence_data['rp'].range_resolution

        # Create the plot for the selected frame
        frame_fig, frame_ax = plt.subplots(figsize=(12, 8))

        # Display the frame selected by the slider
        current_frame = frame_slider.value
        current_frame_data = sequence_data['rdm_db_list'][current_frame]

        frame_im = frame_ax.imshow(current_frame_data, aspect='auto', cmap='viridis',
                       extent=[frame_velocity[0], frame_velocity[-1], frame_range_axis[-1], frame_range_axis[0]],
                       vmin=frame_vmin, vmax=frame_vmax, interpolation='nearest')

        frame_ax.set_xlabel('Radial Velocity (m/s)')
        frame_ax.set_ylabel('Range (m)')
        frame_ax.set_title(f'Frame {current_frame} - Interactive RD Map Sequence')

        # Add colorbar
        frame_cbar = plt.colorbar(frame_im, ax=frame_ax)
        frame_cbar.set_label('Power (dB)')
        plt.tight_layout()

        # Frame timing and detailed information
        frame_time = current_frame / sequence_data['sp'].frame_rate_hz
        total_time = len(sequence_data['rdm_list']) / sequence_data['sp'].frame_rate_hz

        # Display detailed info
        frame_info = mo.md(f"""
        **Frame Information:**

        **Current Frame**: {current_frame} / {len(sequence_data['rdm_list'])-1}  
        **Time**: {frame_time:.2f}s / {total_time:.2f}s  
        **Targets**: {sequence_data['n_targets']} x {sequence_data['target_type']}  
        **SNR**: {sequence_data['target_signal_power'] - sequence_data['noise_power']:.1f} dB  
        **Frame Rate**: {sequence_data['sp'].frame_rate_hz} Hz  
        **RDM Size**: {sequence_data['rp'].n_ranges} x {sequence_data['rp'].n_pulses}  

        *Adjust the slider above to navigate through frames and observe target movement over time.*
        """)

        frame_display = mo.vstack([frame_fig, frame_info])
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
    mo.md("""## **Section 2: Dataset Generation**""")
    return


@app.cell
def _(mo):
    # Configure dataset generation parameters using the simulation settings from Section 1
    # More samples per class improve model generalization; more frames capture temporal patterns
    dataset_samples_per_class = mo.ui.slider(
        start=100, stop=2000, value=500, step=50,
        label="Samples per Class", show_value=True
    )
    dataset_max_targets = mo.ui.slider(
        start=0, stop=15, value=10, step=1,
        label="Maximum Targets", show_value=True
    )
    dataset_n_frames = mo.ui.slider(
        start=1, stop=10, value=3, step=1,
        label="Frames per Sequence", show_value=True
    )
    dataset_name = mo.ui.text(
        value="my_sea_clutter_dataset",
        label="Dataset Name"
    )

    dataset_config = mo.md(
        """
        {dataset_samples_per_class}

        {dataset_max_targets}

        {dataset_n_frames}

        {dataset_name}
        """
    ).batch(
        dataset_samples_per_class=dataset_samples_per_class,
        dataset_max_targets=dataset_max_targets,
        dataset_n_frames=dataset_n_frames,
        dataset_name=dataset_name,
    ).form(
        submit_button_label="🚀 Generate Dataset",
        bordered=False,
        show_clear_button=True,
        clear_button_label="Reset"
    )

    dataset_config
    return (dataset_config,)


@app.cell
def _(
    clutter_ar_coeff,
    clutter_bragg_offset,
    clutter_bragg_power,
    clutter_bragg_width,
    clutter_mean_power,
    clutter_shape_param,
    clutter_wave_speed,
    dataset_config,
    mo,
    noise_power,
    np,
    radar_n_pulses,
    radar_n_ranges,
    radar_prf,
    random,
    sea_clutter,
    simulate_sequence_with_realistic_targets_and_masks,
    target_signal_power,
    target_type_select,
    torch,
):
    mo.stop(
        dataset_config.value is None,
        mo.md("Click the `🚀 Generate Dataset` button to continue").callout(kind="warn")
    )

    # Dataset generation logic - stored as variable instead of file
    generated_dataset = None
    dataset_size_gb = 0.0

    # Use parameters from visualization section (user-configured)
    gen_rp = sea_clutter.RadarParams(
        prf=radar_prf.value,
        n_pulses=radar_n_pulses.value,
        n_ranges=radar_n_ranges.value,
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
    gen_sp.n_frames = dataset_config.value["dataset_n_frames"]

    # Generate multi-frame dataset with custom parameters
    gen_all_sequences = []
    gen_all_mask_sequences = []
    gen_all_labels = []

    # Get target type from user selection
    gen_target_type = getattr(sea_clutter.TargetType, target_type_select.value)

    # Generate data for each class
    for n_targets in mo.status.progress_bar(range(dataset_config.value["dataset_max_targets"]+1)):
        for gen_sample_idx in range(dataset_config.value["dataset_samples_per_class"]):
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
                thermal_noise_db=noise_power, target_signal_power=target_signal_power.value
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
            'samples_per_class': dataset_config.value["dataset_samples_per_class"],
            'max_targets': dataset_config.value["dataset_max_targets"],
            'target_type': target_type_select.value,
            'n_frames': dataset_config.value["dataset_n_frames"],
            'n_ranges': gen_rp.n_ranges,
            'n_doppler_bins': gen_rp.n_pulses,
            'range_resolution': gen_rp.range_resolution,
            'prf': gen_rp.prf,
            'carrier_wavelength': gen_rp.carrier_wavelength,
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
                'noise_power_db': noise_power,
                'snr_db': target_signal_power.value - noise_power
            },
            'class_names': [f"{targets_idx}_targets" for targets_idx in range(dataset_config.value["dataset_max_targets"] + 1)],
            'data_format': 'dB_scale',  # Data is stored in dB scale, normalization applied during loading
        }
    }

    # Calculate dataset size in GB
    total_bytes = 0
    total_bytes += generated_dataset['sequences'].element_size() * generated_dataset['sequences'].nelement()
    total_bytes += generated_dataset['masks'].element_size() * generated_dataset['masks'].nelement()

    total_bytes += generated_dataset['labels'].element_size() * generated_dataset['labels'].nelement()
    dataset_size_gb = total_bytes / (1024**3)  # Convert to GB

    del gen_all_sequences, gen_all_mask_sequences, gen_all_labels
    del gen_sequences, gen_mask_sequences, gen_labels
    del gen_sequences_tensor, gen_mask_sequences_tensor, gen_labels_tensor
    return dataset_size_gb, generated_dataset


@app.cell
def _(dataset_config, generated_dataset, mo):
    mo.stop(
        generated_dataset is None,
        mo.md("No generated dataset found. Please generate a dataset in Section 2 before saving.").callout(kind="warn")
    )

    # Save the generated dataset to disk for future use in training or evaluation
    # The dataset will be stored in the 'data' directory with .pt extension
    save_dataset_name = mo.ui.text(
        value=dataset_config.value["dataset_name"] if (dataset_config is not None and dataset_config.value is not None) else "my_sea_clutter_dataset",
        label="Save Dataset As"
    )

    save_button = mo.md(
        """
        {save_dataset_name}
        """
    ).batch(
        save_dataset_name=save_dataset_name
    ).form(
        submit_button_label="Save Dataset",
        bordered=False,
    )

    save_button
    return (save_button,)


@app.cell
def _(dataset_size_gb, generated_dataset, mo, os, save_button, torch):
    mo.stop(
        save_button.value is None,
        mo.md("Click on 'Save Dataset' if you want to save for future usage.").callout(kind="warn")
    )

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Construct filename with .pt extension
    filename = save_button.value["save_dataset_name"]
    if not filename.endswith('.pt'):
        filename += '.pt'

    save_path = f"data/{filename}"

    # Save the dataset
    torch.save(generated_dataset, save_path)

    save_result = mo.md(f"""
    ✅ **Dataset Saved Successfully!**

    - **File**: `{save_path}`
    - **Size**: {dataset_size_gb:.2f} GB
    - **Samples**: {len(generated_dataset['sequences']):,}
    - **Classes**: {generated_dataset['metadata']['max_targets'] + 1}
    - **Frames per Sample**: {generated_dataset['metadata']['n_frames']}

    The dataset is now available for training in Section 3.
    """)

    save_result
    return


@app.cell
def _(mo):
    mo.md("""## **Section 3: U-Net Training**""")
    return


@app.cell
def _(mo):
    mo.md(
        """
    Configure U-Net model and training parameters:

    **Architecture**: Input channels must match the number of frames per sequence from your dataset.  
    **Training**: Balance epochs, batch size, and learning rate for optimal convergence.  
    **Loss Function**: BCE handles pixel-level accuracy, Dice handles class distribution imbalance.
    """
    )
    return


@app.cell
def _(mo, os):
    # Configure U-Net architecture and training hyperparameters
    # Input channels should match frames per sequence; more base filters = higher model capacity
    # Learning rate and patience control training speed and overfitting prevention
    train_n_channels = mo.ui.slider(
        start=1, stop=10, value=3, step=1,
        label="Input Channels (match the number of frames used)", show_value=True
    )
    train_base_filters = mo.ui.slider(
        start=8, stop=64, value=16, step=8,
        label="Base Filters", show_value=True
    )

    train_epochs = mo.ui.slider(
        start=5, stop=100, value=30, step=5,
        label="Epochs", show_value=True
    )
    train_batch_size = mo.ui.slider(
        start=8, stop=64, value=16, step=8,
        label="Batch Size", show_value=True
    )
    train_learning_rate = mo.ui.slider(
        start=1e-5, stop=1e-2, value=1e-4, step=1e-5,
        label="Learning Rate", show_value=True
    )
    train_patience = mo.ui.slider(
        start=5, stop=50, value=10, step=1,
        label="Early Stopping Patience", show_value=True
    )

    train_bce_weight = mo.ui.slider(
        start=0.1, stop=2.0, value=1.0, step=0.1,
        label="BCE Weight", show_value=True
    )
    train_dice_weight = mo.ui.slider(
        start=0.1, stop=2.0, value=1.0, step=0.1,
        label="Dice Weight", show_value=True
    )

    train_available_datasets = ["Use Generated Dataset (from memory)"] + ["Select dataset..."] + [
        f for f in os.listdir("data") if f.endswith('.pt')
    ]
    train_dataset_dropdown = mo.ui.dropdown(
        options=train_available_datasets,
        value=train_available_datasets[0],
        label="Training Dataset"
    )

    train_model_name = mo.ui.text(
        value="my_trained_model",
        label="Model Save Name"
    )

    # Compose the form
    train_config = mo.md(
        """
        ### **Model Architecture**
        {train_n_channels}  
        {train_base_filters}  

        ### **Training Parameters**
        {train_epochs}  
        {train_batch_size}  
        {train_learning_rate}  
        {train_patience}  

        ### **Loss Weights**
        {train_bce_weight}  
        {train_dice_weight}  

        ### **Dataset & Model**
        {train_dataset_dropdown}  
        {train_model_name}
        """
    ).batch(
        train_n_channels=train_n_channels,
        train_base_filters=train_base_filters,
        train_epochs=train_epochs,
        train_batch_size=train_batch_size,
        train_learning_rate=train_learning_rate,
        train_patience=train_patience,
        train_bce_weight=train_bce_weight,
        train_dice_weight=train_dice_weight,
        train_dataset_dropdown=train_dataset_dropdown,
        train_model_name=train_model_name
    ).form(
        submit_button_label="🚀 Start Training",
        bordered=False,
        show_clear_button=True,
        clear_button_label="Reset"
    )

    train_config
    return (train_config,)


@app.cell
def _(mo, models, train_config):
    mo.stop(train_config.value is None)
    # Preview model parameter counts
    base_filters = train_config.value["train_base_filters"]
    n_channels = train_config.value["train_n_channels"]
    temp_model = models.UNet(n_channels=n_channels, base_filters=base_filters)
    trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    param_info = mo.md(f"""
    ### **Model Stats**
    - Trainable Parameters: **{trainable_params:,}**  
    - **Base Filters**: {base_filters}
    - **Input Channels**: {n_channels}
    """)
    del temp_model
    param_info
    return


@app.cell
def _(generated_dataset, mo, os, time, torch, train_config, train_model):

    mo.stop(
        train_config.value is None,
        mo.md("Please press on the '🚀 Start Training' button.").callout(kind="warn")
    )

    using_generated_dataset = train_config.value["train_dataset_dropdown"] == "Use Generated Dataset (from memory)"
    training_model_save_path = f"pretrained/{train_config.value["train_model_name"]}.pt"

    os.makedirs("pretrained", exist_ok=True)
    training_output = None
    training_start_time = time.time()

    try:
        # Load dataset
        if using_generated_dataset:
            if generated_dataset is None:
                mo.md("❌ **Error:** No generated dataset found. Please generate one first.")
            training_dataset = generated_dataset
            training_dataset_path = None
        else:
            training_dataset_path = f"data/{train_config.value["train_dataset_dropdown"]}"
            training_dataset = torch.load(training_dataset_path)

        # Save generated dataset temporarily if needed
        if using_generated_dataset:
            temp_dataset_path = f"{train_config.value["train_dataset_dropdown"]}_temp.pt"
            torch.save(training_dataset, temp_dataset_path)
            dataset_path_for_training = temp_dataset_path
        else:
            dataset_path_for_training = training_dataset_path

        # Train the model
        train_model(
            dataset_path=dataset_path_for_training,
            n_channels=train_config.value["train_n_channels"],
            num_epochs=train_config.value["train_epochs"],
            patience=train_config.value["train_patience"],
            batch_size=train_config.value["train_batch_size"],
            lr=train_config.value["train_learning_rate"],
            model_save_path=training_model_save_path,
            bce_weight=train_config.value["train_bce_weight"],
            dice_weight=train_config.value["train_dice_weight"],
            base_filters=train_config.value["train_base_filters"]
        )

        # Cleanup
        if using_generated_dataset and os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)
            print("🧹 Removed temporary dataset file")

        del training_dataset

        training_time = time.time() - training_start_time

        # Return final result
        training_output = mo.vstack([
            mo.md(f"""
            ## 🎉 **Training Summary**
            - **Total Time:** {training_time/60:.1f} minutes  
            - **Model Type:** U-Net with {train_config.value["train_base_filters"]} base filters  
            - **Model Saved As:** `{training_model_save_path}`  
            - **Training completed successfully!**

            💡 **Next Step:** Use Section 4 below to evaluate your trained model!
            """)
        ])

    except Exception as e:
        training_output = mo.md(f"❌ **Training failed:** `{str(e)}`").callout(kind="danger")
        try:
            if using_generated_dataset and os.path.exists(temp_dataset_path):
                os.remove(temp_dataset_path)
        except:
            pass

    training_output
    return


@app.cell
def _(mo):
    mo.md("""## **Section 4: Model Evaluation**""")
    return


@app.cell
def _(mo):
    mo.md(
        """
    Configure evaluation parameters and select model and dataset:

    **Model Selection**: Choose a trained model from the pretrained folder.  
    **Dataset**: Use either the generated dataset from memory or load a saved dataset.  
    **Distance Threshold**: Controls how precisely predicted targets must match ground truth locations.
    """
    )
    return


@app.cell
def _(mo, os):
    # Select trained model and dataset for performance evaluation
    # Distance threshold determines how close predictions must be to ground truth targets
    # Base filters must match the training configuration of the selected model

    # Model selection
    eval_available_models = ["Select model..."] + [
        f for f in os.listdir("pretrained") if f.endswith('.pt')
    ] if os.path.exists("pretrained") else ["Select model..."]

    eval_model_dropdown = mo.ui.dropdown(
        options=eval_available_models,
        value=eval_available_models[0],
        label="Model to Evaluate"
    )

    # Dataset selection
    eval_available_datasets = (
        ["Use Generated Dataset (from memory)"] + 
        ["Select dataset..."] + 
        [f for f in os.listdir("data") if f.endswith('.pt')] if os.path.exists("data") else ["Select dataset..."]
    )

    eval_dataset_dropdown = mo.ui.dropdown(
        options=eval_available_datasets,
        value="Select dataset...",
        label="Evaluation Dataset"
    )

    # Evaluation parameters
    eval_distance_threshold = mo.ui.slider(
        start=1.0, stop=15.0, value=5.0, step=0.5,
        label="Distance Threshold (pixels)", show_value=True
    )

    eval_base_filters = mo.ui.slider(
        start=8, stop=64, value=16, step=8,
        label="Base Filters (match training)", show_value=True
    )

    # Compose the evaluation form
    eval_config = mo.md(
        """
        ### **Model & Dataset Selection**
        {eval_model_dropdown}  
        {eval_dataset_dropdown}  

        ### **Evaluation Parameters**
        {eval_distance_threshold}  
        {eval_base_filters}
        """
    ).batch(
        eval_model_dropdown=eval_model_dropdown,
        eval_dataset_dropdown=eval_dataset_dropdown,
        eval_distance_threshold=eval_distance_threshold,
        eval_base_filters=eval_base_filters
    ).form(
        submit_button_label="Run Evaluation",
        bordered=False,
        show_clear_button=True,
        clear_button_label="Reset"
    )

    eval_config
    return (eval_config,)


@app.cell
def _(comprehensive_evaluation, eval_config, generated_dataset, mo, os, torch):
    mo.stop(
        eval_config.value is None,
        mo.md("Please configure evaluation settings and click '🔍 Run Evaluation'").callout(kind="warn")
    )

    # Validate selections
    if eval_config.value["eval_model_dropdown"] in ["Select model...", None]:
        mo.stop(True, mo.md("❌ Please select a model to evaluate").callout(kind="danger"))

    if eval_config.value["eval_dataset_dropdown"] in ["Select dataset...", None]:
        mo.stop(True, mo.md("❌ Please select a dataset for evaluation").callout(kind="danger"))

    evaluation_output = None

    try:
        # Determine model and dataset paths
        using_generated_dataset_eval = eval_config.value["eval_dataset_dropdown"] == "Use Generated Dataset (from memory)"
        evaluation_model_path = f"pretrained/{eval_config.value['eval_model_dropdown']}"

        # Load dataset
        if using_generated_dataset_eval:
            if generated_dataset is None:
                mo.stop(True, mo.md("❌ **Error:** No generated dataset found. Please generate one first.").callout(kind="danger"))
            evaluation_dataset = generated_dataset
            evaluation_dataset_path = None
        else:
            evaluation_dataset_path = f"data/{eval_config.value['eval_dataset_dropdown']}"
            if not os.path.exists(evaluation_dataset_path):
                mo.stop(True, mo.md(f"❌ **Error:** Dataset file not found: `{evaluation_dataset_path}`").callout(kind="danger"))
            evaluation_dataset = torch.load(evaluation_dataset_path)

        # Save generated dataset temporarily if needed
        if using_generated_dataset_eval:
            temp_eval_dataset_path = f"{evaluation_dataset_path}_temp.pt"
            torch.save(evaluation_dataset, temp_eval_dataset_path)
            dataset_path_for_evaluation = temp_eval_dataset_path
        else:
            dataset_path_for_evaluation = evaluation_dataset_path

        # Verify model file exists
        if not os.path.exists(evaluation_model_path):
            mo.stop(True, mo.md(f"❌ **Error:** Model file not found: `{evaluation_model_path}`").callout(kind="danger"))

        print(f"🔍 Starting evaluation...")
        print(f"Model: {evaluation_model_path}")
        print(f"Dataset: {'Generated Dataset (in memory)' if using_generated_dataset_eval else evaluation_dataset_path}")
        print(f"Distance threshold: {eval_config.value['eval_distance_threshold']} pixels")

        # Run evaluation
        evaluation_results = comprehensive_evaluation(
            model_path=evaluation_model_path,
            dataset_path=dataset_path_for_evaluation,
            base_filter_size=eval_config.value["eval_base_filters"],
            marimo_var=True,
            distance_threshold=eval_config.value["eval_distance_threshold"]
        )

        # Extract results
        if evaluation_results and 'spatial_plots' in evaluation_results:
            results_plot = evaluation_results['spatial_plots']
        else:
            results_plot = mo.md("No evaluation results available. Please check the evaluation logs for details.")

        # Cleanup
        if using_generated_dataset_eval and os.path.exists(temp_eval_dataset_path):
            os.remove(temp_eval_dataset_path)
            print("🧹 Removed temporary evaluation dataset file")

        del evaluation_dataset

        eval_summary = ""

        # Add spatial evaluation metrics if available
        if evaluation_results and 'spatial_results' in evaluation_results:
            spatial_metrics = evaluation_results['spatial_results']
            eval_summary += f"""

        ## **Spatial Performance Metrics**
        - **Precision:** {spatial_metrics['precision']:.3f} *(True Positives / (True Positives + False Positives))*
        - **Recall:** {spatial_metrics['recall']:.3f} *(True Positives / (True Positives + False Negatives))*  
        - **F1-Score:** {spatial_metrics['f1_score']:.3f} *(Harmonic mean of Precision and Recall)*
        - **True Positives:** {spatial_metrics['total_true_positives']} *(Correctly detected targets)*
        - **False Positives:** {spatial_metrics['total_false_positives']} *(Incorrectly detected targets)*
        - **False Negatives:** {spatial_metrics['total_false_negatives']} *(Missed targets)*
        - **Samples Evaluated:** {spatial_metrics['num_samples']}
        - **Mean Match Distance:** {spatial_metrics.get('mean_match_distance', 'N/A'):.2f} pixels
        """

        # Return final result
        evaluation_output = mo.vstack([
            mo.md(eval_summary),
            mo.md("## **Detailed Spatial Performance Analysis:**"),
            mo.as_html(results_plot) if results_plot is not None else mo.md("*No evaluation visualization available.*")
        ])

    except Exception as e:
        evaluation_output = mo.md(f"❌ **Evaluation failed:** `{str(e)}`").callout(kind="danger")
        try:
            if using_generated_dataset_eval and os.path.exists("temp_dataset_for_evaluation.pt"):
                os.remove("temp_dataset_for_evaluation.pt")
        except:
            pass

    evaluation_output
    return

@app.cell
def _(mo):
    mo.md(
        """
    ## **Quick Reference**

    ### Parameter Guidelines:

    **Sea Clutter Parameters:**\t
    - **Mean Power**: Controls overall clutter intensity\t
    - **Shape Parameter**: Controls clutter distribution, lower = more spiky\t
    - **AR Coefficient**: Controls temporal correlation, higher = more correlated\t
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
    ## **Usage Tips & Best Practices**

    ### **Getting Started:**
    1. **Design your scenario in Section 1**: Adjust radar, clutter, and target parameters to create realistic simulation characteristics
    2. **Visualize before generating**: Use the interactive plots to understand how parameters affect sea clutter patterns and target visibility
    3. **Generate datasets**: Section 2 uses your Section 1 configuration - validate parameters first
    4. **Train models in Section 3**: Use the generated dataset to train U-Net models for sea clutter suppression

    ### **Model Training Guidelines:**
    - **Training Monitoring**: Watch console output for loss curves, validation metrics, and early stopping
    - **Hyperparameter Tuning**: Start with default values, then adjust learning rate and batch size based on training behavior
    - **Model Architecture**: Base filters control model capacity - increase for complex scenarios, decrease for faster training

    ### **Performance Optimization:**
    - **Dataset Size**: Larger datasets (1000+ samples) generally improve model generalization
    - **Multi-frame vs Single-frame**: Multi-frame models capture temporal dynamics but require more memory

    ### **Troubleshooting:**
    - **Out of Memory**: Reduce batch size or switch to smaller model architecture
    - **Poor Convergence**: Lower learning rate, increase patience, or check data quality
    - **Overfitting**: Add more diverse training data or reduce model complexity
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ##**Credits, Contributions & Resources**

    ###**Project Information**
    This interactive sea clutter suppression notebook and the underlying deep learning framework were developed by **Pepijn Lens** as part of a bachelor thesis and internship at TNO.

    ### **Repository & Collaboration**
    - **GitHub Repository**: [github.com/pepijn-lens/SeaClutterSuppression](https://github.com/pepijn-lens/SeaClutterSuppression)
    - **Report Issues**: Use GitHub Issues for bug reports and feature requests
    - **Contribute**: Pull requests welcome!
    - **Documentation**: Comprehensive README and code documentation available in the repository

    ### **Contact & Support**
    - **Email**: pepijn.lens@tno.nl
    - **Discussions**: Use GitHub Discussions for questions

    ### **Citation**
    If you use this work in your research, please cite the repository.

    ---
    *Last updated: 20th of June 2025 • Version compatible with Marimo 0.13.15+*
    """
    )
    return


if __name__ == "__main__":
    app.run()
