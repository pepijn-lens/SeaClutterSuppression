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

    
    return (
        comprehensive_evaluation,
        gc,
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
    # Interactive Sea Clutter Simulation & U-Net Training Notebook

    This notebook provides four main functionalities: \t
    1. **Parameter Adjustment & Real-time Visualization** - Adjust radar and clutter parameters and see sea clutter simulation \t
    2. **Dataset Generation** - Generate synthetic sea clutter datasets for training \t
    3. **U-Net Training** - Train deep learning models on the generated data\t
    """
    )


@app.cell
def _(mo):
    mo.md("""## **Section 1: Real-time Sea Clutter Visualization**""")


@app.cell
def _(mo):
    mo.md("""### Radar Parameters""")


@app.cell
def _(mo):
    # Radar parameter controls
    radar_prf = mo.ui.slider(start=1000, stop=10000, value=5000, step=100, label="PRF (Hz)")
    radar_n_pulses = mo.ui.slider(start=64, stop=512, value=128, step=64, label="Number of Pulses")
    radar_n_ranges = mo.ui.slider(start=64, stop=512, value=128, step=64, label="Number of Range Bins")

    mo.hstack([radar_prf, radar_n_pulses, radar_n_ranges])
    return radar_n_pulses, radar_n_ranges, radar_prf


@app.cell
def _(mo):
    mo.md("""### Clutter Parameters""")


@app.cell
def _(mo):
    # Clutter parameter controls
    clutter_mean_power = mo.ui.slider(start=0, stop=25, value=16, step=1, label="Mean Clutter Power (dB)")
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
    mo.md("""### Target Parameters""")


@app.cell
def _(mo):
    # Target parameter controls
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
    mo.md("""### Signal (SNR Control)""")


@app.cell
def _(mo):
    # SNR control parameters
    target_signal_power = mo.ui.slider(start=1, stop=30, value=20, step=1, label="Target Signal Power (dB)")
    noise_power = 0
    mo.hstack([target_signal_power])
    return noise_power, target_signal_power


@app.cell
def _(mo):
    mo.md("""### Sequence Parameters (for Multi-Frame Generation)""")


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

    return (frame_slider, slider_display)


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


@app.cell
def _(mo):
    mo.md("""## **Section 2: Dataset Generation**""")


@app.cell
def _(mo):
    # Dataset generation controls
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
        dataset_name=dataset_name
    ).form(
        submit_button_label="🚀 Generate Dataset",
        bordered=False,
        show_clear_button=True,
        clear_button_label="Reset"
    )

    dataset_config


@app.cell
def _(
    clutter_ar_coeff,
    clutter_bragg_offset,
    clutter_bragg_power,
    clutter_bragg_width,
    clutter_mean_power,
    clutter_shape_param,
    clutter_wave_speed,
    noise_power,
    np,
    mo,
    dataset_config,
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
def _(mo):
    mo.md("""## **Section 3: U-Net Training**""")


@app.cell
def _(mo):
    mo.md("""Configure U-Net model and training parameters:""")


@app.cell
def _(mo, os):
    # UI Controls
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

@app.cell
def _(mo, train_config, models):
    mo.stop(train_config.value is None)
    # Preview model parameter counts
    base_filters = train_config.value["train_base_filters"]
    n_channels = train_config.value["train_n_channels"]
    temp_model = models.UNet(n_channels=n_channels, base_filters=base_filters)
    trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in temp_model.parameters())
    param_info = mo.md(f"""
    ### **Model Stats**
    - Trainable Parameters: **{trainable_params:,}**  
    - Total Parameters: **{total_params:,}**  
    - Estimated Model Size: **{(trainable_params * 4) / (1024**2):.1f} MB**
    """)
    param_info

@app.cell
def _(
    comprehensive_evaluation,
    generated_dataset,
    mo,
    os,
    time,
    torch,
    train_config,
    train_model, 
):

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
            temp_dataset_path = "temp_dataset_for_training.pt"
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

        # Run evaluation
        results_plot = comprehensive_evaluation(
            model_path=training_model_save_path,
            dataset_path=dataset_path_for_training,
            base_filter_size=train_config.value["train_base_filters"],
            marimo_var=True
        )

        if results_plot is None:
            results_plot = mo.md("No evaluation results available. Please check the training logs for details.")

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
            """),
            mo.md("## **Model Evaluation Results:**"),
            mo.as_html(results_plot) if results_plot is not None else mo.md("*No evaluation results available.*")
        ])

    except Exception as e:
        training_output = mo.md(f"❌ **Training failed:** `{str(e)}`").callout(kind="error")
        try:
            if using_generated_dataset and os.path.exists("temp_dataset_for_training.pt"):
                os.remove("temp_dataset_for_training.pt")
        except:
            pass

    return training_output



@app.cell
def _(mo):
    mo.md(
        """
    ## **Quick Reference**

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

@app.cell
def _(mo):
    mo.md(
        """
    ## **Credits, Contributions & Resources**

    ### **Project Information**
    This interactive sea clutter suppression notebook and the underlying deep learning framework were developed by **Pepijn Lens** as part of a bachelor thesis and internship at TNO.

    ### **Repository & Collaboration**
    - ** GitHub Repository**: [github.com/pepijn-lens/SeaClutterSuppression](https://github.com/pepijn-lens/SeaClutterSuppression)
    - ** Report Issues**: Use GitHub Issues for bug reports and feature requests
    - ** Contribute**: Pull requests welcome!
    - ** Documentation**: Comprehensive README and code documentation available in the repository

    ### **Contact & Support**
    - ** Email**: pepijn.lens@tno.nl
    - ** Discussions**: Use GitHub Discussions for questions

    ### **Citation**
    If you use this work in your research, please cite the repository.

    ---
    *Last updated: 20th of June 2025 • Version compatible with Marimo 0.13.15+*
    """
    )


if __name__ == "__main__":
    app.run()
