import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import random
    return mo, random


@app.cell
def _():
    from src import simulate_sequence_with_realistic_targets
    import sea_clutter
    return sea_clutter, simulate_sequence_with_realistic_targets


@app.cell(hide_code=True)
def _(
    ar_coeff,
    bragg_offset,
    bragg_power,
    bragg_width_hz,
    mean_power,
    n_targets,
    random,
    sea_clutter,
    simulate_sequence_with_realistic_targets,
    wave_speed_mps,
):
    rp = sea_clutter.RadarParams()
    cp = sea_clutter.ClutterParams(mean_power_db=mean_power.value, ar_coeff=ar_coeff.value, bragg_offset_hz=bragg_offset.value, bragg_width_hz=bragg_width_hz.value, bragg_power_rel=bragg_power.value, wave_speed_mps=wave_speed_mps.value)
    sp = sea_clutter.SequenceParams(n_frames=5)  # Longer sequence to see movement
    min_range = int(round(20 * sp.n_frames/sp.frame_rate_hz/2))  # Minimum range for targets
    max_range = int(round(rp.n_ranges * rp.range_resolution - (20 * sp.n_frames/sp.frame_rate_hz)/2))  # Maximum range for targets

    targets = [sea_clutter.create_realistic_target(sea_clutter.TargetType.FIXED, random.randint(min_range, max_range), rp) for _ in range(n_targets.value)]

    # Print target information
    print("Simulating targets:")
    for i, tgt in enumerate(targets):
        print(f"  {i+1}. {tgt.target_type.value}: Range {tgt.rng_idx*rp.range_resolution:.0f}m, "
              f"Initial velocity {tgt.current_velocity_mps:.1f} m/s")

    rdm_list = simulate_sequence_with_realistic_targets(rp, cp, sp, targets, random_roll=False)
    interval_ms = int(1000.0 / sp.frame_rate_hz)
    sea_clutter.animate_sequence(rdm_list, rp, interval_ms=interval_ms, save_path=None)
    return


@app.cell(hide_code=True)
def _(mo):
    n_targets = mo.ui.slider(start=0, stop=20, value=5, label="Amount of targets: ")
    mean_power = mo.ui.slider(start=-30, stop=30, value= 0, label="Mean clutter power: ")
    ar_coeff = mo.ui.slider(start=0, stop=0.99, value= 0.95, step=0.01, label="Ar coefficient: ")
    bragg_offset = mo.ui.slider(start=0, stop=200, value= 45, label="Bragg offset in hz: ")
    bragg_width_hz = mo.ui.slider(start=0, stop=10, value= 4, label="Bragg width: ")
    bragg_power = mo.ui.slider(start=-20, stop=20, value= 10, label="Bragg power: ")
    wave_speed_mps = mo.ui.slider(start=0, stop=20, value= 4, label="Wave speed in mps: ")


    # Display all the sliders
    mo.vstack([
        n_targets,
        mean_power,
        ar_coeff,
        bragg_offset,
        bragg_width_hz,
        bragg_power,
        wave_speed_mps
    ])
    return (
        ar_coeff,
        bragg_offset,
        bragg_power,
        bragg_width_hz,
        mean_power,
        n_targets,
        wave_speed_mps,
    )


if __name__ == "__main__":
    app.run()
