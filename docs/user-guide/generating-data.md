# Generating Data

This guide demonstrates how to use gwsim to create realistic mock data for gravitational wave detectors.

## Generating Detector Noise

Detector noise can be generated using configuration files from the `examples/noise` directory.

### Example: Einstein Telescope Triangular Configuration

Here's an example for producing noise data for the Einstein Telescope (ET) in triangular configuration (Meuse-Rhine Euregion):

```yaml
globals:
  working-directory: .
  simulator-arguments:
    sampling-frequency: 4096
    max-samples: 22
    duration: 4096
    start_time: 1577491218
  output-arguments: {}
  output-directory: ET_Triangle_EMR_noise
  metadata-directory: ET_Triangle_EMR_noise/metadata

simulators:
  noise:
    class: PyCBCStationaryGaussianNoiseSimulator
    arguments:
      psd: ET_10_full_cryo_psd
      detectors:
        - E1_Triangle_EMR
        - E2_Triangle_EMR
        - E3_Triangle_EMR
      seed: 0
    output:
      file_name: "E-{{ detector }}_STRAIN_NOISE-{{ start_time }}-{{ duration }}.gwf"
      arguments:
        channel: STRAIN
```

This configuration generates 22 frame files per detector (E1, E2, E3), with each file covering 4096 seconds (slightly more than 24 hours starting on 1 January 2030).

Run the simulation:

```bash
gwsim simulate ET_Triangle_EMR_noise_config.yaml
```

#### Storage Requirements

Each GWF file is approximately 125 MB. For three detectors with 22 files each:
- **Data files**: ~8.25 GB
- **Metadata**: ~50 MB
- **Total**: ~8.3 GB

## Generating CBC Signals

Compact Binary Coalescence (CBC) signals can be generated using configuration files from the `examples/signal` directory.

### Binary Black Hole (BBH) Signals

Example configuration for BBH signals:

```yaml
globals:
  working-directory: .
  simulator-arguments:
    sampling-frequency: 4096
    max-samples: 22
    duration: 4096
    start_time: 1577491218
  output-arguments: {}
  output-directory: ET_Triangle_EMR_BBH
  metadata-directory: ET_Triangle_EMR_BBH/metadata

simulators:
  signals:
    class: CBCSignalSimulator
    arguments:
      population_file: bbh_population.h5
      waveform_model: IMRPhenomXPHM
      waveform_arguments:
        earth_rotation: true
        time_dependent_timedelay: true
      minimum_frequency: 2
      detectors:
        - E1_Triangle_EMR
        - E2_Triangle_EMR
        - E3_Triangle_EMR
    output:
      file_name: "E-{{ detector }}_STRAIN_BBH-{{ start_time }}-{{ duration }}.gwf"
      arguments:
        channel: STRAIN
```

Run the simulation:

```bash
gwsim simulate ET_Triangle_EMR_BBH_config.yaml
```

### Binary Neutron Star (BNS) Signals

For BNS signals, modify the configuration:

```yaml
simulators:
  signals:
    arguments:
      population_file: bns_population.h5
      waveform_model: IMRPhenomPv2_NRTidalv2
```

## Using Different Detector Configurations

gwsim includes several pre-configured Einstein Telescope detector geometries:

### Triangular Configuration (Meuse-Rhine Euregion)
- `E1_Triangle_EMR`
- `E2_Triangle_EMR`
- `E3_Triangle_EMR`

### Triangular Configuration (Sardinia)
- `E1_Triangle_Sardinia`
- `E2_Triangle_Sardinia`
- `E3_Triangle_Sardinia`

### 2L Aligned Configuration
- `E1_2L_Aligned_Sardinia`
- `E2_2L_Aligned_EMR`

### 2L Misaligned Configuration
- `E1_2L_Misaligned_Sardinia`
- `E2_2L_Misaligned_EMR`

To use a specific configuration, update the `detectors` list in your configuration file:

```yaml
simulators:
  noise:
    arguments:
      detectors:
        - E1_2L_Aligned_Sardinia
        - E2_2L_Aligned_EMR
```

You don't need to include all detectors. For example, to generate only E1 data:

```yaml
simulators:
  noise:
    arguments:
      detectors:
        - E1_2L_Aligned_Sardinia
```

## Using Different Sensitivity Curves

Multiple Einstein Telescope sensitivity curves (PSD files) are available in `gwsim/detector/noise_curves/`.

To use a specific sensitivity curve:

```yaml
simulators:
  noise:
    arguments:
      psd: ET_15_HF_psd.txt
```

!!! note
    The detector geometries assume 10 km arms for triangular configurations and 15 km arms for 2L configurations. Choose sensitivity curves accordingly.

## Adjusting Dataset Duration

Control dataset length with three parameters:

```yaml
globals:
  start_time: 1577491218        # GPS start time
  duration: 4096                # Duration per frame file (seconds)

simulators:
  noise:
    arguments:
      max_samples: 22           # Number of consecutive files
```

**Total dataset duration:**

```
total_duration (seconds) = duration × max_samples
end_time = start_time + (duration × max_samples)
```

!!! tip
    Sampling frequencies are often powers of 2 for efficiency. Common choices:
    - 4096 Hz (standard for GW data analysis)
    - 2048 Hz
    - 16384 Hz (high-frequency instruments)

    Lowering sampling frequency reduces computation time but also reduces the highest resolvable frequency (Nyquist limit = sampling_frequency / 2).

## Resume Interrupted Simulations

If a simulation is interrupted, resume it by running the same command:

```bash
# Start
gwsim simulate config.yaml

# If interrupted, resume
gwsim simulate config.yaml
```

gwsim automatically detects and continues from the last checkpoint.

## Combining Data Types

To create realistic mock data, you may generate noise, signals, and glitches separately, then combine them:

```bash
gwsim simulate noise_config.yaml
gwsim simulate signal_config.yaml
gwsim simulate glitch_config.yaml
```

Then merge the files using GWpy (see [Reading Data](reading-data.md) for details).
