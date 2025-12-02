# Examples

This page provides links to and descriptions of example configuration files for common gwsim use cases.

## Overview

All example configurations in the `examples/` directory generate 22 frame files per detector, each 4096 seconds long, sampled at 4096 Hz, resulting in slightly more than 24 hours of data starting from 1 January 2030.

For instructions on modifying dataset duration or other properties, see the [Generating Data](generating-data.md) guide.

## Noise Generation

Example configurations for generating detector noise with various configurations and sensitivities.

### Einstein Telescope - Triangular (EMR)

Generate uncorrelated and correlated noise for the Einstein Telescope triangular configuration (Meuse-Rhine Euregion):

- **Uncorrelated noise**: `examples/noise/uncorrelated_ET_triangle_EMR/`
- **Correlated noise**: `examples/noise/correlated_ET_triangle_EMR/`

```bash
gwsim simulate examples/noise/pycbc_gaussian_noise_simulator/config.yaml
```

### Einstein Telescope - Triangular (Sardinia)

Generate noise for the Sardinia site:

- **Uncorrelated noise**: `examples/noise/uncorrelated_ET_triangle_Sardinia/`
- **Correlated noise**: `examples/noise/correlated_ET_triangle_Sardinia/`

### Einstein Telescope - 2L Aligned

Generate noise for the 2L aligned configuration (E1 in Sardinia, E2 in EMR):

- **Uncorrelated noise**: `examples/noise/uncorrelated_ET_2L_aligned/`

### Einstein Telescope - 2L Misaligned

Generate noise for the 2L misaligned configuration:

- **Uncorrelated noise**: `examples/noise/uncorrelated_ET_2L_misaligned/`

## CBC Signals

Example configurations for generating gravitational wave signals from compact binary coalescences.

### Binary Black Hole (BBH)

Generate BBH signals for various ET configurations:

- **Triangular EMR**: `examples/signal/cbc/bbh_triangular_EMR/config.yaml`
- **Triangular Sardinia**: `examples/signal/cbc/bbh_triangular_Sardinia/config.yaml`
- **2L Aligned**: `examples/signal/cbc/bbh_2L_aligned/config.yaml`
- **2L Misaligned**: `examples/signal/cbc/bbh_2L_misaligned/config.yaml`

Run a BBH simulation:

```bash
gwsim simulate examples/signal/cbc/bbh_triangular_EMR/config.yaml
```

### Binary Neutron Star (BNS)

Generate BNS signals (with tidal deformability) for various configurations:

- **Triangular EMR**: `examples/signal/cbc/bns_triangular_EMR/config.yaml`
- **Triangular Sardinia**: `examples/signal/cbc/bns_triangular_Sardinia/config.yaml`
- **2L Aligned**: `examples/signal/cbc/bns_2L_aligned/config.yaml`
- **2L Misaligned**: `examples/signal/cbc/bns_2L_misaligned/config.yaml`

Run a BNS simulation:

```bash
gwsim simulate examples/signal/cbc/bns_triangular_EMR/config.yaml
```

## Glitch Generation

Example configurations for generating detector glitches (transient artifacts):

- **Triangular EMR**: `examples/glitch/triangular_EMR/config.yaml`
- **Triangular Sardinia**: `examples/glitch/triangular_Sardinia/config.yaml`
- **2L Aligned**: `examples/glitch/2L_aligned/config.yaml`
- **2L Misaligned**: `examples/glitch/2L_misaligned/config.yaml`

Run a glitch simulation:

```bash
gwsim simulate examples/glitch/triangular_EMR/config.yaml
```

## Using Examples as Templates

To create your own configuration, copy and modify an example:

```bash
# Copy an example as a template
cp examples/noise/pycbc_gaussian_noise_simulator/config.yaml my_config.yaml

# Edit for your needs
nano my_config.yaml

# Run your custom simulation
gwsim simulate my_config.yaml
```

## Example Configuration Structure

Here's the basic structure of an example configuration file:

```yaml
globals:
  working-directory: .
  simulator-arguments:
    sampling-frequency: 4096
    duration: 4096
    max-samples: 22
  output-directory: ./output
  metadata-directory: ./metadata

simulators:
  main_simulator:
    class: SimulatorClassName
    arguments:
      # Simulator-specific arguments
    output:
      file_name: "{{ detector }}-TYPE-{{ start_time }}-{{ duration }}.gwf"
      arguments:
        channel: STRAIN
```

## Customization Guide

### Adjust Duration

Modify `max-samples` and `duration` in globals:

```yaml
globals:
  duration: 86400        # 1 day per file
  max-samples: 7         # 7 files = 7 days
```

### Change Sensitivity

Update the `psd` or `psd_file` parameter:

```yaml
simulators:
  noise:
    arguments:
      psd: aLIGOZeroDetHighPower  # or ET_10_full_cryo_psd, etc.
```

### Use Different Detectors

Modify the `detectors` list:

```yaml
simulators:
  noise:
    arguments:
      detectors:
        - E1_Triangle_Sardinia
        - E2_Triangle_Sardinia
        - E3_Triangle_Sardinia
```

### Adjust Random Seed

Set `seed-base` in globals for reproducibility:

```yaml
globals:
  seed-base: 12345
```

## Storage Estimates

For reference, typical storage requirements:

- **Noise**: ~125 MB per file, ~8.3 GB for 3 detectors, 24 hours
- **Signals**: Variable depending on waveform complexity, typically similar to noise
- **Glitches**: Variable depending on number and duration

## Next Steps

- See [Generating Data](generating-data.md) for detailed instructions
- See [Configuration Guide](configuration.md) for all available options
- See [Reading Data](reading-data.md) to analyze generated files
