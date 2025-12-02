# Quick Start

This guide will get you running gwsim simulations in minutes.

## 1. Create a Simple Configuration

Create a file called `basic_config.yaml`:

```yaml
globals:
  working-directory: "./gwsim_output"
  sampling-frequency: 4096
  duration: 4
  start-time: 1000000000
  output-directory: "{ working-directory }/data"
  metadata-directory: "{ working-directory }/metadata"
  seed-base: 42

simulators:
  basic_noise:
    class: PyCBCStationaryGaussianNoiseSimulator
    arguments:
      psd: aLIGOZeroDetHighPower
      seed: "{ seed-base }"
      detectors: ["H1"]
      max_samples: 1
    output:
      file_name: "H1-NOISE-{ start-time }-{ duration }.gwf"
      arguments:
        channel: "H1:STRAIN"
```

## 2. Run Your First Simulation

```bash
gwsim simulate basic_config.yaml
```

You should see output like:
```
[INFO] Loading configuration from basic_config.yaml
[INFO] Validating simulation plan...
[INFO] Starting simulation: basic_noise
[INFO] Generating batch 0/1 for basic_noise
[INFO] Simulation completed successfully
[INFO] Output files: ./gwsim_output/data/
[INFO] Metadata files: ./gwsim_output/metadata/
```

## 3. Check the Output

Your working directory will contain:

```
gwsim_output/
├── data/
│   └── H1-NOISE-1000000000-4.gwf
└── metadata/
    ├── basic_noise-0.metadata.yaml
    └── index.yaml
```

### Data File
The GWF file contains the simulated strain data that can be read with any gravitational wave analysis software.

### Metadata File
The metadata file contains everything needed to reproduce this exact simulation:

```yaml
simulator_name: basic_noise
batch_index: 0
simulator_config:
  class_: PyCBCStationaryGaussianNoiseSimulator
  arguments:
    psd: aLIGOZeroDetHighPower
    seed: 42
    detectors: ["H1"]
    max_samples: 1
pre_batch_state:
  counter: 0
  rng_state: {...}
output_files:
  - H1-NOISE-1000000000-4.gwf
versions:
  gwsim: 0.1.0
  pycbc: 2.9.0
source: config
```

## 4. Explore Different Simulators

### Signal Simulation

Create `signal_config.yaml`:

```yaml
globals:
  working-directory: "./gwsim_signals"
  sampling-frequency: 4096
  duration: 4
  start-time: 1000000000
  output-directory: "{ working-directory }/data"
  metadata-directory: "{ working-directory }/metadata"

simulators:
  cbc_signals:
    class: SignalSimulator
    arguments:
      population_file: "examples/cbc_population.csv"
      waveform_model: "IMRPhenomD"
      detectors: ["H1", "L1"]
      minimum_frequency: 20.0
      max_samples: 10
    output:
      file_name: "CBC-SIGNALS-{ start-time }-{ duration }.gwf"
      arguments:
        channel: "{ detector }:STRAIN"
```

### Multi-Detector Correlated Noise

Create `correlated_noise_config.yaml`:

```yaml
globals:
  working-directory: "./gwsim_correlated"
  sampling-frequency: 4096
  duration: 4
  start-time: 1000000000
  output-directory: "{ working-directory }/data"
  metadata-directory: "{ working-directory }/metadata"

simulators:
  correlated_noise:
    class: CorrelatedNoiseSimulator
    arguments:
      psd_file: "examples/ET_psd.txt"
      csd_file: "examples/ET_csd.txt"
      detectors: ["E1", "E2", "E3"]
      seed: 12345
      max_samples: 1
    output:
      file_name: "ET-CORR-NOISE-{ start-time }-{ duration }.gwf"
      arguments:
        channel: "{ detector }:STRAIN"
```

## 5. Advanced Features

### Template Variables
Use Jinja2-style templates in your configuration:

```yaml
globals:
  detector: "H1"
  network: ["H1", "L1", "V1"]

simulators:
  network_noise:
    arguments:
      detectors: "{ network }"
    output:
      file_name: "{ detector }-NOISE-{ start-time }-{ duration }.gwf"
      arguments:
        channel: "{ detector }:STRAIN"
```

### Configuration Inheritance
Extend base configurations:

```yaml
# base_config.yaml
globals:
  sampling-frequency: 4096
  duration: 4

simulators:
  base_noise:
    class: PyCBCStationaryGaussianNoiseSimulator
    arguments:
      psd: aLIGOZeroDetHighPower

# specific_config.yaml
inherits: base_config.yaml

globals:
  start-time: 1200000000  # Override base value

simulators:
  base_noise:
    arguments:
      detectors: ["L1"]  # Extend base configuration
```

### Checkpointing
For long simulations, gwsim automatically creates checkpoints:

```bash
# Start simulation
gwsim simulate long_config.yaml

# If interrupted, resume from checkpoint
gwsim simulate long_config.yaml  # Automatically resumes
```

### Dry Run
Test your configuration without generating data:

```bash
gwsim simulate --dry-run config.yaml
```

## Next Steps

- [Configuration Guide](configuration.md) - Complete configuration reference
- [Examples](examples.md) - Real-world usage examples
- [API Reference](../reference/index.md) - Programmatic usage
