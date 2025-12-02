# Configuration Guide

## Command-Line Options

### Basic Simulation Command

```bash
gwsim simulate config.yaml
```

### Common Flags

#### `--overwrite`

By default, gwsim does not overwrite existing output files. If a file already exists, the tool will raise an error and halt execution.

To force overwriting of existing files, use the `--overwrite` flag:

```bash
gwsim simulate config.yaml --overwrite
```

#### `--dry-run`

Test your configuration without generating data:

```bash
gwsim simulate config.yaml --dry-run
```

This validates the configuration and shows what would be generated without actually creating files.

#### `--metadata`

Generate metadata files along with the data (automatically enabled by default):

```bash
gwsim simulate config.yaml --metadata
```

Metadata files contain complete provenance information including:
- Simulator configuration
- Random number generator state
- Output file names
- Version information

## Configuration File Structure

The configuration file uses YAML format with the following main sections:

### Globals

Top-level shared parameters used across all simulators:

```yaml
globals:
  working-directory: .
  sampling-frequency: 4096
  duration: 4096
  start-time: 1577491218
  output-directory: ./output
  metadata-directory: ./metadata
  seed-base: 42
```

**Key parameters:**
- `working-directory`: Base directory for operations
- `sampling-frequency`: Sample rate in Hz
- `duration`: Duration of each segment in seconds
- `start-time`: GPS start time
- `output-directory`: Where to save generated data files
- `metadata-directory`: Where to save metadata files
- `seed-base`: Base random seed for reproducibility

### Simulators

List of simulators to run, each with configuration:

```yaml
simulators:
  noise:
    class: PyCBCStationaryGaussianNoiseSimulator
    arguments:
      psd: aLIGOZeroDetHighPower
      detectors:
        - H1
      seed: 0
      max_samples: 22
    output:
      file_name: "{{ detector }}-NOISE-{{ start_time }}-{{ duration }}.gwf"
      arguments:
        channel: "{{ detector }}:STRAIN"
```

**Simulator properties:**
- `class`: Fully qualified class name of the simulator
- `arguments`: Parameters passed to the simulator
- `output.file_name`: Template for output file naming (supports Jinja2 syntax)
- `output.arguments`: Channel naming and other output metadata

## Template Variables

Use Jinja2-style templates in configuration values:

```yaml
globals:
  detector: "H1"
  network: ["H1", "L1", "V1"]

simulators:
  noise:
    arguments:
      detectors: "{{ network }}"
    output:
      file_name: "{{ detector }}-NOISE-{{ start_time }}-{{ duration }}.gwf"
```

**Common variables:**
- `{{ start_time }}`: GPS start time from globals
- `{{ duration }}`: Segment duration from globals
- `{{ detector }}`: Current detector being processed
- `{{ seed_base }}`: Random seed base from globals

## Checkpointing

gwsim automatically creates checkpoints during long simulations. If a process is interrupted:

1. A `checkpoint.json` file is saved in the working directory
2. Rerun the same command to resume from the last checkpoint
3. The tool automatically detects and continues from where it left off

```bash
# Start simulation
gwsim simulate config.yaml

# If interrupted (Ctrl+C, crash, etc.), resume with same command
gwsim simulate config.yaml
```

The checkpoint contains:
- Simulator state
- Progress information
- Already-generated file tracking

## Configuration Inheritance

Extend base configurations to reduce duplication:

```yaml
# base_config.yaml
globals:
  sampling-frequency: 4096
  duration: 4096
  output-directory: ./output

simulators:
  noise:
    class: PyCBCStationaryGaussianNoiseSimulator
```

```yaml
# specific_config.yaml
inherits: base_config.yaml

globals:
  start-time: 1200000000  # Override base value

simulators:
  noise:
    arguments:
      detectors: ["L1"]  # Extend base configuration
```

## Best Practices

1. **Use templates**: Leverage Jinja2 templates for dynamic configuration
2. **Set seeds**: Always set `seed-base` for reproducibility
3. **Check space**: Ensure sufficient disk space before long runs
4. **Use dry-run**: Test configurations with `--dry-run` before full simulation
5. **Organize outputs**: Use descriptive `output-directory` and `metadata-directory` names
