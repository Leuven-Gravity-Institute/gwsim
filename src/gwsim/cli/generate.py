"""
A sub-command to handle data generation.
"""

from __future__ import annotations

import atexit
import logging
import signal
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import typer
from tqdm import tqdm

from ..generator.base import Generator
from .config import get_config_value, load_config, process_config, resolve_class_path
from .utils import get_file_name_from_template, handle_signal, import_attribute, save_file_safely

logger = logging.getLogger("gwsim")


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""

    file_name_template: str
    output_directory: Path
    metadata_directory: Path
    output_arguments: dict[str, Any]
    overwrite: bool
    metadata: bool
    checkpoint_file: Path
    checkpoint_file_backup: Path


def get_generator(generator_name: str, generator_config: dict, resolved_class_path: str) -> Generator:
    """Get the generator from a dictionary of configuration.

    Args:
        generator_name (str): Name of the generator for logging.
        generator_config (dict): Generator-specific configuration.
        resolved_class_path (str): Fully resolved class path.

    Raises:
        KeyError: If 'class' is not in generator_config.
        KeyError: If 'arguments' is not in generator_config.

    Returns:
        Generator: An instance of a generator.
    """
    if "class" not in generator_config:
        raise KeyError(
            f"Failed to initialize generator '{generator_name}'. "
            "'class' is not found in the generator configuration."
        )

    if "arguments" not in generator_config:
        raise KeyError(
            f"Failed to initialize generator '{generator_name}'. "
            "'arguments' is not found in the generator configuration."
        )

    generator_cls = import_attribute(resolved_class_path)
    generator = generator_cls(**generator_config["arguments"])

    # Print the information.
    logger.info("Generator '%s' class: %s", generator_name, resolved_class_path)
    logger.info("Generator '%s' arguments: %s", generator_name, generator_config["arguments"])

    return generator


def clean_up_generate(checkpoint_file: Path, checkpoint_file_backup: Path) -> None:
    """Clean-up function to be called when the signal is received.

    Args:
        checkpoint_file (Path): Path to the checkpoint file.
        checkpoint_file_backup (Path): Path to the backup checkpoint file.

    Returns:
        None
    """

    # Check whether a backup checkpoint file exists.
    if checkpoint_file_backup.is_file():
        logger.warning("Interrupted while creating a checkpoint file. Restoring the checkpoint file from a backup.")

        try:
            checkpoint_file_backup.rename(checkpoint_file)
            logger.info("Checkpoint file restored from backup.")
        except (OSError, FileNotFoundError, PermissionError) as e:
            logger.error("Failed to restore checkpoint from backup: %s", e)
            logger.warning("Continuing without checkpoint restoration.")
    else:
        logger.debug("No backup checkpoint file found. Nothing to clean up.")


def setup_signal_handlers(checkpoint_file: Path, checkpoint_file_backup: Path) -> None:
    """Set up signal handlers for graceful shutdown.

    Args:
        checkpoint_file (Path): Path to checkpoint file.
        checkpoint_file_backup (Path): Path to backup checkpoint file.
    """
    clean_up_fn = partial(clean_up_generate, checkpoint_file, checkpoint_file_backup)

    # Register clean-up for normal exit
    atexit.register(clean_up_fn)

    # Register cleanup for Ctrl+C and termination
    signal.signal(signal.SIGINT, handle_signal(clean_up_fn))
    signal.signal(signal.SIGTERM, handle_signal(clean_up_fn))


def process_batch(
    generator: Generator, batch: object, config: BatchProcessingConfig, detector: str | None = None
) -> None:
    """Process a single batch of generated data.

    Args:
        generator: The data generator.
        batch: Generated data batch.
        config: Configuration object containing all processing parameters.
        detector: Optional detector name for multi-detector scenarios.
    """
    # Get the file name from template, handling detector placeholders
    file_name_template = config.file_name_template
    if detector and "{detector}" in file_name_template:
        file_name_template = file_name_template.replace("{detector}", detector)

    file_name = Path(get_file_name_from_template(file_name_template, generator))
    batch_file_name = config.output_directory / file_name

    # Check whether the file exists
    if batch_file_name.is_file():
        if not config.overwrite:
            raise FileExistsError("Exiting. Use --overwrite to allow overwriting.")
        logger.warning("%s already exists. Overwriting the existing file.", batch_file_name)

    # Prepare output arguments, handling detector placeholders in channel names
    output_arguments = config.output_arguments.copy()
    if detector and "channel" in output_arguments:
        channel = output_arguments["channel"]
        if "{detector}" in channel:
            output_arguments["channel"] = channel.replace("{detector}", detector)

    # Handle list of channels with detector placeholders
    if detector and "channels" in output_arguments:
        channels = output_arguments["channels"]
        if isinstance(channels, list):
            output_arguments["channels"] = [
                ch.replace("{detector}", detector) if "{detector}" in ch else ch for ch in channels
            ]

    # Write the batch of data to file.
    generator.save_batch(batch, batch_file_name, overwrite=config.overwrite, **output_arguments)

    # Write the metadata to file.
    if config.metadata:
        metadata_file_name = config.metadata_directory / file_name.with_suffix(".json")
        generator.save_metadata(file_name=metadata_file_name, overwrite=config.overwrite)

    # Update the state if the data is saved successfully.
    generator.update_state()

    # Create checkpoint file.
    save_file_safely(
        file_name=config.checkpoint_file,
        backup_file_name=config.checkpoint_file_backup,
        save_function=generator.save_state,
        overwrite=True,
    )


@dataclass
class GenerationSetup:
    """Configuration setup for generation process."""

    working_directory: Path
    output_directory: Path
    metadata_directory: Path
    checkpoint_file: Path
    checkpoint_file_backup: Path


def setup_generation_directories(globals_config: dict, metadata: bool) -> GenerationSetup:
    """Setup directories and paths for generation.

    Args:
        globals_config: Global configuration dictionary
        metadata: Whether metadata generation is enabled

    Returns:
        GenerationSetup with configured paths
    """
    working_directory = Path(get_config_value(globals_config, "working-directory", "."))

    # Setup base directories
    checkpoint_file = working_directory / "checkpoint.json"
    checkpoint_file_backup = working_directory / "checkpoint.json.bak"

    output_directory = working_directory / get_config_value(globals_config, "output-directory", "output")
    output_directory.mkdir(exist_ok=True)

    metadata_directory = working_directory / get_config_value(globals_config, "metadata-directory", "metadata")
    if metadata:
        metadata_directory.mkdir(exist_ok=True)

    return GenerationSetup(
        working_directory=working_directory,
        output_directory=output_directory,
        metadata_directory=metadata_directory,
        checkpoint_file=checkpoint_file,
        checkpoint_file_backup=checkpoint_file_backup,
    )


def create_generator_instance(generator_name: str, generator_config: dict) -> Generator:
    """Create and configure a generator instance.

    Args:
        generator_name: Name of the generator
        generator_config: Generator configuration

    Returns:
        Configured generator instance
    """
    class_spec = generator_config["class"]
    resolved_class_path = resolve_class_path(class_spec, generator_name)
    return get_generator(generator_name, generator_config, resolved_class_path)


def setup_batch_config(
    generator_name: str,
    generator_config: dict,
    setup: GenerationSetup,
    overwrite: bool,
    metadata: bool,
) -> BatchProcessingConfig:
    """Setup batch processing configuration for a generator.

    Args:
        generator_name: Name of the generator
        generator_config: Generator configuration
        setup: Generation setup paths
        overwrite: Whether to overwrite existing files
        metadata: Whether to generate metadata

    Returns:
        BatchProcessingConfig instance
    """
    output_config = generator_config.get("output", {})
    file_name_template = output_config.get("file_name", f"{generator_name}-{{{{ start_time }}}}-{{{{ duration }}}}.gwf")
    output_arguments = output_config.get("arguments", {})

    generator_checkpoint = setup.checkpoint_file.parent / f"checkpoint_{generator_name}.json"

    return BatchProcessingConfig(
        file_name_template=file_name_template,
        output_directory=setup.output_directory,
        metadata_directory=setup.metadata_directory,
        output_arguments=output_arguments,
        overwrite=overwrite,
        metadata=metadata,
        checkpoint_file=generator_checkpoint,
        checkpoint_file_backup=generator_checkpoint.with_suffix(".json.bak"),
    )


def execute_generator(
    generator: Generator,
    generator_name: str,
    batch_config: BatchProcessingConfig,
    detectors: list[str],
) -> None:
    """Execute generator with appropriate detector handling.

    Args:
        generator: The generator instance
        generator_name: Name of the generator
        batch_config: Batch processing configuration
        detectors: List of detectors
    """
    uses_detector_placeholder = "{detector}" in batch_config.file_name_template or any(
        "{detector}" in str(v) for v in batch_config.output_arguments.values()
    )

    if uses_detector_placeholder and detectors:
        # Multi-detector generator: process each detector separately
        logger.info("Multi-detector mode: generating %s data for %d detectors", generator_name, len(detectors))
        for detector in detectors:
            logger.info("Generating %s data for detector: %s", generator_name, detector)

            # Check if generator supports multi-detector operation
            if hasattr(generator, "set_detector"):
                generator.set_detector(detector)

            # Process batches for this detector
            for batch in tqdm(generator, desc=f"Generating {generator_name} data for {detector}"):
                process_batch(generator=generator, batch=batch, config=batch_config, detector=detector)
    else:
        # Single generator, population generator, or network-wide generator
        logger.info("Single-mode: generating %s data (detector-agnostic)", generator_name)
        for batch in tqdm(generator, desc=f"Generating {generator_name} data"):
            process_batch(generator=generator, batch=batch, config=batch_config)


def process_single_generator(
    generator_name: str,
    generator_config: dict,
    globals_config: dict,
    setup: GenerationSetup,
    overwrite: bool,
    metadata: bool,
) -> None:
    """Process a single generator.

    Args:
        generator_name: Name of the generator
        generator_config: Generator configuration
        globals_config: Global configuration
        setup: Generation setup paths
        overwrite: Whether to overwrite existing files
        metadata: Whether to generate metadata
    """
    logger.info("Processing generator: %s", generator_name)

    # Create generator instance
    generator = create_generator_instance(generator_name, generator_config)

    # Setup batch processing configuration
    batch_config = setup_batch_config(generator_name, generator_config, setup, overwrite, metadata)

    # Load checkpoint if exists
    if batch_config.checkpoint_file.is_file():
        generator.load_state(file_name=batch_config.checkpoint_file)

    # Execute generator with detector handling
    detectors = globals_config.get("detectors", [])
    execute_generator(generator, generator_name, batch_config, detectors)


def generate_command(
    config_file_name: Annotated[str, typer.Argument(help="Configuration file path")],
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing files")] = False,
    metadata: Annotated[bool, typer.Option("--metadata", help="Generate metadata files")] = False,
) -> None:
    """Generate mock data based on the configuration file.

    Args:
        config_file_name (str): Name of the configuration file.
        overwrite (bool): If True, overwrite the existing file, otherwise raise an error if output
            already exists.
        metadata (bool): If True, write the metadata to file.

    Raises:
        FileExistsError: If output file exists and overwrite is False, raise an error.

    Returns:
        None
    """
    # Load and process configuration
    raw_config = load_config(file_name=Path(config_file_name))
    processed_config = process_config(raw_config)
    globals_config = processed_config["globals"]

    # Setup directories and paths
    setup = setup_generation_directories(globals_config, metadata)

    # Set up signal handlers
    setup_signal_handlers(setup.checkpoint_file, setup.checkpoint_file_backup)

    # Process each generator sequentially
    for generator_name, generator_config in processed_config["generators"].items():
        process_single_generator(
            generator_name=generator_name,
            generator_config=generator_config,
            globals_config=globals_config,
            setup=setup,
            overwrite=overwrite,
            metadata=metadata,
        )
