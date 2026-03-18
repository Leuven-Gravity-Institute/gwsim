"""Custom build hook to include only YAML files from examples/ in gwmock/examples/."""

from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Copy only YAML files from examples/ to gwmock/examples/ in the wheel."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:  # pylint: disable=unused-argument
        """Run during wheel build to copy filtered YAML files."""
        if self.target_name != "wheel":
            return  # Only apply to wheel builds

        examples_src = Path(self.root) / "examples"
        if not examples_src.exists():
            return

        # Force-include will be handled by copying files to the build directory
        # We need to add files to the wheel's force_include dynamically
        force_include = build_data.setdefault("force_include", {})

        # Find all YAML files and add them to force_include
        for yaml_file in examples_src.rglob("*.yaml"):
            rel_path = yaml_file.relative_to(examples_src)
            # Map source file to destination in gwmock/examples/
            dest_path = f"gwmock/examples/{rel_path.as_posix()}"
            force_include[str(yaml_file)] = dest_path

        num_files = len([k for k in force_include if str(k).endswith(".yaml")])
        if num_files > 0:
            print(f"✓ Including {num_files} YAML file(s) in gwmock/examples/")
