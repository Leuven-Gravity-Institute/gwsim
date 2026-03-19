# gwsim – This Package Has Moved

[![Status: Moved](https://img.shields.io/badge/status-moved-critical)](https://pypi.org/project/gwmock/)
[![PyPI - gwmock](https://img.shields.io/badge/pypi-gwmock-blue)](https://pypi.org/project/gwmock/)
[![PyPI Version](https://img.shields.io/pypi/v/gwsim)](https://pypi.org/project/gwsim/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwsim)](https://pypi.org/project/gwsim/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Leuven-Gravity-Institute/gwsim/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/1115995501.svg)](https://doi.org/10.5281/zenodo.17925458)

**gwsim** is now **[gwmock](https://pypi.org/project/gwmock/)**.

## Why?

The renaming is intended to avoid confusion with another Python package, [GWSim](https://git.ligo.org/benoit.revenu/gwsim), which is designed for creating mock GW samples for different astrophysical populations and cosmological models of binary black holes.

## How to Upgrade

```bash
pip uninstall gwsim
pip install gwmock
```

## Migration Note

Installing this version of `gwsim` will automatically install `gwmock` as a dependency. Your existing imports will continue to work via a wrapper, but you should update them to import `gwmock` as soon as possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues, please open an issue on [GitHub](https://github.com/Leuven-Gravity-Institute/gwmock/issues/new) or contact the maintainers.
