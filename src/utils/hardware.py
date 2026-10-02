"""
src/utils/hardware.py
---------------------
Inspects and logs system hardware, OS, and package environment details.
"""

import sys
import os
import platform
import json

def get_environment_info() -> dict:
    """Collects runtime hardware and library specifications."""
    info = {
        'os': platform.system(),
        'os_release': platform.release(),
        'os_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'python_version': platform.python_version(),
        'packages': {}
    }

    # Inspect key packages
    packages = [
        'numpy', 'scipy', 'pandas', 'scikit-learn',
        'torch', 'transformers', 'matplotlib', 'seaborn',
        'statsmodels', 'joblib'
    ]
    for pkg in packages:
        mod_name = pkg.replace('-', '_')
        try:
            mod = __import__(mod_name)
            info['packages'][pkg] = getattr(mod, '__version__', 'installed')
        except ImportError:
            info['packages'][pkg] = 'not_installed'

    # Check CUDA / GPU
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_device_name'] = torch.cuda.get_device_name(0)
        else:
            info['cuda_version'] = None
            info['gpu_device_name'] = 'CPU Only'
    except ImportError:
        info['cuda_available'] = False
        info['cuda_version'] = None
        info['gpu_device_name'] = 'PyTorch not installed'

    return info

if __name__ == '__main__':
    env = get_environment_info()
    print(json.dumps(env, indent=2))
