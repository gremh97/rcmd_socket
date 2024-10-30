from setuptools import setup, find_packages
from importlib.metadata import version, PackageNotFoundError

try:
    # the packages are already installed.
    version = version("rcmd")
except PackageNotFoundError:
    # the packages are not yet installed.
    from rcmd import __version__ as version

setup(
    name='rcmd',
    version=version,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "paramiko",
        "scp",
        "tqdm",
        "numpy",
        "tabulate",
        "pycocotools",
    ],
    entry_points='''
        [console_scripts]
        rcmd=rcmd.cli:cli
    ''',
    package_data={
        'rcmd': ['boards.json'],
    },
)