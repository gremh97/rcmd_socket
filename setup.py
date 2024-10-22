from setuptools import setup, find_packages
from importlib.metadata import version, PackageNotFoundError

try:
    # 패키지가 이미 설치되어 있는 경우
    version = version("rcmd")
except PackageNotFoundError:
    # 패키지가 아직 설치되지 않은 경우
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
        "pgrep",
    ],
    entry_points='''
        [console_scripts]
        rcmd=rcmd.cli:cli
    ''',
    package_data={
        'rcmd': ['boards.json'],
    },
)