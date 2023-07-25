from setuptools import setup, find_packages

setup(
    name="measurement_codes_ut",
    packages=find_packages(),
    version="1.0.7",
    install_requires = [
        # Github Private Repository
        'plottr@git+https://github.com/qipe-nlab/plottr.git@search-datadict',
        'qcodes_drivers@git+https://github.com/qipe-nlab/qcodes_drivers.git',
        'sequence_parser@git+https://github.com/qipe-nlab/sequence_parser.git'
    ]
)