from setuptools import find_namespace_packages, setup


setup(
    name="dflow",
    packages=find_namespace_packages(
        include=("dflow", "dflow.*", "openfold", "openfold.*", "ProteinMPNN", "ProteinMPNN.*"),
        exclude=("*.configs",),
    ),
    package_data={
        "dflow": ["configs/*.yaml"],
        "openfold.resources": ["stereo_chemical_props.txt"],
        "ProteinMPNN": ["LICENSE"],
        "ProteinMPNN.training": ["LICENSE"],
    },
)
