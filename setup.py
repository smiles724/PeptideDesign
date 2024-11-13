import setuptools as tools

tools.setup(
    name="dflow",
    packages=[
        'openfold',
        'dflow',
        'ProteinMPNN'
    ],
    package_dir={
        'openfold': './openfold',
        'dflow': './dflow',
        'ProteinMPNN': './ProteinMPNN',
    },
)
