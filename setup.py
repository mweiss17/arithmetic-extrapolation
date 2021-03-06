import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arithmetic_extrapolation",
    version="0.0.1",
    author="Martin Weiss",
    author_email="martin.clyde.weiss@gmail.com",
    description="Investigating the extrapolation properties of neural networks applied to mathematical reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        'torch==1.10',
        'rich', 
        'numpy',
        'google-cloud',
        'wandb',
        'submitit',
        'matplotlib',
        'speedrun @ git+https://git@github.com/inferno-pytorch/speedrun@dev#egg=speedrun',
        ],
)

