from setuptools import setup, find_packages

setup(
    name="mugo",
    version="0.1.1",
    author="MUGO authors",
    author_email="sciml.open.tools@gmail.com",
    description="Differentiable Combinatorial Optimization for Genomics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aicb-ZhangLabs/MUGO",
    license="MIT",

    # Package only the lightweight mugo shell; paper reproduction scripts live in src/.
    packages=find_packages(include=["mugo", "mugo.*"]),

    install_requires=[
        "torch>=2.0",
        "pandas",
        "numpy",
        "pyfaidx",
        "borzoi_pytorch"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
