from setuptools import setup, find_packages

setup(
    name="mugo",
    version="0.1.1",
    author="SciML Team",
    author_email="sciml.open.tools@gmail.com",
    description="Differentiable Combinatorial Optimization for Genomics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://mugo-framework.netlify.app/", # 以后填匿名的
    
    # 【核心】只打包 mugo 文件夹，忽略 src, dataset, results
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