from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "matplotlib==3.10.1",
    "numpy==2.2.4",
    "pandas==2.2.3",
    "scipy==1.15.2",
    "tabulate==0.9.0",
]

setup(
    name="iaa-eval",
    version="0.1.0",
    author="Wameuh",
    author_email="wameuh@gmail.com",
    description="A Command-Line Tool for Inter-Annotator Agreement Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Wameuh/AnnotationQuality",
    packages=find_packages(include=["src", "src.*", "Utils", "Utils.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "iaa-eval=src.iaa_eval:main",
        ],
    },
)
