from setuptools import setup, find_packages

setup(
    name="statsaid",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy",
        "statsmodels",
    ],
    author="Leran Wang",
    author_email="your.email@example.com",
    description="A toolkit for data exploration, cleaning, and analysis in research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/statsaid",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.6",
)