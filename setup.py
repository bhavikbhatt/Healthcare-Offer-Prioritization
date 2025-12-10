"""
Healthcare Offer Prioritization - Databricks Demo
Setup script for package installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="healthcare-offer-prioritization",
    version="1.0.0",
    author="Healthcare Analytics Team",
    author_email="analytics@example.com",
    description="ML-powered healthcare offer prioritization for insurance members",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/healthcare-offer-prioritization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "databricks": [
            "databricks-sdk>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "generate-healthcare-data=data.generate_synthetic_data:generate_all_data",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json"],
    },
)

