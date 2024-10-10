from setuptools import setup, find_packages

setup(
    name="weather2alert",
    version="0.1.0",
    description="A gym environment for optimizing heat alert issuance during heatwaves",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Anonymous",
    author_email="actual.marmots.0d@icloud.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "scipy",
        "tqdm",
        "pyarrow",
        "pandas",
        "torch",
        "gymnasium",
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,  # Include data files from MANIFEST.in
    package_data={
        "weather2alert": [
            "weights/**/*",
            "data/**/*",
        ],
    },
)
