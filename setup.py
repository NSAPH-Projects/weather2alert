from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="weather2alert",
    version="0.1.0",
    license="MIT",
    description="A gym environment for optimizing heat alert issuance during heatwaves",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anonymous",
    author_email="actual.marmots.0d@icloud.com",  # temporarily anonymous
    # url="https://github.com/your-username/your-repo",  # temporarily anonymous
    packages=find_packages(),
    package_data={
        "weather2alert": ["weights"]
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
)
