import os
import shutil
from setuptools import setup, find_packages
from setuptools.command.install import install


with open("README.md", "r") as fh:
    long_description = fh.read()


class CustomInstall(install):
    """A custom command to copy files from 'weights' and 'processed' to the package directory
    before installation."""

    def run(self):
        # Define the source and destination directories
        source_files = [
            "weights/nn_debug_medicare/config.yaml",
            "weights/nn_debug_medicare/posterior_samples.pt",
            "weights/nn_full_medicare/config.yaml",
            "weights/nn_full_medicare/posterior_samples.pt",
            "weights/master.yaml",
            "data/processed/bspline_basis.parquet",
            "data/processed/endogenous_states_actions.parquet",
            "data/processed/exogenous_states.parquet",
            "data/processed/confounders.parquet",
        ]
        destination_dir = "weather2alert"

        # Copy the files
        for source_file in source_files:
            dirname = os.path.dirname(source_file)
            os.makedirs(os.path.join(destination_dir, dirname), exist_ok=True)
            shutil.copy(source_file, os.path.join(destination_dir, source_file))

            # for every subdir of dirname create and __init__.py file if not already there
            curr = "."
            for sub_dir in dirname.split(os.sep):
                curr = os.path.join(curr, sub_dir)
                init_file = os.path.join(destination_dir, curr, "__init__.py")
                if not os.path.exists(init_file):
                    with open(init_file, "w") as f:
                        f.write("")     

        install.run(self)

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
    cmdclass={"install": CustomInstall},
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
