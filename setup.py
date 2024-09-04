import os
import shutil
from setuptools import setup, find_packages, Command


with open("README.md", "r") as fh:
    long_description = fh.read()


class CopyData(Command):
    """A custom command to copy files from 'weights' and 'processed' to the package directory
    before installation."""

    description = "Copy data files to package directory"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Define the source and destination directories
        source_files = [
            "weights/nn_debug_medicare/config.yaml",
            "weights/nn_debug_medicare/posterior_samples.pt",
            "weights/master.yaml",
            "data/processed/bspline_basis.parquet",
            "data/processed/endogenous_states_actions.parquet",
            "data/processed/exogenous_states.parquet",
            "data/processed/heat_alerts.parquet",
        ]
        destination_dir = "weather2alert"

        # Create the destination directory if it doesn't exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Copy the files
        for source_file in source_files:
            tgt_dir = os.path.join(destination_dir, os.path.dirname(source_file))
            os.makedirs(tgt_dir, exist_ok=True)
            shutil.copy(source_file, os.path.join(destination_dir, source_file))

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
        "weather2alert": [
            "weights/nn_debug_medicare/config.yaml"
            "weights/nn_debug_medicare/posterior_samples.pt"
            "weights/master.yaml"
            "data/processed/bspline_basis.parquet"
            "data/processed/endogenous_states_actions.parquet"
            "data/processed/exogenous_states.parquet"
            "data/processed/heat_alerts.parquet"
        ]
    },
#     data_files=[
#         (
#             "weights/nn_debug_medicare",
#             ["weights/nn_debug_medicare/config.yaml", "weights/nn_debug_medicare/posterior_samples.pt"],
#         ),
#         ("weights", ["weights/master.yaml"]),
#         (
#             "data/processed",
#             [
#                 "data/processed/bspline_basis.parquet",
#                 "data/processed/endogenous_states_actions.parquet",
#                 "data/processed/exogenous_states.parquet",
#                 "data/processed/heat_alerts.parquet",
#             ],
#         )
# ,
#     ],
    cmdclass={"copy_data": CopyData},
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
