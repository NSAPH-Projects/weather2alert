from setuptools import setup, find_packages

setup(
    name="broach",
    version="0.1.0",
    description="Bayesian Rewards Over Actual Climate History",
    author="Anonymous",
    author_email="actual.marmots.0d@icloud.com",  # temporarily anonymous
    # url="https://github.com/your-username/your-repo",  # temporarily anonymous
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=open("requirements.txt").read().splitlines(),
)
