from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rapha-ai",
    version="0.2.0",
    author="Rapha Protocol Team",
    author_email="hello@rapha.ltd",
    description="The official SDK for Rapha Protocol: Compute-to-Data AI Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://rapha.ltd",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography"
    ],
    python_requires='>=3.10',
    install_requires=[
        "requests>=2.25.0",
        "torch>=1.9.0"
    ],
    extras_require={
        "huggingface": ["transformers>=4.30.0"],
    },
)
