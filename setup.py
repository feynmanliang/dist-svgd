import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="distsvgd",
    version="0.0.1",
    author="Feynman Liang",
    author_email="feynman@berkeley.edu",
    description="Distributed implementation of Stein's variational gradient descent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/feynmanliang/dist-svgd",
    packages=setuptools.find_packages(),
)
