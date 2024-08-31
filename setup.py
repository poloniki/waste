from setuptools import find_packages, setup

with open("requirements.txt", "r") as file:
    lines = file.readlines()

requirements = [each.strip() for each in lines if "#" not in each]

setup(
    name="waste",
    version="0.0.1",
    install_requires=requirements,
    packages=find_packages(),
)
