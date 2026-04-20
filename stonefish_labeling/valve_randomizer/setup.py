from glob import glob

from setuptools import find_packages, setup

package_name = "valve_randomizer"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vortex",
    maintainer_email="web@vortexntnu.no",
    description="Publishes random valve joint setpoints to the Stonefish valve servos.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "valve_randomizer_node = valve_randomizer.valve_randomizer_node:main",
        ],
    },
)
