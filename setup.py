from setuptools import setup, find_packages
setup(
    name="ir_vo",
    version="0.1",
    packages=find_packages("src"),   # finds vo, vo.ros, etc.
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "opencv-python",
        # plus whatever else you need (torch, scipy …)
    ],
)