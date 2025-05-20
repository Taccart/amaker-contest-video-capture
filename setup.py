from setuptools import setup, find_packages

setup(
    name="amaker-unleash-the-bricks",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "pillow",
        "pyserial",
        "pyapriltags",
        "screeninfo",
        "pyqt6-sip",
        "pyqt6",
        "pyyaml"
    ],
    entry_points={
        'console_scripts': [
            'unleash-the-bricks=amaker.unleash_the_bricks.gui_app:main',
        ],
    },
    author="Th.Accart, aMaker club.",
    description="aMaker microbot tournament controller.",
    python_requires=">=3.6",
    include_package_data=True,
    package_data={
        "amaker.unleash_the_bricks": ["resources/fonts/**/*.ttf", "resources/fonts/**/*.otf"]
    },
)