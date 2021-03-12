from distutils.core import setup  # nopep8

setup(
    author="Franz M. Heuchel",
    name="measuretf",
    version="0.3",
    packages=["measuretf"],
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "matplotlib",
        "sounddevice",
        "response",
        "pysoundfile",
    ],
)
