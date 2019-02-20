from distutils.core import setup  # nopep8

setup(
    author="Franz M. Heuchel",
    name="measuretf",
    version="0.1dev",
    py_modules=["measuretf"],
    install_requires=["numpy", "scipy", "tqdm", "matplotlib", "sounddevice", "response"],
)
