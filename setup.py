from setuptools import setup


CLASSIFIERS = """\
License :: OSI Approved
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

DISTNAME = "ISCAT Post Processing"
AUTHOR = "Sandro Münch"
AUTHOR_EMAIL = "sandro.muench1195@gmail.com"
DESCRIPTION = "IRM image post processing steps to detect particles and determine their molecular weight."
LICENSE = "MIT"
README = "Longer description"

VERSION = "0.1.0"
ISRELEASED = False

PYTHON_MIN_VERSION = "3.9"
PYTHON_REQUIRES = f">={PYTHON_MIN_VERSION}"

INSTALL_REQUIRES = ["numpy>=1.19.0", "pandas", "matplotlib", "scikit-image", "nptyping", "piscat", "jupyter"]

PACKAGES = ["ISCAT_Analysis"]

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    long_description=README,
    packages=PACKAGES,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS],
    license=LICENSE,
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
