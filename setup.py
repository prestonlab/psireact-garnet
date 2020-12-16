from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="psireact-garnet",
    version="1.0.0",
    author="Neal Morton",
    author_email="mortonne@gmail.com",
    description="PsiReact-Garnet: Hierarchical Bayesian modeling of response time data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prestonlab/psireact-garnet",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'seaborn',
        'theano',
        'pymc3',
        'psireact',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ]
)
