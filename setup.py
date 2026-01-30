import setuptools
import pathlib

with open("README.md", "r") as fh:
    long_description = fh.read()

package_name = "geospacelab"


def get_version():
    '''
    read the version string from __init__

    '''
    # get the init file path
    setup_path = pathlib.Path(__file__).parent.resolve()
    package_init_path = setup_path / package_name / '__init__.py'

    # read the file in
    f = open(package_init_path, 'r')
    lines = f.readlines()
    f.close()

    # search for the version
    version = 'unknown'
    for l in lines:
        if '__version__' in l:
            s = l.split('=')
            version = s[-1].strip().strip('"').strip("'")
            break
    return version


package_version = get_version()


setuptools.setup(
    name=package_name,         # How you named your package folder (MyLib)
    version=package_version,      # Start with a small number and increase it with every change you make
    license='BSD 3-Clause License',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='Collect, manage, and visualize geospace data.',   # Give a short description about your library
    author='Lei Cai',                   # Type in your name
    author_email='lei.cai@oulu.fi',      # Type in your E-Mail
    long_description=long_description,      # Long description read from the readme file
    long_description_content_type="text/markdown",
    # packages=setuptools.find_packages(exclude=['*.new', '*.new.*', 'new.*', 'new', 'local']),
    url='https://github.com/JouleCai/geospacelab',   # Provide either the link to your github or to your website
    # download_url = 'https://github.com/JouleCai/geospacelab/archive/refs/tags/v0.1.tar.gz',    # I explain this later on
    keywords=['Geospace', 'EISCAT', 'DMSP', 'Space weather', 'Ionosphere', 'Space', 'Magnetosphere'],   # Keywords that define your package best
    install_requires=[            # I get to this in a second
              'cython',
              'requests>=2.26.0',
              'beautifulsoup4>=4.9.3',
              'natsort>=7.1.1',
              'numpy<2.4.0',
              'scipy>=1.6.0',
              'h5py>=3.2.1',
              'netcdf4>=1.5.7',
              'matplotlib>=3.5',
              'madrigalweb>=3.3',
              'aacgmv2>=2.6.2',
              'cdflib>=1.2.3',
              'geopack>=1.0.10',
              'palettable',
              'tqdm',
              'toml',
              'sscws',
              'pandas>=1.5.3',
              'keyring',
          ],
    python_requires='>=3.9,<3.13',
    # py_modules=["geospacelab"],
    # package_dir={'':'geospacelab'},
    classifiers=[
        'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: BSD License',  # Again, pick a license
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        ],
)
