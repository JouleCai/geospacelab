import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'geospacelab',         # How you named your package folder (MyLib)
    version = '0.1.3',      # Start with a small number and increase it with every change you make
    license='GPL-3.0 License',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description = 'Collect, manage, and visualize geospace data.',   # Give a short description about your library
    author = 'Lei Cai',                   # Type in your name
    author_email = 'lei.cai@oulu.fi',      # Type in your E-Mail
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url = 'https://github.com/JouleCai/geospacelab',   # Provide either the link to your github or to your website
    # download_url = 'https://github.com/JouleCai/geospacelab/archive/refs/tags/v0.1.tar.gz',    # I explain this later on
    keywords = ['Geospace', 'EISCAT', 'DMSP', 'Space weather', 'Ionosphere', 'Space', 'Magnetosphere'],   # Keywords that define your package best
    install_requires=[            # I get to this in a second
              'requests>=2.26.0',
              'beautifulsoup4>=4.9.3',
              'netcdf4>=1.5.7',
              'h5py>=3.2.1',
              'numpy>=1.20',
              'scipy>=1.60',
              'cdflib>=0.3.20',
              'matplotlib>=3.3',
              'madrigalweb>=3.2',
          ],
    python_requires='>=3.8',
    # py_modules=["geospacelab"],
    # package_dir={'':'geospacelab'},
    classifiers=[
        'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',   # Again, pick a license
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        ],
)
