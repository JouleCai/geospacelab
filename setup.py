from distutils.core import setup
setup(
  name = 'geospacelab',         # How you named your package folder (MyLib)
  packages = ['geospacelab'],   # Chose the same as "name"
  version = 'v0.1',      # Start with a small number and increase it with every change you make
  license='GPL-3.0 License',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Collect, manage, and visualize geospace data.',   # Give a short description about your library
  author = 'Lei Cai',                   # Type in your name
  author_email = 'lei.cai@oulu.fi',      # Type in your E-Mail
  url = 'https://github.com/JouleCai/geospacelab',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/JouleCai/geospacelab/archive/refs/tags/v0.1.tar.gz',    # I explain this later on
  keywords = ['Geospace', 'EISCAT', 'DMSP', 'Space weather', 'Ionosphere', 'Space', 'Magnetosphere'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'requests',
          'beautifulsoup4',
          'netcdf4',
          'h5py',
          'numpy',
          'scipy',
          'cdflib',
          'cartopy',
          'matplotlib',
          'madrigalweb',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GPL-3.0 License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
