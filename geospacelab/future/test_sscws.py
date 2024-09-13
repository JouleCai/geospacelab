#  Install these prerequisites once before executing
#  the example code:
#    pip install -U sscws

from sscws.sscws import SscWs
ssc = SscWs()

#  Edit the following time variable to suit your needs.
time = ['2018-12-19T23:00:00.000Z', '2018-12-20T00:00:00.000Z']
result = ssc.get_locations(['dmspf18'], time)
data = result['Data'][0]
coords = data['Coordinates'][0]
print(coords['X'])
#  ...
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from packaging import version
    fig = plt.figure()
    if version.parse(mpl.__version__) < version.parse('3.4'):
        ax = fig.gca(projection='3d')
    else:
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    title = data['Id'] + ' Orbit (' + coords['CoordinateSystem'].value.upper() + ')'
    ax.plot(coords['X'], coords['Y'], coords['Z'], label=title)
    ax.legend()
    plt.show()
except ImportError:
    print('To see the plot, do')
    print('pip install packaging matplotlib')
except Exception as e:
    print(e)
