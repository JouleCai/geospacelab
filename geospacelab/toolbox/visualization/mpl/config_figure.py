import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def test():
    myfig = plt.figure()
    move_figure((500, 50))
    set_figure_size([15, 15])
    plt.show()


def set_figure_size(size=None, unit='centimeters', fig=None):
    if fig is None:
        fig = plt.gcf()
    if unit == 'centimeters':
        size[0] = size[0] / 2.54
        size[1] = size[1] / 2.54
    fig.set_size_inches(size[0], size[1], forward=True)
    return


def move_figure(x, y, fig=None):
    """Move figure's upper left corner to pixel (x, y)"""
    if fig is None:
        fig = plt.gcf()
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        try:
            fig.canvas.manager.window.move(x, y)
        except:
            print('Fail to set the figure position. Backend: ' + backend)



if __name__ == "__main__":
    test()