import geospacelab.visualization.mpl_toolbox as mpl
import geospacelab.datahub as datahub

default_layout_config = {
    'left': 0.15,
    'right': 0.8,
    'bottom': 0.15,
    'top': 0.88,
    'hspace': 0.1,
    'wspace': 0.1
}

default_figure_config = {
    'figsize': (12, 12),    # (width, height)
    'dpi': 100,
}


class GeoViewer(datahub.DataHub, mpl.Dashboard):
    def __init__(self, **kwargs):
        new_figure = kwargs.pop('new_figure', True)
        figure_config = kwargs.pop('figure_config', default_figure_config)

        super().__init__(visual='on', figure_config=figure_config, new_figure=new_figure, **kwargs)

    def set_layout(self, num_rows=None, num_cols=None, left=None, right=None, bottom=None, top=None,
                   hspace=None, wspace=None, **kwargs):

        if left is None:
            left = default_layout_config['left']
        if right is None:
            right = default_layout_config['right']
        if bottom is None:
            bottom = default_layout_config['bottom']
        if top is None:
            top = default_layout_config['top']
        if hspace is None:
            hspace = default_layout_config['hspace']
        if hspace is None:
            wspace = default_layout_config['wspace']

        super().set_layout(num_rows=num_rows, num_cols=num_cols, left=left, right=right, bottom=bottom, top=top,
                           hspace=hspace, wspace=wspace, **kwargs)

