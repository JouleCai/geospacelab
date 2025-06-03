import matplotlib as mpl
import matplotlib.pyplot as plt

from geospacelab.config._preferences import pref


from cycler import cycler

try:
    mpl_style = pref.user_config['visualization']['mpl']['style']
except KeyError:
    uc = pref.user_config
    uc['visualization']['mpl']['style'] = 'light'
    pref.set_user_config(user_config=uc, set_as_default=True)


# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'book'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

# plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')
mpl_style = pref.user_config['visualization']['mpl']['style']

if mpl_style == 'light':
    plt.rcParams['axes.facecolor'] = '#FCFCFC'
    plt.rcParams['text.color'] = 'k'
    default_cycler = (cycler(color=['tab:blue', 'tab:red', 'tab:green', 'tab:purple',  'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']))
    default_cycler = (cycler(color=['#1f77b4DD', '#ff7f0eDD', '#2ca02cDD', '#d62728DD', '#9467bdDD', '#8c564bDD', '#e377c2DD', '#7f7f7fDD', '#bcbd22DD', '#17becfDD']))
    # colors = [
    #     (0.8980392156862745, 0.5254901960784314, 0.023529411764705882),
    #     (0.36470588235294116, 0.4117647058823529, 0.6941176470588235),
    #     (0.3215686274509804, 0.7372549019607844, 0.6392156862745098),
    #     (0.6, 0.788235294117647, 0.27058823529411763),
    #     (0.8, 0.3803921568627451, 0.6901960784313725),
    #     (0.1411764705882353, 0.4745098039215686, 0.4235294117647059),
    #     (0.8549019607843137, 0.6470588235294118, 0.10588235294117647),
    #     (0.1843137254901961, 0.5411764705882353, 0.7686274509803922),
    #     (0.4627450980392157, 0.3058823529411765, 0.6235294117647059),
    #     (0.9294117647058824, 0.39215686274509803, 0.35294117647058826),
    # ]
    colors = [
        (0.36470588235294116, 0.4117647058823529, 0.6941176470588235),
        (0.9294117647058824, 0.39215686274509803, 0.35294117647058826),
        (0.3215686274509804, 0.7372549019607844, 0.6392156862745098),
        (0.8980392156862745, 0.5254901960784314, 0.023529411764705882),
        (0.6, 0.788235294117647, 0.27058823529411763),
        (0.8, 0.3803921568627451, 0.6901960784313725),
        (0.1411764705882353, 0.4745098039215686, 0.4235294117647059),
        (0.8549019607843137, 0.6470588235294118, 0.10588235294117647),
        (0.1843137254901961, 0.5411764705882353, 0.7686274509803922),
        (0.4627450980392157, 0.3058823529411765, 0.6235294117647059),

    ]
    default_cycler = (cycler(color=colors))
    plt.rc('axes', prop_cycle=default_cycler)
elif mpl_style == 'dark':
    plt.rcParams['figure.facecolor'] = '#0C1C23'
    plt.rcParams['savefig.facecolor'] = '#0C1C23'

    plt.rcParams['axes.facecolor'] = '#FFFFFF20'
    plt.rcParams['axes.edgecolor'] = '#FFFFFF3D'
    plt.rcParams['axes.labelcolor'] = '#FFFFFFD9'

    plt.rcParams['xtick.color'] = '#FFFFFFD9'
    plt.rcParams['ytick.color'] = '#FFFFFFD9'
    plt.rcParams['text.color'] = 'white'

    plt.rcParams['grid.color'] = '#FFFFFF'
    plt.rcParams['legend.facecolor'] = plt.rcParams['axes.facecolor']
    plt.rcParams['legend.edgecolor'] = '#FFFFFFD9'

    # seaborn dark:['#001c7f', '#b1400d', '#12711c', '#8c0800', '#591e71', '#592f0d', '#a23582', '#3c3c3c', '#b8850a', '#006374']
    # seaborn pastel '#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff', '#debb9b', '#fab0e4', '#cfcfcf', '#fffea3', '#b9f2f0'
    default_cycler = (cycler(color=['#F5EE33', '#33FF99', 'r', '#9467bd', '#08C7FE', '#FE66BB', ]))
    colors = [
        (0.1843137254901961, 0.5411764705882353, 0.7686274509803922),
        (0.9294117647058824, 0.39215686274509803, 0.35294117647058826),
        (0.3215686274509804, 0.7372549019607844, 0.6392156862745098),
        (0.8980392156862745, 0.5254901960784314, 0.023529411764705882),
        (0.6, 0.788235294117647, 0.27058823529411763),
        (0.8, 0.3803921568627451, 0.6901960784313725),
        (0.1411764705882353, 0.4745098039215686, 0.4235294117647059),
        (0.8549019607843137, 0.6470588235294118, 0.10588235294117647),
        (0.36470588235294116, 0.4117647058823529, 0.6941176470588235),
        (0.4627450980392157, 0.3058823529411765, 0.6235294117647059),
    ]
    default_cycler = (cycler(color=colors))
    # default_cycler = (cycler(color=palettable.cartocolors.qualitative.Safe_10.mpl_colors))
    plt.rc('axes', prop_cycle=default_cycler)
else:
    plt.style.use(mpl_style)