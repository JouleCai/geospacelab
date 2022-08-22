# import the required packages and modules
import datetime
from geospacelab.visualization.mpl.dashboards import TSDashboard

# Set the starting and stopping times
dt_fr = datetime.datetime(2015, 3, 16, 12)  # from
dt_to = datetime.datetime(2015, 3, 19, 12)  # to

# Create a DataHub instance
db = TSDashboard(dt_fr=dt_fr, dt_to=dt_to)

# Dock the sourced datasets
ds_omni = db.dock(datasource_contents=['cdaweb', 'omni'], omni_type='OMNI2', omni_res='1min', load_mode='AUTO', allow_load=True)
ds_kp = db.dock(datasource_contents=['gfz', 'kpap'])
ds_sym = db.dock(datasource_contents=['wdc', 'asysym'])

# Get the variables stored in their parent datasets.
b_x = ds_omni['B_x_GSM']
b_y = ds_omni['B_y_GSM']
b_z = ds_omni['B_z_GSM']
v_sw = ds_omni['v_sw']
n_p = ds_omni['n_p']
p_dyn = ds_omni['p_dyn']

kp = ds_kp['Kp']
sym_h = ds_sym['SYM_H']

# Assign the panel layouts
panel_layouts = [[b_x], [b_y], [b_z]]
# panel_layouts = [[b_x, b_y, b_z]]
# panel_layouts = [[b_x, b_y, b_z], [v_sw]]

# panel_layouts = [[b_x, b_y, b_z], [v_sw], [p_dyn], [kp, [sym_h]]]

# Set the layouts of the dashboard and panels
db.set_layout(panel_layouts=panel_layouts)

# Make the plots
db.draw()
# dt_1 = datetime.datetime(2015, 3, 17, 4, 45)
# db.add_vertical_line(dt_1, color='r')
# dt_2 = datetime.datetime(2015, 3, 17, 6, 40)
# db.add_vertical_line(dt_2)
# db.add_shading(dt_1, dt_2, bottom_extend=0, color='y', alpha=0.2)
# db.add_top_bar(dt_1, dt_2, bottom=0., top=0.02, color='y', label='SSC')
#
# dt_3 = datetime.datetime(2015, 3, 17, 22, 50)
# db.add_vertical_line(dt_3)
# db.add_shading(dt_2, dt_3, bottom_extend=0, color='r', alpha=0.2)
# db.add_top_bar(dt_2, dt_3, bottom=0., top=0.02, color='r', label='MP')
# dt_4 = datetime.datetime(2015, 3, 18, 18, 0)
# db.add_vertical_line(dt_4)
# db.add_shading(dt_3, dt_4, bottom_extend=0, color='g', alpha=0.2)
# db.add_top_bar(dt_3, dt_4, bottom=0., top=0.02, color='g', label='RP')
# Show the plots
# db.show()

db.save_figure(file_name='manuscript_exmaple_2_omni_1')