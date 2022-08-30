# import the required packages and modules
import datetime
from geospacelab.datahub import DataHub


# Set the starting and stopping times
dt_fr = datetime.datetime(2015, 3, 16, 12)  # from
dt_to = datetime.datetime(2015, 3, 19, 12)  # to

# Create a DataHub instance
dh = DataHub(dt_fr=dt_fr, dt_to=dt_to)

# Dock the sourced datasets
ds_omni = dh.dock(datasource_contents=['cdaweb', 'omni'], omni_type='OMNI2', omni_res='1min', load_mode='AUTO', allow_load=True)

# Extract the varaible and the data array stored in the variable.
b_x = ds_omni['B_x_GSM']  # equivalent to b_x = db.asign_variable('B_x_GSM', dataset=ds_omni)

