import h5py
import pathlib
import geospacelab.toolbox.utilities.pylogging as mylog


def list_eiscat_hdf5_variables(fh5, var_groups=None, var_names_queried=None, display=False):
    if isinstance(fh5, str) or isinstance(fh5, pathlib.PurePath):
        fh5 = h5py.File(fh5, 'r')
    elif not isinstance(fh5, h5py._hl.files.File):
        raise TypeError
    if var_groups is None:
        var_groups = ['par0d', 'par1d', 'par2d']
    var_info = {
        'name': [],
        'unit': [],
        'index': [],
        'group': [],
        'name_GUISDAP': [],
        'note': []
    }
    for var_group in var_groups:
        metadata_var = fh5['metadata'][var_group]
        for ind in range(metadata_var.shape[0]):
            var_name = metadata_var[ind, 0].decode('UTF-8').strip()

            if var_names_queried is not None:
                if var_name not in var_names_queried:
                    continue
            var_note = metadata_var[ind, 1].decode('UTF-8').strip()
            var_unit = metadata_var[ind, 2].decode('UTF-8').strip()
            var_name_GUISDAP = metadata_var[ind, 3].decode('UTF-8').strip()

            var_info['name'].append(var_name)
            var_info['index'].append(ind)
            var_info['unit'].append(var_unit)
            var_info['note'].append(var_note)
            var_info['group'].append(var_group)
            var_info['name_GUISDAP'].append(var_name_GUISDAP)

    len_vars = len(var_info['name'])
    if len_vars == 0:
        print('Cannot find the queried variable!')

    if display:
        mylog.simpleinfo.info("Listing the variables in the ESICAT hdf5 file ({})".format(fh5))
        mylog.simpleinfo.info(
            '{:<4s}{:20s}{:10s}{:10s}{:20s}{:20s}{:^60s}'.format(
                '', 'Name', 'Group', 'Index', 'Unit', 'Name (GUISDAP)', 'Note'
            )
        )
        for i in range(len_vars):
            mylog.simpleinfo.info('{:<4s}{:20s}{:10s}{:<10d}{:20s}{:20s}{:60s}'.format(
                '', var_info['name'][i], var_info['group'][i], var_info['index'][i],
                var_info['name_GUISDAP'][i], var_info['unit'][i], var_info['note'][i])
            )
        print()
    return var_info



