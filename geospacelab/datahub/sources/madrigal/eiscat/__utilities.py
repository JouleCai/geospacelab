import h5py
import pathlib


def list_eiscat_hdf5_variables(fh5, var_groups=None, var_names_queried=None, display=True):
    if isinstance(fh5, str) or isinstance(fh5, pathlib.PurePath):
        fh5 = h5py.File(fh5, 'r')
    elif not isinstance(fh5, h5py._hl.files.File):
        raise TypeError
    if var_groups is None:
        var_groups = ['par0d', 'par1d', 'par2d']
    var_info = {
        'var_name': [],
        'var_unit': [],
        'var_ind': [],
        'var_group': [],
        'var_name_GUISDAP': [],
        'var_note': []
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

            var_info['var_name'].append(var_name)
            var_info['var_ind'].append(ind)
            var_info['var_unit'].append(var_unit)
            var_info['var_note'].append(var_note)
            var_info['var_group'].append(var_group)
            var_info['var_name_GUISDAP'].append(var_name_GUISDAP)

    len_vars = len(var_info['var_name'])
    if len_vars == 0:
        print('Cannot find the queried variable!')

    if display:
        print(
            '{:20s}{:10s}{:10s}{:20s}{:20s}{:^60s}'.format(
                'Name', 'Group', 'Index', 'Unit', 'Name (GUISDAP)', 'Note'
            )
        )
        for i in range(len_vars):
            print('{:20s}{:10s}{:<10d}{:20s}{:20s}{:60s}'.format(
                var_info['var_name'][i], var_info['var_group'][i], var_info['var_ind'][i],
                var_info['var_name_GUISDAP'][i], var_info['var_unit'][i], var_info['var_note'][i])
            )
    return var_info



