import h5py
import os
import geospacelab.toolbox.utilities.pybasic as pybasic


def show_structure(filename, filepath=''):
    """
    Show madrigal hdf5 file structure in console.
    Example:
        fn = "/home/leicai/01_work/00_data/madrigal/DMSP/20151102/dms_20151102_16s1.001.hdf5"
        show_structure(fn)
    """
    with h5py.File(os.path.join(filepath, filename), 'r') as fh5:
        pybasic.dict_print_tree(fh5, value_repr=True, dict_repr=True, max_level=None)


def show_metadata(filename, filepath='', fields=None):
    """
    Show madrigal hdf5 file metadata values.
    Example:
        fn = "/home/leicai/01_work/00_data/madrigal/DMSP/20151102/dms_20151102_16s1.001.hdf5"
        show_metadata(fn)
    """
    with h5py.File(os.path.join(filepath, filename), 'r') as fh5:
        if "metadata" in fh5.keys():
            metadata_key = "metadata"
        elif "Metadata" in fih5.keys():
            metadata_key = "Metadata"
        else:
            print("Cannot find the key either metadata or Metadata!")
            return KeyError

        keys_in = fh5[metadata_key].keys()
        if fields is None:
            keys = keys_in
        else:
            keys = fields
        print("\x1b[0;31;40m" + pybasic.retrieve_name(fh5) + "\x1b[0m")
        # for key in keys:
        #    print("\x1b[1;33;40m" + "Metadata<--" + key + ": " + "\x1b[0m")
        #    print(fh5[metadata_key][key][:])
        pybasic.dict_print_tree(fh5[metadata_key], full_value=True)


def show_group(filename, filepath='', group_name=""):
    """
    Show madrigal hdf5 file group information.
    Example:
        fn = "/home/leicai/01_work/00_data/madrigal/DMSP/20151102/dms_20151102_16s1.001.hdf5"
        show_metadata(fn, group_name="Metadata")
    """
    with h5py.File(os.path.join(filepath, filename), 'r') as fh5:
        if group_name not in fh5.keys():
            print("Cannot find the key either metadata or Metadata!")
            return KeyError

        keys_in = fh5[group_name].keys()
        print("\x1b[0;31;40m" + pybasic.retrieve_name(fh5) + "\x1b[0m")
        # for key in keys:
        #    print("\x1b[1;33;40m" + "Metadata<--" + key + ": " + "\x1b[0m")
        #    print(fh5[metadata_key][key][:])
        pybasic.dict_print_tree(fh5[group_name], full_value=True)


if __name__ == "__main__":
    fn = "/Users/lcai/Downloads/EISCAT_2021-03-10_beata_ant@uhfa.hdf5"
    show_structure(fn)
    show_metadata(fn)
    show_group(fn, group_name="figures")


