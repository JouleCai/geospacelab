import os
import geospacelab.datahub.sources.madrigal.utilities as madrigal
import geospacelab.datahub.sources.madrigal.eiscat as eiscat

fn = "EISCAT_2021-03-10_beata_ant@uhfa.hdf5"
madrigal.show_hdf5_structure(filename=fn, filepath="")
madrigal.show_hdf5_metadata(filename=fn, filepath="")
eiscat.list_eiscat_hdf5_variables("./examples/" + fn)

fn = "MAD6300_2021-03-10_beata_ant@uhfa.hdf5"
madrigal.show_hdf5_structure(filename=fn, filepath="")
madrigal.show_hdf5_metadata(filename=fn, filepath="")

fn = "MAD6400_2021-03-10_beata_ant@uhfa.hdf5"
madrigal.show_hdf5_structure(filename=fn, filepath="")
madrigal.show_hdf5_metadata(filename=fn, filepath="")
