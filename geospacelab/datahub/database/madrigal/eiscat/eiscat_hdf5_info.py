import os
import geospacelab.datahub.database.madrigal.madrigal_utilities as madrigal

fn = "EISCAT_2021-03-10_beata_ant@uhfa.hdf5"
madrigal.show_hdf5_structure(filename=fn, filepath="./example")
madrigal.show_hdf5_metadata(filename=fn, filepath="./example")

fn = "MAD6300_2021-03-10_beata_ant@uhfa.hdf5"
madrigal.show_hdf5_structure(filename=fn, filepath="./example")
madrigal.show_hdf5_metadata(filename=fn, filepath="./example")

fn = "MAD6400_2021-03-10_beata_ant@uhfa.hdf5"
madrigal.show_hdf5_structure(filename=fn, filepath="./example")
madrigal.show_hdf5_metadata(filename=fn, filepath="./example")
