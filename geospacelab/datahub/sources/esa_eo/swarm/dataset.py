# Licensed under the BSD 3-Clause License
# Copyright (C) 2021 GeospaceLab (geospacelab)
# Author: Lei Cai, Space Physics and Astronomy, University of Oulu

import numpy as np
import datetime
import copy
import pathlib
import natsort
import re

import geospacelab.datahub as datahub
from geospacelab.datahub import DatabaseModel, FacilityModel, InstrumentModel, ProductModel
from geospacelab.datahub.sources.esa_eo import esaeo_database
from geospacelab.datahub.sources.esa_eo.swarm import swarm_facility
from geospacelab.config import prf
import geospacelab.toolbox.utilities.pybasic as basic
import geospacelab.toolbox.utilities.pylogging as mylog
import geospacelab.toolbox.utilities.pydatetime as dttool


FILE_RECORD_MODEL = {
    'id': np.empty((0, ), dtype=int),
    'file_path': np.empty((0, ), dtype=object),
    'file_name': np.empty((0, ), dtype=object),
    'product_version': np.empty((0, ), dtype=object),
    'product_name': np.empty((0, ), dtype=object),
    'mission': np.empty((0, ), dtype=object),
    'sat_id': np.empty((0, ), dtype=object),
    'datetime_fr': np.empty((0, ), dtype=object),
    'datetime_to': np.empty((0, ), dtype=object),
}


# default_data_search_recursive = True

default_attrs_required = []

class Dataset(datahub.DatasetSourced):
    _default_variable_names = None
    _default_dataset_attrs = None
    _default_downloader = None
    _default_loader = None
    _default_variable_config = None
    
    def __init__(self, **kwargs):
        # kwargs = basic.dict_set_default(kwargs, **Dataset._default_dataset_attrs)

        super().__init__(**kwargs)

        self.database = kwargs.pop('database', 'ESA/EarthOnline')
        self.mission = kwargs.pop('mission', 'Swarm')
        self.instrument = kwargs.pop('instrument', '')
        self.data_file_ext = kwargs.pop('data_file_ext', '.cdf')
        self.product = kwargs.pop('product', '')
        self.product_version = kwargs.pop('product_version', '')
        self.data_file_versions = None
        self.allow_download = kwargs.pop('allow_download', False)
        self.force_download = kwargs.pop('force_download', False)
        self.dry_run = kwargs.pop('dry_run', False)
        self.quality_control = kwargs.pop('quality_control', False)
        self.calib_control = kwargs.pop('calib_control', False)
        self.add_AACGM = kwargs.pop('add_AACGM', False) 
        self.add_APEX = kwargs.pop('add_APEX', False)
        self.add_GEO_LST = kwargs.pop('add_GEO_LST', True)
        self._data_root_dir_init = copy.deepcopy(self.data_root_dir)    # Record the initial root dir

        self.sat_id = kwargs.pop('sat_id', 'A')

        self.metadata = None

        allow_load = kwargs.pop('allow_load', False)

        # self.config(**kwargs)

        if self.loader is None:
            self.loader = self._default_loader

        if self.downloader is None:
            self.downloader = self._default_downloader

        self._validate_attrs()

        if allow_load:
            self.load_data()

    def _validate_attrs(self):
        for attr_name in default_attrs_required:
            attr = getattr(self, attr_name)
            if not list(attr):
                mylog.StreamLogger.warning("The parameter {} is required before loading data!".format(attr_name))
        if not self.allow_download:
            if self.product_version in ['latest', '', None]:
                raise ValueError("The product version must be specified when downloading is not allowed.")

    def label(self, **kwargs):
        label = super().label()
        return label

    def load_data(self, **kwargs):
        self.check_data_files(**kwargs)

        default_variable_names = kwargs.pop('default_variable_names', self._default_variable_names)
        omit_join_variables = kwargs.pop('omit_join_variables', [])

        self._set_default_variables(
            default_variable_names,
            configured_variables=self._default_variable_config.configured_variables
        )
        for i, (file_path, product_version) in enumerate(zip(self.data_file_paths, self.data_file_versions)):
            load_obj = self.loader(file_path, file_type='cdf', product_version=product_version)

            for var_name in self._variables.keys():
                if i > 0 and var_name in omit_join_variables:
                    continue
                value = load_obj.variables[var_name]
                self._variables[var_name].join(value)

            # self.select_beams(field_aligned=True)
        if self.time_clip:
            self.time_filter_by_range(var_datetime_name='SC_DATETIME')
        if self.quality_control:
            self.time_filter_by_quality()
        if self.calib_control:
            self.time_filter_by_calib()
            
        if self.add_AACGM:
            self.convert_to_AACGM()

        if self.add_APEX:
            self.convert_to_APEX()

        if self.add_GEO_LST:
            self.calc_GEO_LST()
    
    def time_filter_by_range(self, **kwargs):
        kwargs.setdefault('var_datetime_name', 'SC_DATETIME')
        super().time_filter_by_range(**kwargs)

    def calc_GEO_LST(self, var_name_datetime='SC_DATETIME', var_name_glon='SC_GEO_LON'):
        import geospacelab.observatory.earth.sun_position as sun_position
        lons = self[var_name_glon].flatten()
        uts = self[var_name_datetime].flatten()

        lsts = sun_position.convert_datetime_longitude_to_local_solar_time(
            dts=uts, lons=lons
        )
        
        var = self[var_name_glon].clone()
        var.name = var_name_glon.replace('GEO_LON', 'GEO_LST')
        var.value = np.array(lsts)[:, np.newaxis]
        var.label = 'LST'
        var.unit = 'h'
        var.unit_label = 'h'
        var.group = 'GEO_LST'
        var.visual.axis[1].lim = [0, 24]
        var.visual.axis[1].ticks = np.arange(0, 24, 6)
        self[var.name] = var
        return var
            
    
    def convert_to_APEX(self, var_name_glat='SC_GEO_LAT', var_name_glon='SC_GEO_LON', var_name_gr='SC_GEO_r', var_name_datetime='SC_DATETIME'):
        import geospacelab.cs as gsl_cs
        
        glats = self[var_name_glat].flatten()
        glons = self[var_name_glon].flatten()
        grs = self[var_name_gr].flatten()

        coords_in = {
            'lat': glats,
            'lon': glons,
            'r': grs
        }
        
        dts = self[var_name_datetime].value.flatten()
        cs_sph = gsl_cs.GEOCSpherical(coords=coords_in, ut=dts)
        cs_apex = cs_sph.to_APEX(append_mlt=True)
        
        var = self[var_name_glat].clone()
        var.name = var_name_glat.replace('GEO_LAT', 'APEX_LAT')
        var.value = cs_apex['lat'].reshape(self[var_name_datetime].value.shape)
        var.label = 'APEX MLAT'
        self[var.name] = var
        
        var = self[var_name_glon].clone()
        var.name = var_name_glon.replace('GEO_LON', 'APEX_LON')
        var.value = cs_apex['lon'].reshape(self[var_name_datetime].value.shape)
        var.label = 'APEX MLON'
        self[var.name] = var
        
        var = self[var_name_glon].clone()
        var.name = var_name_glon.replace('GEO_LON', 'APEX_MLT')
        var.value = cs_apex['mlt'].reshape(self[var_name_datetime].value.shape)
        var.label = 'APEX MLT'
        var.unit = 'h'
        var.unit_label = 'h'
        var.group = 'APEX_MLT'
        var.visual.axis[1].lim = [0, 24]
        var.visual.axis[1].ticks = np.arange(0, 24, 6)
        self[var.name] = var
        

    def convert_to_AACGM(self, var_name_glat='SC_GEO_LAT', var_name_glon='SC_GEO_LON', var_name_gr='SC_GEO_r', var_name_datetime='SC_DATETIME'):
        import geospacelab.cs as gsl_cs
        
        glats = self[var_name_glat].flatten()
        glons = self[var_name_glon].flatten()
        grs = self[var_name_gr].flatten()

        coords_in = {
            'lat': glats,
            'lon': glons,
            'r': grs
        }

        dts = self[var_name_datetime].value.flatten()
        cs_sph = gsl_cs.GEOCSpherical(coords=coords_in, ut=dts)
        cs_aacgm = cs_sph.to_AACGM(append_mlt=True)
        
        var = self[var_name_glat].clone()
        var.name = var_name_glat.replace('GEO_LAT', 'AACGM_LAT')
        var.value = cs_aacgm['lat'].reshape(self[var_name_datetime].value.shape)
        var.label = 'AACGM MLAT'
        self[var.name] = var
        
        var = self[var_name_glon].clone()
        var.name = var_name_glon.replace('GEO_LON', 'AACGM_LON')
        var.value = cs_aacgm['lon'].reshape(self[var_name_datetime].value.shape)
        var.label = 'AACGM MLON'
        self[var.name] = var    
        
        var = self[var_name_glon].clone()
        var.name = var_name_glon.replace('GEO_LON', 'AACGM_MLT')
        var.value = cs_aacgm['mlt'].reshape(self[var_name_datetime].value.shape)
        var.label = 'AACGM MLT'
        var.unit = 'h'
        var.unit_label = 'h'
        var.group = 'AACGM_MLT'
        var.visual.axis[1].lim = [0, 24]
        var.visual.axis[1].ticks = np.arange(0, 24, 6)
        self[var.name] = var
        
    def time_filter_by_flag(self, flag_name, condition=None):
        
        flag_values = self[flag_name].flatten()
        inds = condition(flag_values) if callable(condition) else None
        self.time_filter_by_inds(inds)

    def search_data_files(
        self, 
        dt_fr=None, dt_to=None, 
        file_patterns=None,
        archive_yearly=True,
        file_name_by_day=True,
        file_name_by_month=False, 
        recursive=True,
        **kwargs):

        dt_fr = self.dt_fr if dt_fr is None else dt_fr
        dt_to = self.dt_to if dt_to is None else dt_to
        
        file_patterns = file_patterns if file_patterns is not None else []
        
        initial_file_dir = kwargs.pop(
            'initial_file_dir', self.data_root_dir
        )
        
        from_download = False
        if self.product_version in ['latest', '', None] or self.force_download:
            from_download = True
        else:
            if archive_yearly:
                diff_years = dt_to.year - dt_fr.year
                yys = [dt_fr.year + i for i in range(diff_years + 1)]
                file_paths = []
            else:
                yys = [None]
            
            if file_name_by_day:
                diff_days = dttool.get_diff_days(dt_fr, dt_to)
                times = [
                    dttool.get_start_of_the_day(dt_fr) + datetime.timedelta(days=i) 
                    for i in range(diff_days + 1)
                    ]
            elif file_name_by_month:
                diff_months = dttool.get_diff_months(dt_fr, dt_to)
                times = [
                    dttool.get_start_of_the_month(dt_fr) + dttool.relativedelta(months=i) 
                    for i in range(diff_months + 1)
                    ]
            else:
                times = [None]
            
            file_paths = []
            for yy in yys:
                initial_file_dir = initial_file_dir / self.product_version / 'Sat_{}'.format(self.sat_id)
                if yy is not None:
                    initial_file_dir = initial_file_dir / '{:04d}'.format(yy)
                for t in times:
                    file_patterns_t = copy.deepcopy(file_patterns)
                    if t is not None:
                        if file_name_by_day:
                            file_patterns_t.append(t.strftime('%Y%m%d') + 'T')
                        elif file_name_by_month:
                            file_patterns_t.append(t.strftime('%Y%m%d') + 'T')
                    # remove empty str
                    file_patterns_t = [pattern for pattern in file_patterns_t if str(pattern)]
                    search_pattern = '*' + '*'.join(file_patterns_t) + '*'
                    if self.data_file_ext:
                        search_pattern += self.data_file_ext
                    if recursive:
                        search_pattern = '**/' + search_pattern
                    file_paths_seg = list(pathlib.Path(initial_file_dir).glob(search_pattern))
                    file_paths_seg = natsort.natsorted(file_paths_seg, reverse=False)
                    if not file_paths_seg:
                        from_download = True
                        mylog.StreamLogger.warning(
                            "No file found for the patterns ({}) in the directory {}.".format(
                                ', '.join(file_patterns_t), initial_file_dir))
                    else:
                        file_paths.extend(file_paths_seg)

            if file_paths:
                files_record = self._parse_searched_files(file_paths=file_paths,)
                files_record = self._filtering_files_by_same_start_time(files_record)
                files_record = self._filtering_files_by_time(files_record, dt_fr=dt_fr, dt_to=dt_to)
                self.data_file_paths = files_record['file_path']
                self.data_file_versions = files_record['product_version']
        
        if from_download and self.allow_download:
            mylog.simpleinfo.info("Searching the data product \"{}\" with the version \"{}\" on the server...".format(self.product, self.product_version))
            download_obj = self.download_data(dt_fr=dt_fr, dt_to=dt_to)
            file_paths = []
            for fp in download_obj.file_paths_local:
                dt_fr, dt_to, version = self._parse_file_name(fp.name)
                search_pattern = f"*{dt_fr.strftime('%Y%m%dT%H%M%S')}*{dt_to.strftime('%Y%m%dT%H%M%S')}*{version}*{self.data_file_ext}"
                file_path = list(pathlib.Path(fp.parent).glob(search_pattern))[0]
                file_paths.append(file_path)
            files_record = download_obj._files_record_remote
            self.data_file_versions = files_record['product_version']
            self.data_file_paths = file_paths
        return 
    
    def _filtering_files_by_same_start_time(self, files_record):
        dts_fr = files_record['datetime_fr']
        versions = files_record['product_version']
        dts_to = files_record['datetime_to']
        
        dt_fr_unique, inds_dt_fr_unique, inds_dt_fr_inverse = np.unique(dts_fr, return_index=True, return_inverse=True)
        records = []
        for ind_dt_fr_u, dt_fr_u in enumerate(dt_fr_unique):
            ii = np.where(inds_dt_fr_inverse == ind_dt_fr_u)[0]
            versions_u = versions[ii]

            ver_unique, inds_v_unique, inds_v_inverse = np.unique(versions_u, return_index=True, return_inverse=True)
            for ind_v_u, ver_u in enumerate(ver_unique):
                ii_v = np.where(inds_v_inverse == ind_v_u)[0]
                records.append(
                    {
                        'datetime_fr': dts_fr[ii][ii_v],
                        'version': versions_u[ii_v],
                        'datetime_to': dts_to[ii][ii_v],
                        'index': ii[ii_v],
                    }
                )
        inds = []
        for record in records:
            ii = np.argmax(record['datetime_to'])
            ind_to_keep = record['index'][ii]
            inds.append(ind_to_keep)
        files_record = {key: files_record[key][inds] for key in files_record.keys()}
        return files_record
    
    def _filtering_files_by_time(self, files_record, dt_fr=None, dt_to=None):
        dts_fr = files_record['datetime_fr']
        dts_to = files_record['datetime_to']
        inds_t_invalid = np.where((dt_to < dts_fr) | (dt_fr > dts_to))[0]
        inds_t = np.array([i for i in range(len(dts_fr)) if i not in inds_t_invalid])
        if not list(inds_t):
            mylog.StreamLogger.warning("No matching files found on the ftp after filtering by time.")
            return copy.deepcopy(FILE_RECORD_MODEL)
        files_record_filtered = {key: files_record[key][inds_t] for key in files_record.keys()}
        return files_record_filtered
    
    def _filtering_files_by_version(self, files_record, version=None):
        if version is None:
            version = self.product_version
        files_with_versions = self._check_file_versions(files_record)

        inds_f = []
        for file_with_version in files_with_versions:
            versions = file_with_version['version']
            if version == 'latest':
                version_latest = max(versions)
                version = version_latest
            ii_v = np.where(versions == version)[0]
            if not list(ii_v):
                mylog.StreamLogger.warning(f"No file with the version {version} found for the file name pattern {file_with_version['file_names'][0]}.")
                continue
            inds_f.append(file_with_version['index'][ii_v][0])
        files_record_filtered = {key: files_record[key][inds_f] for key in files_record.keys()}
        return files_record_filtered

    def _check_file_versions(self, files_record, ):
        versions = files_record['product_version']
        dts_fr = files_record['datetime_fr']
        dts_fr_unique, inds_dt_fr_unique, inds_dt_fr_inverse = np.unique(dts_fr, return_index=True, return_inverse=True)
        records = []
        for ind_dt_fr_u, dt_fr_u in enumerate(dts_fr_unique):
            ii = np.where(inds_dt_fr_inverse == ind_dt_fr_u)[0]
            versions_u = versions[ii]

            records.append(
                {
                    'datetime_fr': dts_fr[ii],
                    'file_names': files_record['file_name'][ii],
                    'file_path': files_record['file_path'][ii],
                    'version': versions_u,
                    'index': ii,
                }
            )
        # file_names_ = [fn.replace(v, '') for fn, v in zip(file_names, versions)]
        # fn_unique, inds_fn_unique, inds_fn_inverse = np.unique(file_names_, return_index=True, return_inverse=True) 
        # file_paths = files_record['file_path']
        
        # files_with_versions = []
        # for ifn, fn in enumerate(fn_unique):
        #     ii = np.where(inds_fn_inverse == ifn)[0]
        #     versions_c = versions[ii]
            
        #     files_with_versions.append(
        #         {
        #             'file_names': file_names[ii],
        #             'file_paths': file_paths[ii],
        #             'versions': versions_c,
        #             'indices': ii,
        #         }
        #     )
        return records
    
    def _parse_file_name(self, file_name, rc_pattern=None):
        if rc_pattern is None:
            rc_pattern = r'(\d{8}T\d{6})_(\d{8}T\d{6})_(\d+)[\._]+'
        rc = re.compile(rc_pattern)
        rm = rc.findall(file_name)
        if not list(rm):
            raise ValueError(f"Cannot parse the file name {file_name} with the regex pattern {rc_pattern}.")
        dt_fr = datetime.datetime.strptime(rm[0][0], '%Y%m%dT%H%M%S')
        dt_to = datetime.datetime.strptime(rm[0][1], '%Y%m%dT%H%M%S')
        version = rm[0][2]
        return dt_fr, dt_to, version

    def _parse_searched_files(self, file_paths, rc_pattern=None, ):
        
        records = copy.deepcopy(FILE_RECORD_MODEL)
        for i, file_path in enumerate(file_paths):
            file_name = pathlib.Path(file_path).name
            dt_fr, dt_to, version = self._parse_file_name(file_name, rc_pattern)
            
            records['id'] = np.concatenate((records['id'], [i]))
            records['file_path'] = np.concatenate((records['file_path'], [file_path]))
            records['file_name'] = np.concatenate((records['file_name'], [file_name]))
            records['product_version'] = np.concatenate((records['product_version'], [version]))
            records['product_name'] = np.concatenate((records['product_name'], [self.product]))
            records['mission'] = np.concatenate((records['mission'], [self.mission if self.mission is not None else '']))
            records['sat_id'] = np.concatenate((records['sat_id'], [self.sat_id if self.sat_id is not None else '']))
            records['datetime_fr'] = np.concatenate((records['datetime_fr'], [dt_fr]))
            records['datetime_to'] = np.concatenate((records['datetime_to'], [dt_to]))
        
        return records
        

    def download_data(self, dt_fr=None, dt_to=None):
        if dt_fr is None:
            dt_fr = self.dt_fr
        if dt_to is None:
            dt_to = self.dt_to
        download_obj = self.downloader(
            dt_fr, dt_to,
            sat_id=self.sat_id,
            product=self.product,
            product_version=self.product_version,
            force_download=self.force_download,
            dry_run=self.dry_run
        )
        
        return download_obj

    @property
    def database(self):
        return self._database

    @database.setter
    def database(self, value):
        if isinstance(value, str):
            self._database = DatabaseModel(value)
        elif issubclass(value.__class__, DatabaseModel):
            self._database = value
        else:
            raise TypeError

    @property
    def product(self):
        return self._product

    @product.setter
    def product(self, value):
        if isinstance(value, str):
            self._product = ProductModel(value)
        elif issubclass(value.__class__, ProductModel):
            self._product = value
        else:
            raise TypeError

    @property
    def mission(self):
        return self._mission

    @mission.setter
    def mission(self, value):
        if isinstance(value, str):
            self._mission = FacilityModel(value)
        elif issubclass(value.__class__, FacilityModel):
            self._mission = value
        else:
            raise TypeError

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, value):
        if isinstance(value, str):
            self._instrument = InstrumentModel(value)
        elif issubclass(value.__class__, InstrumentModel):
            self._instrument = value
        else:
            raise TypeError
