function [data, metadata] = load_EISCAT_hdf5(varargin)
%% load EISCAT-type hdf5 file.
% Author: L. Cai.
% Create date: 2021-07-20
% the hdf5 file can be downloaded from eiscat.se
% Input:
%       file_path: the file full path include the directory and file name

    file_path = './example/EISCAT_2021-03-10_beata_ant@uhfa.hdf5';
    
    if length(varargin) == 1
       file_path = varargin{1}; 
    end
    
    % Set the variable list, which will be stored in "data"
    var_name_dict = struct( ...
        'r_XMITloc1', 'XMITloc1', ...
        'r_XMITloc2', 'XMITloc2', ...
        'r_XMITloc3', 'XMITloc3', ...
        'r_RECloc1', 'RECloc1', ...
        'r_RECloc2', 'RECloc2', ...
        'r_RECloc3', 'RECloc3', ...
        'magic_constant', 'Magic_const', ...
        'r_SCangle', 'SCangle', ...
        'r_m0_2', 'm02', ...
        'r_m0_1', 'm01', ...
        'az', 'az', ...
        'el', 'el', ...
        'P_Tx', 'Pt', ...
        'height', 'h', ...
        'range', 'range', ...
        'n_e', 'Ne', ...
        'T_i', 'Ti', ...
        'T_r', 'Tr', ...
        'nu_i', 'Collf', ...
        'v_i_los', 'Vi', ...
        'comp_mix', 'pm', ...
        'comp_O_p', 'po+', ...
        'n_e_var', 'var_Ne', ...
        'T_i_var', 'var_Ti', ...
        'T_r_var', 'var_Tr', ...
        'nu_i_var', 'var_Collf', ...
        'v_i_los_var', 'var_Vi', ...
        'comp_mix_var', 'var_pm', ...
        'comp_O_p_var', 'var_po+', ...
        'status', 'status', ...
        'residual', 'res1' ...
    );
    
    % Set the site information for convert the sigle letter to 3-letter
    % site code
    site_info = struct( ...
        'T', 'UHF', ...
        'V', 'VHF', ...
        'K', 'KIR', ...
        'S', 'SOD', ...
        'L', 'ESR' ...
    );
    
    % load h5 data
    h5data.par0d = h5read(file_path, '/data/par0d');
    h5data.par1d = h5read(file_path, '/data/par1d');
    h5data.par2d = h5read(file_path, '/data/par2d');
    
    % get variable information 
    var_info_list = list_EISCAT_hdf5_variables(file_path);
    
    % get the gate numbers (nrec) for each beam. Note: nrec can be a single
    % integar or a 1-D array
    ind_nrec = strcmp(var_info_list.name, 'nrec');
    nrec_group = var_info_list.group{ind_nrec};
    nrec_index = var_info_list.index(ind_nrec);
    nrec = h5data.(nrec_group)(:, nrec_index);
    
    % load each variables
    field_names = fieldnames(var_name_dict);
    [ncol, ans] = size(h5data.par1d);
    for i=1 : length(field_names)
        var_name = field_names{i};
        var_name_hdf5 = var_name_dict.(var_name);
        % get variable information
        ix = strcmp(var_info_list.name, var_name_hdf5);
        var_group = var_info_list.group{ix};
        var_index = var_info_list.index(ix);
        var_value = h5data.(var_group)(:, var_index);
        if strcmp(var_group, 'par0d') % for a single number
            data.(var_name) = var_value;
        elseif strcmp(var_group, 'par1d') % for a 1-D array
            data.(var_name) = reshape(var_value, 1, ncol);
        elseif strcmp(var_group, 'par2d') % for a 2-D array
            if strcmp(nrec_group, 'par0d') % for nrec is a single number
                nrow = length(var_value) / ncol;
                if nrow ~= nrec
                    disp("Note: the number of range gates doesn't match nrec!")
                end
                data.(var_name) = reshape(var_value, nrow, ncol);
            elseif strcmp(nrec_group, 'par1d') % for nrec is a 1-D array
                num_gates = max(nrec);
                var_array = nan(num_gates, ncol);
                rec_ind_1 = 1;
                for j = 1 : ncol % reshape the 1-D array to a 2-D array
                   rec_ind_2 = rec_ind_1 + nrec(j) - 1;
                   var_array(1:nrec(j), j) = var_value(rec_ind_1:rec_ind_2);
                   rec_ind_1 = rec_ind_2 + 1;
                end
                data.(var_name) = var_array;
            end
       end  
    end
    
    field_names = fieldnames(data);
    for i = 1: length(field_names)
       var_name = field_names{i};
       if  contains(var_name, '_var')
           var_name_new = replace(var_name, '_var', '_err');
           data.(var_name_new) = sqrt(data.(var_name)); 
       end
    end
    
    % convert the unix times to datetime
    utimes = h5read(file_path, '/data/utime');
    utime_1 = utimes(:, 1);
    datetime_1 = datetime(utime_1,'ConvertFrom','epochtime','TicksPerSecond',1,'Format','dd-MMM-yyyy HH:mm:ss');
    data.datevec_1 = datevec(datetime_1)';
    data.datenum_1 = datenum(datetime_1)';

    utime_2 = utimes(:, 2);
    datetime_2 = datetime(utime_2,'ConvertFrom','epochtime','TicksPerSecond',1,'Format','dd-MMM-yyyy HH:mm:ss');
    data.datevec_2 = datevec(datetime_2)';
    data.datenum_2 = datenum(datetime_2)';
    
    % Calculate T_e and error
    data.T_e = data.T_i .* data.T_r;
    data.T_e_err = data.T_e .* sqrt((data.T_i_err./data.T_i).^2 +   ...
        (data.T_r_err./data.T_r).^2);
    data.height = data.height / 1000.;
    data.range = data.range / 1000.;
    
    % load the metadata
    names = convertCharsToStrings(strtrim(h5read(file_path, '/metadata/names')));
    metadata.name_site = names{2, 2};
    metadata.name_expr = names{2, 1};
    metadata.site = site_info.(metadata.name_site);
    metadata.pulse_code = names{2, 1};
    metadata.antenna = names{2, 3};
    metadata.r_XMITloc = [data.r_XMITloc1 data.r_XMITloc2 data.r_XMITloc3/1000.];
    metadata.r_RECloc = [data.r_RECloc1 data.r_RECloc2 data.r_RECloc3/1000.];
    metadata.title = h5read(file_path, '/metadata/schemes/DataCite/Title');
    metadata.rawdata_path = h5read(file_path, '/metadata/software/gfd/data_path');
        
end

