function [var_info_list] = list_EISCAT_hdf5_variables(varargin)
%% list variables in the EISCAT-type hdf5 file
%   Author: L. Cai.
%   Create date: 2021-07-20 
%   Input:
%       varargin{1}:  file_path (file_dir + file_name)
%       varargin{2}:  queried variable names, default - []: load all
%                     variables
%       varargin{3}:  variable group is known, input as a cell
%       varargin{4}:  display, [1] - display the list, 0 - not

   file_path = ''; 
   var_name_queried = [];
   var_groups = {};
   display = 1;
   nvar = length(varargin);
   if nvar >= 1
       file_path = varargin{1};
   end
   if nvar >= 2
       var_names = varargin{2};
   end
   if nvar >= 3
       var_groups = varargin{3};
   end
   if nvar == 4
       display = varargin{4};
   end
   
   if isempty(var_groups)
      var_groups = {'par0d', 'par1d', 'par2d'}; 
   end
   
    
   var_info_list.name = {};
   var_info_list.group = {};
   var_info_list.index = [];
   var_info_list.unit = {};
   var_info_list.name_GUISDAP = {};
   var_info_list.note = {};
   % get the variable info from metadata
   for i = 1 : length(var_groups)
       var_group = var_groups{i};
       metadata_var = convertCharsToStrings(    ...
           h5read(file_path, ['/metadata/' var_group]));
       [m, n] = size(metadata_var);
       for j = 1 : n
           var_name = convertCharsToStrings(strtrim(metadata_var{1, j}));
           if ~isempty(var_name_queried)
               if ~any(strcmp(var_name_queried, var_name))
                   continue
               end
           end
           var_unit = convertCharsToStrings(strtrim(metadata_var{3, j}));
           var_name_GUISDAP = convertCharsToStrings(strtrim(metadata_var{4, j}));
           var_note = convertCharsToStrings(strtrim(metadata_var{2, j}));
           
           var_info_list.name = [var_info_list.name var_name];
           var_info_list.group = [var_info_list.group var_group];
           var_info_list.index = [var_info_list.index j];
           var_info_list.unit = [var_info_list.unit var_unit];
           var_info_list.name_GUISDAP = [var_info_list.name_GUISDAP var_name_GUISDAP];
           var_info_list.note = [var_info_list.note var_note];
       end
   end
   len_vars = length(var_info_list.name);
   if len_vars == 0
       disp('Cannot find the queried variables!')
   end
   if display
       fprintf("%-20s%-10s%-10s%-20s%-20s%-60s\n",   ...
           'Name', 'Group', 'Index', 'Unit', 'Name (GUISDAP)', 'Note');
       for i=1 : len_vars
          fprintf('%-20s%-10s%-10d%-20s%-20s%-60s\n',    ...
          var_info_list.name{i}, var_info_list.group{i}, ...
          var_info_list.index(i), var_info_list.unit{i}, ...
          var_info_list.name_GUISDAP{i}, var_info_list.note{i});
       end
   end
        
end