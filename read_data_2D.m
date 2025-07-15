function [field] = read_data_2D(var_name, size, nX, nY)
  file_name = strcat(var_name, '_', int2str(size), '_.dat');
  if isfile(file_name)
    fil = fopen(file_name, 'rb');
    field = fread(fil, 'double');
    fclose(fil);
    field = reshape(field, nX, nY);
    field = transpose(field);
  else
    disp(strjoin({'Error! File', file_name, 'does not exist!'}));
    field = zeros(nX, nY);
  end
end % function