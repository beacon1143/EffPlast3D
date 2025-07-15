function A_av = av4(A, i)
  if i == 1
    A_av = 0.25 * (A(1:end-1, 1:end-1, :) + A(1:end-1, 2:end, :) + A(2:end, 1:end-1, :) + A(2:end, 2:end, :));
  elseif i == 2
    A_av = 0.25 * (A(1:end-1, :, 1:end-1) + A(1:end-1, :, 2:end) + A(2:end, :, 1:end-1) + A(2:end, :, 2:end));
  else
    A_av = 0.25 * (A(:, 1:end-1, 1:end-1) + A(:, 1:end-1, 2:end) + A(:, 2:end, 1:end-1) + A(:, 2:end, 2:end));
  end % if
end