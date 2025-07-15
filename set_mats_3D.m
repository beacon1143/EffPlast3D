function [K, G] = set_mats_3D(N, Nx, Ny, Nz, Lx, Ly, Lz, x, y, z, rad, K0, G0)
  %K0 = 10.0;
  %G0 = 0.01;
  %E0 = 9.0 * K0 * G0 / (3.0 * K0 + G0);
  %nu0 = 0.5 * (3.0 * K0 - 2.0 * G0) / (3.0 * K0 + G0);
  K = K0 * ones(Nx, Ny, Nz);
  G = G0 * ones(Nx, Ny, Nz);
  for i = 0 : N - 1
    for j = 0 : N - 1
      for k = 0 : N - 1
        K((x - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .^ 2 + (y - 0.5*Ly*(1-1/N) + (Ly/N)*j) .^ 2 + (z - 0.5*Lz*(1-1/N) + (Lz/N)*k) .^ 2 < rad * rad)  = 0.01 * K0;
        G((x - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .^ 2 + (y - 0.5*Ly*(1-1/N) + (Ly/N)*j) .^ 2 + (z - 0.5*Lz*(1-1/N) + (Lz/N)*k) .^ 2 < rad * rad)  = 0.01 * G0;
      end % for(k)
    end % for(j)
  end % for(i)
end % function