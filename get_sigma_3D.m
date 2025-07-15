function [Keff, Geff] = get_sigma_3D(Lx, Ly, Lz, loadValue, loadType, nGrid, nTimeSteps, nIter, eIter, N, Y, porosity, needCPUcalc)
  % PHYSICS
  rho0 = 1.0;                         % density
  K0   = 1.0;                         % bulk modulus
  G0   = 0.01;                         % shear modulus
  coh  = Y * sqrt(3.0);
  P0 = 0.0; %1.0 * coh;
  %porosity = 0.005;
  rad = (0.75 * porosity * Lx * Ly * Lz / (pi * N ^ 3)) ^ (1 / 3);
  
  % NUMERICS
  Nx  = 8 * nGrid;     % number of space steps
  Ny  = 8 * nGrid;
  Nz  = 8 * nGrid;
  CFL = 0.5;     % Courant-Friedrichs-Lewy
  
  % PREPROCESSING
  dX     = Lx / (Nx - 1);                                   % space step
  dY     = Ly / (Ny - 1);
  dZ     = Lz / (Nz - 1);
  x      = (-Lx / 2) : dX : (Lx / 2);                       % space discretization
  y      = (-Ly / 2) : dY : (Ly / 2);
  z      = (-Lz / 2) : dZ : (Lz / 2);
  [x, y, z] = ndgrid(x, y, z);                                    % 3D mesh
  xCxy   = av4(x, 1);
  xCxz   = av4(x, 2);
  xCyz   = av4(x, 3);
  yCxy   = av4(y, 1);
  yCxz   = av4(y, 2);
  yCyz   = av4(y, 3);
  zCxy   = av4(z, 1);
  zCxz   = av4(z, 2);
  zCyz   = av4(z, 3);
  %radC   = sqrt(xC .* xC + yC .* yC);
  [xUx, yUx, zUx] = ndgrid((-(Lx + dX)/2) : dX : ((Lx + dX)/2), (-Ly/2) : dY : (Ly/2), (-Lz/2) : dZ : (Lz/2));
  [xUy, yUy, zUy] = ndgrid((-Lx/2) : dX : (Lx/2), (-(Ly + dY)/2) : dY : ((Ly + dY)/2), (-Lz/2) : dZ : (Lz/2));
  [xUz, yUz, zUz] = ndgrid((-Lx/2) : dX : (Lx/2), (-Ly/2) : dY : (Ly/2), (-(Lz + dZ)/2) : dZ : ((Lz + dZ)/2));
  dt      = CFL * min(min(dX, dY), dZ) / sqrt( (K0 + 4*G0/3) / rho0);    % time step
  dampX   = 4.0 / dt / Nx;
  dampY   = 4.0 / dt / Ny;
  dampZ   = 4.0 / dt / Nz;
  
  % INPUT FILES
  pa = [dX, dY, dZ, dt, K0, G0, rho0, dampX, dampY, dampZ, coh, rad, N];
  
  Keff = zeros(nTimeSteps);
  
  % parameters
  if not(isfolder('data'))
    mkdir 'data';
  end %if
  
  fil = fopen('data\pa.dat', 'wb');
  fwrite(fil, pa(:), 'double');
  fclose(fil);
  
  if needCPUcalc
    % MATERIALS
    K = zeros(Nx, Ny, Nz); %E ./ (3.0 * (1 - 2 * nu));             % bulk modulus
    G = zeros(Nx, Ny, Nz); %E ./ (2.0 + 2.0 * nu);                 % shear modulus
    [K, G] = set_mats_3D(N, Nx, Ny, Nz, Lx, Ly, Lz, x, y, z, rad, K0, G0);     % Young's modulus and Poisson's ratio
    
    % INITIAL CONDITIONS
    Pinit   = zeros(Nx, Ny, Nz);            % initial hydrostatic stress
    P       = zeros(Nx, Ny, Nz);
    tauxyAv = zeros(Nx, Ny, Nz);
    tauxzAv = zeros(Nx, Ny, Nz);
    tauyzAv = zeros(Nx, Ny, Nz);
    Pinit(sqrt(x.*x + y.*y + z.*z) < rad) = P0;    % hydrostatic stress (ball part of tensor)
    Ux    = zeros(Nx + 1, Ny, Nz);        % displacement
    Uy    = zeros(Nx, Ny + 1, Nz);
    Uz    = zeros(Nx, Ny, Nz + 1);
    Vx    = zeros(Nx + 1, Ny, Nz);        % velocity
    Vy    = zeros(Nx, Ny + 1, Nz);
    Vz    = zeros(Nx, Ny, Nz + 1);
    tauxx = zeros(Nx, Ny, Nz);            % deviatoric stress
    tauyy = zeros(Nx, Ny, Nz);
    tauzz = zeros(Nx, Ny, Nz);
    tauxy = zeros(Nx - 1, Ny - 1, Nz);
    tauxz = zeros(Nx - 1, Ny, Nz - 1);
    tauyz = zeros(Nx, Ny - 1, Nz - 1);
    J2 = zeros(Nx, Ny, Nz);
    J2xy = zeros(Nx - 1, Ny - 1, Nz);
    J2xz = zeros(Nx - 1, Ny, Nz - 1);
    J2yz = zeros(Nx, Ny - 1, Nz - 1);
    %Plast = zeros(Nx, Ny, Nz);
    %PlastXY = zeros(Nx - 1, Ny - 1, Nz);
    
    % BOUNDARY CONDITIONS
    dUxdx = loadValue * loadType(1);
    dUydy = loadValue * loadType(2);
    dUzdz = loadValue * loadType(3);
    dUxdy = loadValue * loadType(4);
    dUxdz = loadValue * loadType(5);
    dUydz = loadValue * loadType(6);
    
    % CPU CALCULATION
    for it = 1 : nTimeSteps
      Ux = Ux + (dUxdx * xUx + dUxdy * yUx + dUxdz * zUx) / nTimeSteps;
      Uy = Uy + (dUydy * yUy + dUydz * zUy) / nTimeSteps;
      Uz = Uz + (dUzdz * zUz) / nTimeSteps;

      error = 0.0;
      
      for iter = 1 : nIter
        % displacement divergence
        divU = diff(Ux,1,1) / dX + diff(Uy,1,2) / dY + diff(Uz,1,3) / dZ;
        
        % constitutive equation - Hooke's law
        P     = Pinit - K .* divU;
        %P     = P - G .* divU * dt / Nx;    % incompressibility
        tauxx = 2.0 * G .* (diff(Ux,1,1)/dX - divU/3.0);
        tauyy = 2.0 * G .* (diff(Uy,1,2)/dY - divU/3.0);
        tauzz = 2.0 * G .* (diff(Uz,1,3)/dZ - divU/3.0);
        tauxy = av4(G, 1) .* (diff(Ux(2:end-1, :, :), 1, 2)/dY + diff(Uy(:, 2:end-1, :), 1, 1)/dX);
        tauxz = av4(G, 2) .* (diff(Ux(2:end-1, :, :), 1, 3)/dZ + diff(Uz(:, :, 2:end-1), 1, 1)/dX);
        tauyz = av4(G, 3) .* (diff(Uy(:, 2:end-1, :), 1, 3)/dZ + diff(Uz(:, :, 2:end-1), 1, 2)/dY);
        
        for i = 0 : N - 1
          for j = 0 : N - 1
            for k = 0 : N - 1
              P((x - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .^ 2 + (y - 0.5*Ly*(1-1/N) + (Ly/N)*j) .^ 2 + (z - 0.5*Lz*(1-1/N) + (Lz/N)*k) .^ 2 < rad * rad)  = 0.0;
              tauxx((x - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .^ 2 + (y - 0.5*Ly*(1-1/N) + (Ly/N)*j) .^ 2 + (z - 0.5*Lz*(1-1/N) + (Lz/N)*k) .^ 2 < rad * rad)  = 0.0;
              tauyy((x - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .^ 2 + (y - 0.5*Ly*(1-1/N) + (Ly/N)*j) .^ 2 + (z - 0.5*Lz*(1-1/N) + (Lz/N)*k) .^ 2 < rad * rad)  = 0.0;
              tauzz((x - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .^ 2 + (y - 0.5*Ly*(1-1/N) + (Ly/N)*j) .^ 2 + (z - 0.5*Lz*(1-1/N) + (Lz/N)*k) .^ 2 < rad * rad)  = 0.0;
              tauxy((xCxy - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .^ 2 + (yCxy - 0.5*Ly*(1-1/N) + (Ly/N)*j) .^ 2 + (zCxy - 0.5*Lz*(1-1/N) + (Lz/N)*k) .^ 2 < rad * rad)  = 0.0;
              tauxz((xCxz - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .^ 2 + (yCxz - 0.5*Ly*(1-1/N) + (Ly/N)*j) .^ 2 + (zCxz - 0.5*Lz*(1-1/N) + (Lz/N)*k) .^ 2 < rad * rad)  = 0.0;
              tauyz((xCyz - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .^ 2 + (yCyz - 0.5*Ly*(1-1/N) + (Ly/N)*j) .^ 2 + (zCyz - 0.5*Lz*(1-1/N) + (Lz/N)*k) .^ 2 < rad * rad)  = 0.0;
            end % for(k)
          end % for(j)
        end % for(i)
        
        % tauIJ for plasticity
        tauxyAv(2:end-1, 2:end-1, :) = av4(tauxy, 1);        
        tauxyAv(1, 2:end-1, :) = tauxyAv(2, 2:end-1, :);
        tauxyAv(end, 2:end-1, :) = tauxyAv(end-1, 2:end-1, :);
        tauxyAv(2:end-1, 1, :) = tauxyAv(2:end-1, 2, :);
        tauxyAv(2:end-1, end, :) = tauxyAv(2:end-1, end-1, :);
        tauxyAv(1, 1, :) = 0.5 * (tauxyAv(1, 2, :) + tauxyAv(2, 1, :));
        tauxyAv(end, 1, :) = 0.5 * (tauxyAv(end, 2, :) + tauxyAv(end-1, 1, :));
        tauxyAv(1, end, :) = 0.5 * (tauxyAv(2, end, :) + tauxyAv(1, end-1, :));
        tauxyAv(end, end, :) = 0.5 * (tauxyAv(end, end-1, :) + tauxyAv(end-1, end, :));
        
        tauxzAv(2:end-1, :, 2:end-1) = av4(tauxz, 2);
        tauxzAv(1, :, 2:end-1) = tauxzAv(2, :, 2:end-1);
        tauxzAv(end, :, 2:end-1) = tauxzAv(end-1, :, 2:end-1);
        tauxzAv(2:end-1, :, 1) = tauxzAv(2:end-1, :, 2);
        tauxzAv(2:end-1, :, end) = tauxzAv(2:end-1, :, end-1);
        tauxzAv(1, :, 1) = 0.5 * (tauxzAv(1, :, 2) + tauxzAv(2, :, 1));
        tauxzAv(end, :, 1) = 0.5 * (tauxzAv(end, :, 2) + tauxzAv(end - 1, :, 1));
        tauxzAv(1, :, end) = 0.5 * (tauxzAv(2, :, end) + tauxzAv(1, :, end - 1));
        tauxzAv(end, :, end) = 0.5 * (tauxzAv(end, :, end-1) + tauxzAv(end-1, :, end));
        
        tauyzAv(:, 2:end-1, 2:end-1) = av4(tauyz, 3);
        tauyzAv(:, 1, 2:end-1) = tauyzAv(:, 2, 2:end-1);
        tauyzAv(:, end, 2:end-1) = tauyzAv(:, end-1, 2:end-1);
        tauyzAv(:, 2:end-1, 1) = tauyzAv(:, 2:end-1, 2);
        tauyzAv(:, 2:end-1, end) = tauyzAv(:, 2:end-1, end-1);
        tauyzAv(:, 1, 1) = 0.5 * (tauyzAv(:, 1, 2) + tauyzAv(:, 2, 1));
        tauyzAv(:, end, 1) = 0.5 * (tauyzAv(:, end, 2) + tauyzAv(:, end-1, 1));
        tauyzAv(:, 1, end) = 0.5 * (tauyzAv(:, 2, end) + tauyzAv(:, 1, end-1));
        tauyzAv(:, end, end) = 0.5 * (tauyzAv(:, end, end-1) + tauyzAv(:, end-1, end));
        
        % plasticity
        J2 = sqrt(tauxx .^ 2 + tauyy .^ 2 + tauzz .^ 2 + 2.0 * (tauxyAv .^ 2 + tauxzAv .^ 2 + tauyzAv .^ 2));    % Mises criteria
        
        % motion equation
        dVxdt = diff(-P(:,2:end-1,2:end-1) + tauxx(:,2:end-1,2:end-1), 1, 1)/dX / rho0 + (diff(tauxy(:,:,2:end-1),1,2)/dY + diff(tauxz(:, 2:end-1, :), 1, 3)/dZ) / rho0;
        Vx(2:end-1, 2:end-1, 2:end-1) = Vx(2:end-1, 2:end-1, 2:end-1) * (1 - dt * dampX) + dVxdt * dt;
        dVydt = diff(-P(2:end-1,:,2:end-1) + tauyy(2:end-1,:,2:end-1), 1, 2)/dY / rho0 + (diff(tauxy(:,:,2:end-1),1,1)/dX + diff(tauyz(2:end-1, :, :), 1, 3)/dZ) / rho0;
        Vy(2:end-1, 2:end-1, 2:end-1) = Vy(2:end-1, 2:end-1, 2:end-1) * (1 - dt * dampY) + dVydt * dt;
        dVzdt = diff(-P(2:end-1,2:end-1,:) + tauzz(2:end-1,2:end-1,:), 1, 3)/dZ / rho0 + (diff(tauyz(2:end-1,:,:),1,2)/dY + diff(tauxz(:, 2:end-1, :), 1, 1)/dX) / rho0;
        Vz(2:end-1, 2:end-1, 2:end-1) = Vz(2:end-1, 2:end-1, 2:end-1) * (1 - dt * dampZ) + dVzdt * dt;
        
        % displacements
        Ux = Ux + Vx * dt;
        Uy = Uy + Vy * dt;
        Uz = Uz + Vz * dt;
        
        % exit criteria
        if mod(iter, 100) == 0
          error = (max(abs(Vx(:))) / Lx + max(abs(Vy(:))) / Ly + max(abs(Vz(:))) / Lz) * dt / max(abs(loadValue * loadType));
          outStr = sprintf('Iteration %d: Error is %d', iter, error);
          disp(outStr);
          if error < eIter
            outStr = sprintf('Number of iterations is %d', iter);
            disp(outStr);
            break
          else
            if iter == nIter
              outStr = sprintf('WARNING: Maximum number of iterations reached!\nError is %d', error);
              disp(outStr);
            end
          end
        end %if
      end %for(iter)
    end %for(it)
    
    fil = fopen(strcat('data\PmXY_', int2str(Nx), '_.dat'), 'wb');
    fwrite(fil, P(:, :, end/2), 'double');
    fclose(fil);
    
    fil = fopen(strcat('data\PmXZ_', int2str(Nx), '_.dat'), 'wb');
    fwrite(fil, P(:, end/2, :), 'double');
    fclose(fil);
    
    fil = fopen(strcat('data\PmYZ_', int2str(Nx), '_.dat'), 'wb');
    fwrite(fil, P(end/2, :, :), 'double');
    fclose(fil);
    
    fil = fopen(strcat('data\J2mXY_', int2str(Nx), '_.dat'), 'wb');
    fwrite(fil, J2(:, :, end/2), 'double');
    fclose(fil);
    
    fil = fopen(strcat('data\tauXXm_', int2str(Nx), '_.dat'), 'wb');
    fwrite(fil, tauxx(:, :, end/2), 'double');
    fclose(fil);
    
    fil = fopen(strcat('data\tauYYm_', int2str(Nx), '_.dat'), 'wb');
    fwrite(fil, tauyy(:, :, end/2), 'double');
    fclose(fil);
    
    fil = fopen(strcat('data\tauXZmXY_', int2str(Nx), '_.dat'), 'wb');
    fwrite(fil, tauxz(:, :, (end-1)/2), 'double');
    fclose(fil);
    
    fil = fopen(strcat('data\UxmXY_', int2str(Nx), '_.dat'), 'wb');
    fwrite(fil, Ux(:, :, end/2), 'double');
    fclose(fil);
    
    fil = fopen(strcat('data\VxmXY_', int2str(Nx), '_.dat'), 'wb');
    fwrite(fil, Vx(:, :, end/2), 'double');
    fclose(fil);
    
  end %if
end % function