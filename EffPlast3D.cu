#include "EffPlast3D.h"

__global__ void ComputeDisp(double* Ux, double* Uy, double* Uz, double* Vx, double* Vy, double* Vz,
  const double* const P,
  const double* const tauXX, const double* const tauYY, const double* const tauZZ,
  const double* const tauXY, const double* const tauXZ, const double* const tauYZ,
  const double* const pa,
  const long int nX, const long int nY, const long int nZ)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  const double dX = pa[0], dY = pa[1], dZ = pa[2];
  const double dT = pa[3];
  const double rho = pa[6];
  const double dampX = pa[7], dampY = pa[8], dampZ = pa[9];

  // motion equation
  if (i > 0 && i < nX && j > 0 && j < nY - 1 && k > 0 && k < nZ - 1) {
    Vx[k * (nX + 1) * nY + j * (nX + 1) + i] = Vx[k * (nX + 1) * nY + j * (nX + 1) + i] * (1.0 - dT * dampX) + (dT / rho) * ((
      -P[k * nX * nY + j * nX + i] + P[k * nX * nY + j * nX + i - 1] + tauXX[k * nX * nY + j * nX + i] - tauXX[k * nX * nY + j * nX + i - 1]
      ) / dX +
      (
        tauXY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i - 1] - tauXY[k * (nX - 1) * (nY - 1) + (j - 1) * (nX - 1) + i - 1]
        ) / dY +
      (
        tauXZ[k * (nX - 1) * nY + j * (nX - 1) + i - 1] - tauXZ[(k - 1) * (nX - 1) * nY + j * (nX - 1) + i - 1]
        ) / dZ
      );
  }
  if (i > 0 && i < nX - 1 && j > 0 && j < nY && k > 0 && k < nZ - 1) {
    Vy[k * nX * (nY + 1) + j * nX + i] = Vy[k * nX * (nY + 1) + j * nX + i] * (1.0 - dT * dampY) + (dT / rho) * ((    // why dT * dampY ?
      -P[k * nX * nY + j * nX + i] + P[k * nX * nY + (j - 1) * nX + i] + tauYY[k * nX * nY + j * nX + i] - tauYY[k * nX * nY + (j - 1) * nX + i]
      ) / dY +
      (
        tauXY[k * (nX - 1) * (nY - 1) + (j - 1) * (nX - 1) + i] - tauXY[k * (nX - 1) * (nY - 1) + (j - 1) * (nX - 1) + i - 1]
        ) / dX +
      (
        tauYZ[k * nX * (nY - 1) + (j - 1) * nX + i] - tauYZ[(k - 1) * nX * (nY - 1) + (j - 1) * nX + i]
        ) / dZ
      );
  }
  if (i > 0 && i < nX - 1 && j > 0 && j < nY - 1 && k > 0 && k < nZ) {
    Vz[k * nX * nY + j * nX + i] = Vz[k * nX * nY + j * nX + i] * (1.0 - dT * dampZ) + (dT / rho) * ((
      -P[k * nX * nY + j * nX + i] + P[(k - 1) * nX * nY + j * nX + i] + tauZZ[k * nX * nY + j * nX + i] - tauZZ[(k - 1) * nX * nY + j * nX + i]
      ) / dZ +
      (
        tauXZ[(k - 1) * (nX - 1) * nY + j * (nX - 1) + i] - tauXZ[(k - 1) * (nX - 1) * nY + j * (nX - 1) + i - 1]
        ) / dX +
      (
        tauYZ[(k - 1) * nX * (nY - 1) + j * nX + i] - tauYZ[(k - 1) * nX * (nY - 1) + (j - 1) * nX + i]
        ) / dY
      );
  }

  Ux[k * (nX + 1) * nY + j * (nX + 1) + i] = Ux[k * (nX + 1) * nY + j * (nX + 1) + i] + Vx[k * (nX + 1) * nY + j * (nX + 1) + i] * dT;
  Uy[k * nX * (nY + 1) + j * nX + i] = Uy[k * nX * (nY + 1) + j * nX + i] + Vy[k * nX * (nY + 1) + j * nX + i] * dT;
  Uz[k * nX * nY + j * nX + i] = Uz[k * nX * nY + j * nX + i] + Vz[k * nX * nY + j * nX + i] * dT;
}

__global__ void ComputeStress(const double* const Ux, const double* const Uy, const double* Uz,
  const double* const K, const double* const G,
  const double* const P0, double* P,
  double* tauXX, double* tauYY, double* tauZZ,
  double* tauXY, double* tauXZ, double* tauYZ,
  const double* const pa,
  const long int nX, const long int nY, const long int nZ)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  const double dX = pa[0], dY = pa[1], dZ = pa[2];
  // const double dT = pa[2];
  const double rad = pa[11];
  const double N = pa[12];

  // constitutive equation - Hooke's law
  P[k * nX * nY + j * nX + i] = P0[k * nX * nY + j * nX + i] - K[k * nX * nY + j * nX + i] * (
    (Ux[k * (nX + 1) * nY + j * (nX + 1) + i + 1] - Ux[k * (nX + 1) * nY + j * (nX + 1) + i]) / dX +     // divU
    (Uy[k * nX * (nY + 1) + (j + 1) * nX + i] - Uy[k * nX * (nY + 1) + j * nX + i]) / dY +
    (Uz[(k + 1) * nX * nY + j * nX + i] - Uz[k * nX * nY + j * nX + i]) / dZ
    );

  /*P[j * nX + i] = P[j * nX + i] - G[j * nX + i] * ( // incompressibility
  (Ux[j * (nX + 1) + i + 1] - Ux[j * (nX + 1) + i]) / dX + (Uy[(j + 1) * nX + i] - Uy[j * nX + i]) / dY    // divU
  ) * dT / nX;*/

  tauXX[k * nX * nY + j * nX + i] = 2.0 * G[k * nX * nY + j * nX + i] * (
    (Ux[k * (nX + 1) * nY + j * (nX + 1) + i + 1] - Ux[k * (nX + 1) * nY + j * (nX + 1) + i]) / dX -    // dUx/dx
      (
        (Ux[k * (nX + 1) * nY + j * (nX + 1) + i + 1] - Ux[k * (nX + 1) * nY + j * (nX + 1) + i]) / dX +
        (Uy[k * nX * (nY + 1) + (j + 1) * nX + i] - Uy[k * nX * (nY + 1) + j * nX + i]) / dY +
        (Uz[(k + 1) * nX * nY + j * nX + i] - Uz[k * nX * nY + j * nX + i]) / dZ
      ) / 3.0    // divU / 3.0
    );
  tauYY[k * nX * nY + j * nX + i] = 2.0 * G[k * nX * nY + j * nX + i] * (
    (Uy[k * nX * (nY + 1) + (j + 1) * nX + i] - Uy[k * nX * (nY + 1) + j * nX + i]) / dY -    // dUy/dy
      (
        (Ux[k * (nX + 1) * nY + j * (nX + 1) + i + 1] - Ux[k * (nX + 1) * nY + j * (nX + 1) + i]) / dX +
        (Uy[k * nX * (nY + 1) + (j + 1) * nX + i] - Uy[k * nX * (nY + 1) + j * nX + i]) / dY +
        (Uz[(k + 1) * nX * nY + j * nX + i] - Uz[k * nX * nY + j * nX + i]) / dZ
      ) / 3.0    // divU / 3.0
    );
  tauZZ[k * nX * nY + j * nX + i] = 2.0 * G[k * nX * nY + j * nX + i] * (
    (Uz[(k + 1) * nX * nY + j * nX + i] - Uz[k * nX * nY + j * nX + i]) / dZ -    // dUz/dz
      (
        (Ux[k * (nX + 1) * nY + j * (nX + 1) + i + 1] - Ux[k * (nX + 1) * nY + j * (nX + 1) + i]) / dX +
        (Uy[k * nX * (nY + 1) + (j + 1) * nX + i] - Uy[k * nX * (nY + 1) + j * nX + i]) / dY +
        (Uz[(k + 1) * nX * nY + j * nX + i] - Uz[k * nX * nY + j * nX + i]) / dZ
      ) / 3.0    // divU / 3.0
    );

  if (i < nX - 1 && j < nY - 1) {
    tauXY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i] = 0.25 * (
        G[k * nX * nY + j * nX + i] + G[k * nX * nY + j * nX + i + 1] + G[k * nX * nY + (j + 1) * nX + i] + G[k * nX * nY + (j + 1) * nX + i + 1]
      ) * (
        (Ux[k * (nX + 1) * nY + (j + 1) * (nX + 1) + i + 1] - Ux[k * (nX + 1) * nY + j * (nX + 1) + i + 1]) / dY + 
        (Uy[k * nX * (nY + 1) + (j + 1) * nX + i + 1] - Uy[k * nX * (nY + 1) + (j + 1) * nX + i]) / dX    // dUx/dy + dUy/dx
      );
  }
  if (i < nX - 1 && k < nZ - 1) {
    tauXZ[k * (nX - 1) * nY + j * (nX - 1) + i] = 0.25 * (
        G[k * nX * nY + j * nX + i] + G[k * nX * nY + j * nX + i + 1] + G[(k + 1) * nX * nY + j * nX + i] + G[(k + 1) * nX * nY + j * nX + i + 1]
      ) * (
        (Ux[(k + 1) * (nX + 1) * nY + j * (nX + 1) + i + 1] - Ux[k * (nX + 1) * nY + j * (nX + 1) + i + 1]) / dZ +
        (Uz[(k + 1) * nX * nY + j * nX + i + 1] - Uz[(k + 1) * nX * nY + j * nX + i]) / dX    // dUx/dz + dUz/dx
      );
  }
  if (j < nY - 1 && k < nZ - 1) {
    tauYZ[k * nX * (nY - 1) + j * nX + i] = 0.25 * (
        G[k * nX * nY + j * nX + i] + G[k * nX * nY + (j + 1) * nX + i] + G[(k + 1) * nX * nY + j * nX + i] + G[(k + 1) * nX * nY + (j + 1) * nX + i]
      ) * (
        (Uy[(k + 1) * nX * (nY + 1) + (j + 1) * nX + i] - Uy[k * nX * (nY + 1) + (j + 1) * nX + i]) / dZ +
        (Uz[(k + 1) * nX * nY + (j + 1) * nX + i] - Uz[(k + 1) * nX * nY + j * nX + i]) / dY    // dUy/dz + dUz/dy
      );
  }

  for (int a = 0; a < N; a++) {
    for (int b = 0; b < N; b++) {
      for (int c = 0; c < N; c++) {
        if (sqrt((-0.5 * dX * (nX - 1) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * a) *
          (-0.5 * dX * (nX - 1) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * a) +
          (-0.5 * dY * (nY - 1) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * b) *
          (-0.5 * dY * (nY - 1) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * b) +
          (-0.5 * dZ * (nZ - 1) + dZ * k - 0.5 * dZ * (nZ - 1) * (1.0 - 1.0 / N) + (dZ * (nZ - 1) / N) * c) * 
          (-0.5 * dZ * (nZ - 1) + dZ * k - 0.5 * dZ * (nZ - 1) * (1.0 - 1.0 / N) + (dZ * (nZ - 1) / N) * c)) < rad) {
          P[k * nX * nY + j * nX + i] = 0.0;
          tauXX[k * nX * nY + j * nX + i] = 0.0;
          tauYY[k * nX * nY + j * nX + i] = 0.0;
          tauZZ[k * nX * nY + j * nX + i] = 0.0;
        }

        if (i < nX - 1 && j < nY - 1) {
          if (sqrt((-0.5 * dX * (nX - 2) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * a) *
            (-0.5 * dX * (nX - 2) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * a) +
            (-0.5 * dY * (nY - 2) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * b) *
            (-0.5 * dY * (nY - 2) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * b) +
            (-0.5 * dZ * (nZ - 1) + dZ * k - 0.5 * dZ * (nZ - 1) * (1.0 - 1.0 / N) + (dZ * (nZ - 1) / N) * c) * 
            (-0.5 * dZ * (nZ - 1) + dZ * k - 0.5 * dZ * (nZ - 1) * (1.0 - 1.0 / N) + (dZ * (nZ - 1) / N) * c)) < rad) {
            tauXY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i] = 0.0;
          }
        }
        if (i < nX - 1 && k < nZ - 1) {
          if (sqrt((-0.5 * dX * (nX - 2) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * a) *
            (-0.5 * dX * (nX - 2) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * a) +
            (-0.5 * dY * (nY - 1) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * b) *
            (-0.5 * dY * (nY - 1) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * b) +
            (-0.5 * dZ * (nZ - 2) + dZ * k - 0.5 * dZ * (nZ - 1) * (1.0 - 1.0 / N) + (dZ * (nZ - 1) / N) * c) * 
            (-0.5 * dZ * (nZ - 2) + dZ * k - 0.5 * dZ * (nZ - 1) * (1.0 - 1.0 / N) + (dZ * (nZ - 1) / N) * c)) < rad) {
            tauXZ[k * (nX - 1) * nY + j * (nX - 1) + i] = 0.0;
          }
        }
        if (j < nY - 1 && k < nZ - 1) {
          if (sqrt((-0.5 * dX * (nX - 1) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * a) *
            (-0.5 * dX * (nX - 1) + dX * i - 0.5 * dX * (nX - 1) * (1.0 - 1.0 / N) + (dX * (nX - 1) / N) * a) +
            (-0.5 * dY * (nY - 2) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * b) *
            (-0.5 * dY * (nY - 2) + dY * j - 0.5 * dY * (nY - 1) * (1.0 - 1.0 / N) + (dY * (nY - 1) / N) * b) +
            (-0.5 * dZ * (nZ - 2) + dZ * k - 0.5 * dZ * (nZ - 1) * (1.0 - 1.0 / N) + (dZ * (nZ - 1) / N) * c) * 
            (-0.5 * dZ * (nZ - 2) + dZ * k - 0.5 * dZ * (nZ - 1) * (1.0 - 1.0 / N) + (dZ * (nZ - 1) / N) * c)) < rad) {
            tauYZ[k * nX * (nY - 1) + j * nX + i] = 0.0;
          }
        }
      } // for(c)
    } // for(b)
  } // for(a)
}

__global__ void ComputeJ2(double* tauXX, double* tauYY, double* tauZZ,
  double* tauXY, double* tauXZ, double* tauYZ,
  double* const tauXYav, double* const tauXZav, double* const tauYZav,
  double* const J2, double* const J2XY, double* const J2XZ, double* const J2YZ,
  const long int nX, const long int nY, const long int nZ)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  // tauIJ for plasticity
  if (i > 0 && i < nX - 1 && j > 0 && j < nY - 1) {
    tauXYav[k * nX * nY + j * nX + i] = 0.25 * (
      tauXY[k * (nX - 1) * (nY - 1) + (j - 1) * (nX - 1) + i - 1] + tauXY[k * (nX - 1) * (nY - 1) + (j - 1) * (nX - 1) + i] + 
      tauXY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i - 1] + tauXY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i]
    );
  }
  if (i > 0 && i < nX - 1 && k > 0 && k < nZ - 1) {
    tauXZav[k * nX * nY + j * nX + i] = 0.25 * (
      tauXZ[(k - 1) * (nX - 1) * nY + j * (nX - 1) + i - 1] + tauXZ[(k - 1) * (nX - 1) * nY + j * (nX - 1) + i] +
      tauXZ[k * (nX - 1) * nY + j * (nX - 1) + i - 1] + tauXZ[k * (nX - 1) * nY + j * (nX - 1) + i]
    );
  }
  if (j > 0 && j < nY - 1 && k > 0 && k < nZ - 1) {
    tauYZav[k * nX * nY + j * nX + i] = 0.25 * (
      tauYZ[(k - 1) * nX * (nY - 1) + (j - 1) * nX + i] + tauYZ[(k - 1) * nX * (nY - 1) + j * nX + i] +
      tauYZ[k * nX * (nY - 1) + (j - 1) * nX + i] + tauYZ[k * nX * (nY - 1) + j * nX + i]
    );
  }
  J2[k * nX * nY + j * nX + i] = sqrt(
    tauXX[k * nX * nY + j * nX + i] * tauXX[k * nX * nY + j * nX + i] + 
    tauYY[k * nX * nY + j * nX + i] * tauYY[k * nX * nY + j * nX + i] +
    tauZZ[k * nX * nY + j * nX + i] * tauZZ[k * nX * nY + j * nX + i] +
    2.0 * (
      tauXYav[k * nX * nY + j * nX + i] * tauXYav[k * nX * nY + j * nX + i] +
      tauXZav[k * nX * nY + j * nX + i] * tauXZav[k * nX * nY + j * nX + i] +
      tauYZav[k * nX * nY + j * nX + i] * tauYZav[k * nX * nY + j * nX + i]
    )
  );
  if (i < nX - 1 && j < nY - 1 && k > 0 && k < nZ - 1) {
    J2XY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i] = sqrt(
      pow(0.25 * (tauXX[k * nX * nY + (j + 1) * nX + i + 1] + tauXX[k * nX * nY + (j + 1) * nX + i] + tauXX[k * nX * nY + j * nX + i + 1] + tauXX[k * nX * nY + j * nX + i]), 2.0) +
      pow(0.25 * (tauYY[k * nX * nY + (j + 1) * nX + i + 1] + tauYY[k * nX * nY + (j + 1) * nX + i] + tauYY[k * nX * nY + j * nX + i + 1] + tauYY[k * nX * nY + j * nX + i]), 2.0) +
      pow(0.25 * (tauZZ[k * nX * nY + (j + 1) * nX + i + 1] + tauZZ[k * nX * nY + (j + 1) * nX + i] + tauZZ[k * nX * nY + j * nX + i + 1] + tauZZ[k * nX * nY + j * nX + i]), 2.0) +
      2.0 * (
        pow(tauXY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i], 2.0) +
        pow(0.25 * (
          tauXZ[(k - 1) * (nX - 1) * nY + j * (nX - 1) + i] + tauXZ[(k - 1) * (nX - 1) * nY + (j + 1) * (nX - 1) + i] + 
          tauXZ[k * (nX - 1) * nY + j * (nX - 1) + i] + tauXZ[k * (nX - 1) * nY + (j + 1) * (nX - 1) + i]
        ), 2.0) +
        pow(0.25 * (
          tauYZ[(k - 1) * nX * (nY - 1) + j * nX + i] + tauYZ[(k - 1) * nX * (nY - 1) + j * nX + i + 1] +
          tauYZ[k * nX * (nY - 1) + j * nX + i] + tauYZ[k * nX * (nY - 1) + j * nX + i + 1]
        ), 2.0)
      )
    ); // sqrt
  }
  if (i < nX - 1 && j > 0 && j < nY - 1 && k < nZ - 1) {
    J2XZ[k * (nX - 1) * nY + j * (nX - 1) + i] = sqrt(
      pow(0.25 * (
        tauXX[k * nX * nY + j * nX + i + 1] + tauXX[k * nX * nY + j * nX + i] + tauXX[(k + 1) * nX * nY + j * nX + i + 1] + tauXX[(k + 1) * nX * nY + j * nX + i]
        ), 2.0) +
      pow(0.25 * (
        tauYY[k * nX * nY + j * nX + i + 1] + tauYY[k * nX * nY + j * nX + i] + tauYY[(k + 1) * nX * nY + j * nX + i + 1] + tauYY[(k + 1) * nX * nY + j * nX + i]
        ), 2.0) +
      pow(0.25 * (
        tauZZ[k * nX * nY + j * nX + i + 1] + tauZZ[k * nX * nY + j * nX + i] + tauZZ[(k + 1) * nX * nY + j * nX + i + 1] + tauZZ[(k + 1) * nX * nY + j * nX + i]
        ), 2.0) +
      2.0 * (
        pow(tauXZ[k * (nX - 1) * nY + j * (nX - 1) + i], 2.0) +
        pow(0.25 * (
          tauXY[k * (nX - 1) * (nY - 1) + (j - 1) * (nX - 1) + i] + tauXY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i] +
          tauXY[(k + 1) * (nX - 1) * (nY - 1) + (j - 1) * (nX - 1) + i] + tauXY[(k + 1) * (nX - 1) * (nY - 1) + j * (nX - 1) + i]
        ), 2.0) +
        pow(0.25 * (
          tauYZ[k * nX * (nY - 1) + (j - 1) * nX + i] + tauYZ[k * nX * (nY - 1) + (j - 1) * nX + i + 1] +
          tauYZ[k * nX * (nY - 1) + j * nX + i] + tauYZ[k * nX * (nY - 1) + j * nX + i + 1]
        ), 2.0)
      )
    ); // sqrt
  }
  if (i > 0 && i < nX - 1 && j < nY - 1 && k < nZ - 1) {
    J2YZ[k * nX * (nY - 1) + j * nX + i] = sqrt(
      pow(0.25 * (
        tauXX[(k + 1) * nX * nY + j * nX + i] + tauXX[k * nX * nY + j * nX + i] + tauXX[(k + 1) * nX * nY + (j + 1) * nX + i] + tauXX[k * nX * nY + (j + 1) * nX + i]
        ), 2.0) +
      pow(0.25 * (
        tauYY[(k + 1) * nX * nY + j * nX + i] + tauYY[k * nX * nY + j * nX + i] + tauYY[(k + 1) * nX * nY + (j + 1) * nX + i] + tauYY[k * nX * nY + (j + 1) * nX + i]
        ), 2.0) +
      pow(0.25 * (
        tauZZ[(k + 1) * nX * nY + j * nX + i] + tauZZ[k * nX * nY + j * nX + i] + tauZZ[(k + 1) * nX * nY + (j + 1) * nX + i] + tauZZ[k * nX * nY + (j + 1) * nX + i]
        ), 2.0) +
      2.0 * (
        pow(tauYZ[k * nX * (nY - 1) + j * nX + i], 2.0) +
        pow(0.25 * (
          tauXY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i - 1] + tauXY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i] +
          tauXY[(k + 1) * (nX - 1) * (nY - 1) + j * (nX - 1) + i - 1] + tauXY[(k + 1) * (nX - 1) * (nY - 1) + j * (nX - 1) + i]
        ), 2.0) +
        pow(0.25 * (
          tauXZ[k * (nX - 1) * nY + j * (nX - 1) + i - 1] + tauXZ[k * (nX - 1) * nY + j * (nX - 1) + i] +
          tauXZ[k * (nX - 1) * nY + (j + 1) * (nX - 1) + i - 1] + tauXZ[k * (nX - 1) * nY + (j + 1) * (nX - 1) + i]
        ), 2.0)
      )
    ); // sqrt
  }
}

__global__ void ComputePlasticity(double* tauXX, double* tauYY, double* tauZZ,
  double* tauXY, double* tauXZ, double* tauYZ,
  double* const tauXYav, double* const tauXZav, double* const tauYZav,
  double* const J2, double* const J2XY, double* const J2XZ, double* const J2YZ,
  const long int nX, const long int nY, const long int nZ)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  // plasticity
  if (J2[k * nX * nY + j * nX + i] > coh_cuda) {
    tauXX[k * nX * nY + j * nX + i] *= coh_cuda / J2[k * nX * nY + j * nX + i];
    tauYY[k * nX * nY + j * nX + i] *= coh_cuda / J2[k * nX * nY + j * nX + i];
    tauZZ[k * nX * nY + j * nX + i] *= coh_cuda / J2[k * nX * nY + j * nX + i];
    tauXYav[k * nX * nY + j * nX + i] *= coh_cuda / J2[k * nX * nY + j * nX + i];
    tauXZav[k * nX * nY + j * nX + i] *= coh_cuda / J2[k * nX * nY + j * nX + i];
    tauYZav[k * nX * nY + j * nX + i] *= coh_cuda / J2[k * nX * nY + j * nX + i];
  }

  if (i < nX - 1 && j < nY - 1) {
    if (J2XY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i] > coh_cuda) {
      tauXY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i] *= coh_cuda / J2XY[k * (nX - 1) * (nY - 1) + j * (nX - 1) + i];
    }
  }
  if (i < nX - 1 && k < nZ - 1) {
    if (J2XZ[k * (nX - 1) * nY + j * (nX - 1) + i] > coh_cuda) {
      tauXZ[k * (nX - 1) * nY + j * (nX - 1) + i] *= coh_cuda / J2XZ[k * (nX - 1) * nY + j * (nX - 1) + i];
    }
  }
  if (j < nY - 1 && k < nZ - 1) {
    if (J2YZ[k * nX * (nY - 1) + j * nX + i] > coh_cuda) {
      tauYZ[k * nX * (nY - 1) + j * nX + i] *= coh_cuda / J2YZ[k * nX * (nY - 1) + j * nX + i];
    }
  }
  
  // recalculation J2 for correct visualization only
  J2[k * nX * nY + j * nX + i] = sqrt(
    tauXX[k * nX * nY + j * nX + i] * tauXX[k * nX * nY + j * nX + i] + 
    tauYY[k * nX * nY + j * nX + i] * tauYY[k * nX * nY + j * nX + i] +
    tauZZ[k * nX * nY + j * nX + i] * tauZZ[k * nX * nY + j * nX + i] +
    2.0 * (
      tauXYav[k * nX * nY + j * nX + i] * tauXYav[k * nX * nY + j * nX + i] +
      tauXZav[k * nX * nY + j * nX + i] * tauXZav[k * nX * nY + j * nX + i] +
      tauYZav[k * nX * nY + j * nX + i] * tauYZav[k * nX * nY + j * nX + i]
      )
  );
}

double EffPlast3D::ComputeEffModuli(const double initLoadValue, [[deprecated]] const double loadValue, 
  const unsigned int nTimeSteps, const std::array<double, 6>& loadType)
{
  if (nPores <= 0) {
    throw std::invalid_argument("Error! The number of pores must be positive!\n");
  }

  const auto start = std::chrono::system_clock::now();
  nTimeSteps_ = nTimeSteps;
  loadType_ = loadType;

  std::array<double, 6> sphericalLoadType{
    (loadType_[0] + loadType_[1] + loadType_[2]) / 3.0,
    (loadType_[0] + loadType_[1] + loadType_[2]) / 3.0,
    (loadType_[0] + loadType_[1] + loadType_[2]) / 3.0,
    0.0, 0.0, 0.0
  };
  //std::array<double, 6> deviatoricLoadType{loadType_[0] - sphericalLoadType[0], loadType_[1] - sphericalLoadType[1], loadType_[2]};

  printCalculationType();

  ComputeEffParams(0, initLoadValue, loadType_, nTimeSteps_);
  if (NL == 1) {
    calcBulkModuli_PureElast();
  }
  else {
    ComputeEffParams(1, initLoadValue * incPercent, sphericalLoadType, 1);
    calcBulkModuli_ElastPlast();
  }

  /*if (NL == 3) {
    ComputeEffParams(2, initLoadValue * incPercent, deviatoricLoadType, 1);
    calcShearModulus();
  }

  printEffectiveModuli();
  printWarnings();
  */

  /* OUTPUT DATA WRITING */
  SaveSlice(P_cpu, P_cuda, nX, nY, nZ, nZ / 2, "data/PcXY_" + std::to_string(8 * NGRID) + "_.dat");
  SaveSlice(tauXX_cpu, tauXX_cuda, nX, nY, nZ, nZ / 2, "data/tauXXc_" + std::to_string(8 * NGRID) + "_.dat");
  SaveSlice(tauXZ_cpu, tauXZ_cuda, nX - 1, nY, nZ - 1, nZ / 2, "data/tauXZcXY_" + std::to_string(8 * NGRID) + "_.dat");
  if (NL > 1) {
    SaveSlice(J2_cpu, J2_cuda, nX, nY, nZ, nZ / 2, "data/J2cXY_" + std::to_string(8 * NGRID) + "_.dat");
    SaveSlice(J2XY_cpu, J2XY_cuda, nX - 1, nY - 1, nZ, nZ / 2, "data/J2XYcXY_" + std::to_string(8 * NGRID) + "_.dat");
    SaveSlice(J2XZ_cpu, J2XZ_cuda, nX - 1, nY, nZ - 1, nZ / 2, "data/J2XZcXY_" + std::to_string(8 * NGRID) + "_.dat");
    SaveSlice(J2YZ_cpu, J2YZ_cuda, nX, nY - 1, nZ - 1, nZ / 2, "data/J2YZcXY_" + std::to_string(8 * NGRID) + "_.dat");
  }
  SaveSlice(Ux_cpu, Ux_cuda, nX + 1, nY, nZ, nZ / 2, "data/UxcXY_" + std::to_string(8 * NGRID) + "_.dat");
  SaveSlice(Vx_cpu, Vx_cuda, nX + 1, nY, nZ, nZ / 2, "data/VxcXY_" + std::to_string(8 * NGRID) + "_.dat");
  //SaveMatrix(tauYY_cpu, tauYY_cuda, nX, nY, "data/tauYYc_" + std::to_string(32 * NGRID) + "_.dat");
  //SaveMatrix(tauXYav_cpu, tauXYav_cuda, nX, nY, "data/tauXYavc_" + std::to_string(32 * NGRID) + "_.dat");
  //SaveMatrix(J2_cpu, J2_cuda, nX, nY, "data/J2c_" + std::to_string(32 * NGRID) + "_.dat");
  //SaveMatrix(Uy_cpu, Uy_cuda, nX, nY + 1, "data/Uyc_" + std::to_string(32 * NGRID) + "_.dat");

  /*const double tauXYmax = findMaxAbs(tauXZ_cpu, (nX - 1) * nY * (nZ - 1));
  std::cout << "tauXYmax = " << tauXYmax << "\n";*/

  //gpuErrchk(cudaDeviceReset());
  const auto end = std::chrono::system_clock::now();
  int elapsed_sec = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(end - start).count());
  printDuration(elapsed_sec);

  return 0.0;
}

void EffPlast3D::ComputeEffParams(const size_t step, const double loadStepValue, const std::array<double, 6>& loadType, const size_t nTimeSteps) {
  printStepInfo(step);

  PeffNonper[step].resize(nTimeSteps);
  PeffPer[step].resize(nTimeSteps);
  /*tauInfty[step].resize(nTimeSteps);*/
  dPhiNonper[step].resize(nTimeSteps);
  dPhiPer[step].resize(nTimeSteps);
  /*epsilon[step].resize(nTimeSteps);
  epsilonPer[step].resize(nTimeSteps);
  sigma[step].resize(nTimeSteps);
  sigmaPer[step].resize(nTimeSteps);*/

  double dUxdx = 0.0;
  double dUydy = 0.0;
  double dUzdz = 0.0;
  double dUxdy = 0.0;
  double dUxdz = 0.0;
  double dUydz = 0.0;

  if (step == 0) {
    curEffStrain = { 0.0 };
    memset(Ux_cpu, 0, (nX + 1) * nY * nZ * sizeof(double));
    memset(Uy_cpu, 0, nX * (nY + 1) * nZ * sizeof(double));
    memset(Uz_cpu, 0, nX * nY * (nZ + 1) * sizeof(double));
  }
  else { // additional loading
    gpuErrchk(cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * nZ * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * nZ * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(Uz_cpu, Uz_cuda, nX * nY * (nZ + 1) * sizeof(double), cudaMemcpyDeviceToHost));
  }

  /* ACTION LOOP */
  for (int it = 0; it < nTimeSteps; it++) {
    std::cout << "Time step " << (it + 1) << " from " << nTimeSteps << "\n";
    log_file << "Time step " << (it + 1) << " from " << nTimeSteps << "\n";

    /*epsilon[step][it] = { 0.0 };
    epsilonPer[step][it] = { 0.0 };
    sigma[step][it] = { 0.0 };
    sigmaPer[step][it] = { 0.0 };*/

    dUxdx = loadStepValue * loadType[0] / static_cast<double>(nTimeSteps);
    dUydy = loadStepValue * loadType[1] / static_cast<double>(nTimeSteps);
    dUzdz = loadStepValue * loadType[2] / static_cast<double>(nTimeSteps);
    dUxdy = loadStepValue * loadType[3] / static_cast<double>(nTimeSteps);
    dUxdz = loadStepValue * loadType[4] / static_cast<double>(nTimeSteps);
    dUydz = loadStepValue * loadType[5] / static_cast<double>(nTimeSteps);
    //dUydx = dUxdy;

    curEffStrain[0] += dUxdx;
    curEffStrain[1] += dUydy;
    curEffStrain[2] += dUzdz;
    curEffStrain[3] += dUxdy;
    curEffStrain[4] += dUxdz;
    curEffStrain[5] += dUydz;
    //epsilon[step][it] = curEffStrain;

    std::cout << "Macro strain: (" << curEffStrain[0] << ", " << curEffStrain[1] << ", " << curEffStrain[2] << ", " << curEffStrain[3] << ", " << curEffStrain[4] << ", " << curEffStrain[5] << ")\n";
    log_file << "Macro strain: (" << curEffStrain[0] << ", " << curEffStrain[1] << ", " << curEffStrain[2] << ", " << curEffStrain[3] << ", " << curEffStrain[4] << ", " << curEffStrain[5] << ")\n";

    if (it > 0) {    // non-first time step
      gpuErrchk(cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * nZ * sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * nZ * sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(Uz_cpu, Uz_cuda, nX * nY * (nZ + 1) * sizeof(double), cudaMemcpyDeviceToHost));
    }

    //std::cout << "Ux = " << Ux_cpu[(3 * nY / 4) * (nX + 1) + 3 * nX / 4] << "\nUy = " << Uy_cpu[(3 * nY / 4) * nX + 3 * nX / 4] << "\n";

    for (int i = 0; i < nX + 1; i++) {
      for (int j = 0; j < nY; j++) {
        for (int k = 0; k < nZ; k++) {
          Ux_cpu[k * (nX + 1) * nY + j * (nX + 1) + i] += (-0.5 * dX * nX + dX * i) * dUxdx + (-0.5 * dY * (nY - 1) + dY * j) * dUxdy + (-0.5 * dZ * (nZ - 1) + dZ * k) * dUxdz;
        }
      }
    }
    gpuErrchk(cudaMemcpy(Ux_cuda, Ux_cpu, (nX + 1) * nY * nZ * sizeof(double), cudaMemcpyHostToDevice));
    for (int i = 0; i < nX; i++) {
      for (int j = 0; j < nY + 1; j++) {
        for (int k = 0; k < nZ; k++) {
          Uy_cpu[k * nX * (nY + 1) + j * nX + i] += (-0.5 * dY * nY + dY * j) * dUydy + (-0.5 * dZ * (nZ - 1) + dZ * k) * dUydz;
        }
      }
    }
    gpuErrchk(cudaMemcpy(Uy_cuda, Uy_cpu, nX * (nY + 1) * nZ * sizeof(double), cudaMemcpyHostToDevice));
    for (int i = 0; i < nX; i++) {
      for (int j = 0; j < nY; j++) {
        for (int k = 0; k < nZ + 1; k++) {
          Uz_cpu[k * nX * nY + j * nX + i] += (-0.5 * dZ * nZ + dZ * k) * dUzdz;
        }
      }
    }
    gpuErrchk(cudaMemcpy(Uz_cuda, Uz_cpu, nX * nY * (nZ + 1) * sizeof(double), cudaMemcpyHostToDevice));

    //std::cout << "dUxdx = " << dUxdx << "\ndUydy = " << dUydy << "\ndUxdy = " << dUxdy << "\n";
    //std::cout << "Ux = " << Ux_cpu[(3 * nY / 4) * (nX + 1) /*+ 3 * nX / 4*/] << "\nUy = " << Uy_cpu[(3 * nY / 4) * nX /*+ 3 * nX / 4*/] << "\n";

    double error = 0.0;

    /* ITERATION LOOP */
    for (int iter = 0; iter < NITER; iter++) {
      ComputeStress<<<grid, block>>>(Ux_cuda, Uy_cuda, Uz_cuda,
        K_cuda, G_cuda, P0_cuda, P_cuda,
        tauXX_cuda, tauYY_cuda, tauZZ_cuda,
        tauXY_cuda, tauXZ_cuda, tauYZ_cuda,
        /*tauXYav_cuda, J2_cuda, J2XY_cuda,*/ pa_cuda, nX, nY, nZ);
      gpuErrchk(cudaDeviceSynchronize());
      
      if (NL > 1) {
        ComputeJ2<<<grid, block>>>(tauXX_cuda, tauYY_cuda, tauZZ_cuda,
          tauXY_cuda, tauXZ_cuda, tauYZ_cuda,
          tauXYav_cuda, tauXZav_cuda, tauYZav_cuda,
          J2_cuda, J2XY_cuda, J2XZ_cuda, J2YZ_cuda,
          nX, nY, nZ);
        gpuErrchk(cudaDeviceSynchronize());
        ComputePlasticity<<<grid, block>>>(tauXX_cuda, tauYY_cuda, tauZZ_cuda,
          tauXY_cuda, tauXZ_cuda, tauYZ_cuda,
          tauXYav_cuda, tauXZav_cuda, tauYZav_cuda,
          J2_cuda, J2XY_cuda, J2XZ_cuda, J2YZ_cuda,
          nX, nY, nZ);
        gpuErrchk(cudaDeviceSynchronize());
      }
      ComputeDisp<<<grid, block>>>(Ux_cuda, Uy_cuda, Uz_cuda,
        Vx_cuda, Vy_cuda, Vz_cuda, P_cuda,
        tauXX_cuda, tauYY_cuda, tauZZ_cuda,
        tauXY_cuda, tauXZ_cuda, tauYZ_cuda,
        pa_cuda, nX, nY, nZ);
      gpuErrchk(cudaDeviceSynchronize());

      /*if (iter == 1000) {
      gpuErrchk(cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * sizeof(double), cudaMemcpyDeviceToHost));
      std::cout << "Ux1 = " << Ux_cpu[(3 * nY / 4) * (nX + 1) + 3 * nX / 4] << "\nUy1 = " << Uy_cpu[(3 * nY / 4) * nX + 3 * nX / 4] << "\n";
      }*/

      if ((iter + 1) % output_step == 0) {
        gpuErrchk(cudaMemcpy(Vx_cpu, Vx_cuda, (nX + 1) * nY * nZ * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(Vy_cpu, Vy_cuda, nX * (nY + 1) * nZ * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(Vz_cpu, Vz_cuda, nX * nY * (nZ + 1) * sizeof(double), cudaMemcpyDeviceToHost));

        error = (
          findMaxAbs(Vx_cpu, (nX + 1) * nY * nZ) / (dX * (nX - 1)) + 
          findMaxAbs(Vy_cpu, nX * (nY + 1) * nZ) / (dY * (nY - 1)) +
          findMaxAbs(Vz_cpu, nX * nY * (nZ + 1)) / (dZ * (nZ - 1))
        ) * dT /
          (std::max(std::abs(curEffStrain[0]), std::max(curEffStrain[1], curEffStrain[2])));
        //(std::abs(loadStepValue) * std::max(std::max(std::abs(loadType[0]), std::abs(loadType[1])), std::abs(loadType[2])));

        std::cout << "    Iteration " << iter + 1 << ": Error is " << error << std::endl;
        log_file << "    Iteration " << iter + 1 << ": Error is " << error << std::endl;

        if (error < EITER) {
          std::cout << "Number of iterations is " << iter + 1 << "\n\n";
          log_file << "Number of iterations is " << iter + 1 << "\n\n";
          break;
        }
        else if (iter >= NITER - 1) {
          std::cout << "WARNING: Maximum number of iterations reached!\nError is " << error << "\n\n";
          log_file << "WARNING: Maximum number of iterations reached!\nError is " << error << "\n\n";
        }
      }
    } // for(iter), iteration loop

    /* AVERAGING */
    gpuErrchk(cudaMemcpy(P_cpu, P_cuda, nX * nY * nZ * sizeof(double), cudaMemcpyDeviceToHost));
    /*gpuErrchk(cudaMemcpy(tauXX_cpu, tauXX_cuda, nX * nY * nZ * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(tauYY_cpu, tauYY_cuda, nX * nY * nZ * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(tauZZ_cpu, tauZZ_cuda, nX * nY * nZ * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(tauXY_cpu, tauXY_cuda, (nX - 1) * (nY - 1) * nZ * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(tauXZ_cpu, tauXZ_cuda, (nX - 1) * nY * (nZ - 1) * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(tauYZ_cpu, tauYZ_cuda, nX * (nY - 1) * (nZ - 1) * sizeof(double), cudaMemcpyDeviceToHost));
    //gpuErrchk(cudaMemcpy(tauXYav_cpu, tauXYav_cuda, nX * nY * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(J2_cpu, J2_cuda, nX * nY * nZ * sizeof(double), cudaMemcpyDeviceToHost));*/
    gpuErrchk(cudaMemcpy(Ux_cpu, Ux_cuda, (nX + 1) * nY * nZ * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(Uy_cpu, Uy_cuda, nX * (nY + 1) * nZ * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(Uz_cpu, Uz_cuda, nX * nY * (nZ + 1) * sizeof(double), cudaMemcpyDeviceToHost));

    PeffNonper[step][it] = getPeffNonper();
    PeffPer[step][it] = getPeffPer();

    std::cout << "    P / Y = " << PeffNonper[step][it] / Y << '\n';
    log_file << "    P / Y = " << PeffNonper[step][it] / Y << '\n';
    if (nPores > 2) {
      std::cout << "    Pper / Y = " << PeffPer[step][it] / Y << '\n';
      log_file << "    Pper / Y = " << PeffPer[step][it] / Y << '\n';
    }

    zeroingPoresDisp();
    /*SaveSlice(Ux_cpu, Ux_cuda, nX + 1, nY, nZ, nZ / 2, "data/UxcXY_" + std::to_string(8 * NGRID) + "_.dat");
    SaveSlice(Uy_cpu, Uy_cuda, nX, nY + 1, nZ, nZ / 2, "data/UycXY_" + std::to_string(8 * NGRID) + "_.dat");
    SaveSlice(Uz_cpu, Uz_cuda, nX, nY, nZ + 1, nZ / 2, "data/UzcXY_" + std::to_string(8 * NGRID) + "_.dat");*/
    calcPoreVolume();

    const double Phi0 = 3.1415926 * 4.0 * pow(rad * nPores, 3.0) / (3.0 * lX * lY * lZ);
    //std::cout << "    Phi0 = " << Phi0 << '\n';
    log_file << "    Phi0 = " << Phi0 << '\n';
    const double PhiNonper = 3.1415926 * 4.0 * poreVolume43Pi / (3.0 * lX * lY * lZ);
    //std::cout << "    PhiNonper = " << PhiNonper << '\n';
    log_file << "    PhiNonper = " << PhiNonper << '\n';
    const double PhiPer = nPores > 2 ? 
      3.1415926 * 4.0 * internalPoreVolume43Pi / (3.0 * lX * lY * lZ * pow(static_cast<double>(nPores - 2) / nPores, 3.0)) :
      0.0;
    //std::cout << "    PhiPer = " << PhiPer << '\n';
    log_file << "    PhiPer = " << PhiPer << '\n';


    dPhiNonper[step][it] = std::abs(PhiNonper - Phi0);
    std::cout << "    dPhiNonper = " << dPhiNonper[step][it] << '\n';
    log_file << "    dPhiNonper = " << dPhiNonper[step][it] << '\n';
    dPhiPer[step][it] = std::abs(PhiPer - Phi0);
    std::cout << "    dPhiPer = " << dPhiPer[step][it] << '\n';
    log_file << "    dPhiPer = " << dPhiPer[step][it] << '\n';
  } // for(it), action loop
}

void EffPlast3D::ReadParams(const std::string& filename) {
  std::ifstream pa_fil(filename, std::ios_base::binary);
  if (!pa_fil.is_open()) {
    throw std::runtime_error("ERROR:  Cannot open file " + filename + "!\n");
  }
  pa_fil.read((char*)pa_cpu, sizeof(double) * NPARS);
  gpuErrchk(cudaMemcpy(pa_cuda, pa_cpu, NPARS * sizeof(double), cudaMemcpyHostToDevice));
}
void EffPlast3D::SetMaterials() {
  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY; j++) {
      for (int k = 0; k < nZ; k++) {
        K_cpu[k * nX * nY + j * nX + i] = K0;
        G_cpu[k * nX * nY + j * nX + i] = G0;
        const double x = -0.5 * dX * (nX - 1) + dX * i;
        const double y = -0.5 * dY * (nY - 1) + dY * j;
        const double z = -0.5 * dZ * (nZ - 1) + dZ * k;
        const double Lx = dX * (nX - 1);
        const double Ly = dY * (nY - 1);
        const double Lz = dZ * (nZ - 1);
        for (int a = 0; a < nPores; a++) {
          for (int b = 0; b < nPores; b++) {
            for (int c = 0; c < nPores; c++) {
              if (sqrt(
                (x - 0.5 * Lx * (1.0 - 1.0 / nPores) + (Lx / nPores) * a) * (x - 0.5 * Lx * (1.0 - 1.0 / nPores) + (Lx / nPores) * a) +
                (y - 0.5 * Ly * (1.0 - 1.0 / nPores) + (Ly / nPores) * b) * (y - 0.5 * Ly * (1.0 - 1.0 / nPores) + (Ly / nPores) * b) +
                (z - 0.5 * Lz * (1.0 - 1.0 / nPores) + (Lz / nPores) * c) * (z - 0.5 * Lz * (1.0 - 1.0 / nPores) + (Lz / nPores) * c)
              ) < rad) {
                K_cpu[k * nX * nY + j * nX + i] = 0.01 * K0;
                G_cpu[k * nX * nY + j * nX + i] = 0.01 * G0;
                //empty_spaces.emplace(i, j);
              }
            } // for(c)
          } // for(b)
        } // for(a)
      } // for(k)
    } // for(j)
  } // for(i)
  
  gpuErrchk(cudaMemcpy(K_cuda, K_cpu, nX * nY * nZ * sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(G_cuda, G_cpu, nX * nY * nZ * sizeof(double), cudaMemcpyHostToDevice));
}
void EffPlast3D::SetInitPressure(const double coh) {
  const double P0 = 0.0; //1.0 * coh;

  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY; j++) {
      for (int k = 0; k < nZ; k++) {
        P0_cpu[j * nX + i] = 0.0;
        if (sqrt(
          (-0.5 * dX * (nX - 1) + dX * i) * (-0.5 * dX * (nX - 1) + dX * i) + 
          (-0.5 * dY * (nY - 1) + dY * j) * (-0.5 * dY * (nY - 1) + dY * j) +
          (-0.5 * dZ * (nZ - 1) + dZ * k) * (-0.5 * dZ * (nZ - 1) + dZ * k)
        ) < rad) {
          P0_cpu[k * nX * nY + j * nX + i] = P0;
        }
      } // for(k)
    } // for(j)
  } // for(i)

  gpuErrchk(cudaMemcpy(P0_cuda, P0_cpu, nX * nY * nZ * sizeof(double), cudaMemcpyHostToDevice));
}

void EffPlast3D::SetTensorZero(double** A_cpu, double** A_cuda, const int m, const int n, const int o) {
  *A_cpu = new double[m * n * o];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < o; k++) {
        (*A_cpu)[k * m * n + j * m + i] = 0.0;
      }
    }
  }
  gpuErrchk(cudaMalloc(A_cuda, m * n * o * sizeof(double)));
  gpuErrchk(cudaMemcpy(*A_cuda, *A_cpu, m * n * o * sizeof(double), cudaMemcpyHostToDevice));
}
void EffPlast3D::SaveSlice(double* const A_cpu, const double* const A_cuda, const int m, const int n, const int o, const int k, const std::string& filename) {
  gpuErrchk(cudaMemcpy(A_cpu, A_cuda, m * n * o * sizeof(double), cudaMemcpyDeviceToHost));
  std::ofstream A_filw(filename, std::ios_base::binary);
  A_filw.write((char*)A_cpu + sizeof(double) * m * n * k, sizeof(double) * m * n);
}

double EffPlast3D::findMaxAbs(const double* const arr, const int size) {
  double max_el = 0.0;
  for (int i = 0; i < size; i++) {
    if (std::abs(arr[i]) > std::abs(max_el)) {
      max_el = std::abs(arr[i]);
    }
  }
  return max_el;
}
double EffPlast3D::findMaxAbs(const std::vector<double>& vec) {
  double max_el = 0.0;
  for (auto i : vec) {
    if (std::abs(i) > std::abs(max_el)) {
      max_el = i;
    }
  }
  return max_el;
}
void EffPlast3D::zeroingPoresDisp() {
  // set zero Ux in the pores
  for (int i = 0; i < nX + 1; i++) {
    for (int j = 0; j < nY; j++) {
      for (int k = 0; k < nZ; k++) {
        const double x = -0.5 * dX * nX + dX * i;
        const double y = -0.5 * dY * (nY - 1) + dY * j;
        const double z = -0.5 * dZ * (nZ - 1) + dZ * k;
        for (int a = 0; a < nPores; a++) {
          for (int b = 0; b < nPores; b++) {
            for (int c = 0; c < nPores; c++) {
              if (sqrt(
                (x - 0.5 * lX * (1.0 - 1.0 / nPores) + (lX / nPores) * a) * (x - 0.5 * lX * (1.0 - 1.0 / nPores) + (lX / nPores) * a) +
                (y - 0.5 * lY * (1.0 - 1.0 / nPores) + (lY / nPores) * b) * (y - 0.5 * lY * (1.0 - 1.0 / nPores) + (lY / nPores) * b) +
                (z - 0.5 * lZ * (1.0 - 1.0 / nPores) + (lZ / nPores) * c) * (z - 0.5 * lZ * (1.0 - 1.0 / nPores) + (lZ / nPores) * c)
              ) < rad)
              {
                Ux_cpu[k * (nX + 1) * nY + j * (nX + 1) + i] = 0.0;
              }
            } // for(c)
          } // for(b)
        } // for(a)
      } // for(k)
    } // for(j)
  }
  // set zero Uy in the pores
  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY + 1; j++) {
      for (int k = 0; k < nZ; k++) {
        const double x = -0.5 * dX * (nX - 1) + dX * i;
        const double y = -0.5 * dY * nY + dY * j;
        const double z = -0.5 * dZ * (nZ - 1) + dZ * k;
        for (int a = 0; a < nPores; a++) {
          for (int b = 0; b < nPores; b++) {
            for (int c = 0; c < nPores; c++) {
              if (sqrt(
                (x - 0.5 * lX * (1.0 - 1.0 / nPores) + (lX / nPores) * a) * (x - 0.5 * lX * (1.0 - 1.0 / nPores) + (lX / nPores) * a) +
                (y - 0.5 * lY * (1.0 - 1.0 / nPores) + (lY / nPores) * b) * (y - 0.5 * lY * (1.0 - 1.0 / nPores) + (lY / nPores) * b) +
                (z - 0.5 * lZ * (1.0 - 1.0 / nPores) + (lZ / nPores) * c) * (z - 0.5 * lZ * (1.0 - 1.0 / nPores) + (lZ / nPores) * c)
              ) < rad)
              {
                Uy_cpu[k * nX * (nY + 1) + j * nX + i] = 0.0;
              }
            } // for(c)
          } // for(b)
        } // for(a)
      } // for(k)
    } // for(j)
  }
  // set zero Uz in the pores
  for (int i = 0; i < nX; i++) {
    for (int j = 0; j < nY; j++) {
      for (int k = 0; k < nZ + 1; k++) {
        const double x = -0.5 * dX * (nX - 1) + dX * i;
        const double y = -0.5 * dY * (nY - 1) + dY * j;
        const double z = -0.5 * dZ * nZ + dZ * k;
        for (int a = 0; a < nPores; a++) {
          for (int b = 0; b < nPores; b++) {
            for (int c = 0; c < nPores; c++) {
              if (sqrt(
                (x - 0.5 * lX * (1.0 - 1.0 / nPores) + (lX / nPores) * a) * (x - 0.5 * lX * (1.0 - 1.0 / nPores) + (lX / nPores) * a) +
                (y - 0.5 * lY * (1.0 - 1.0 / nPores) + (lY / nPores) * b) * (y - 0.5 * lY * (1.0 - 1.0 / nPores) + (lY / nPores) * b) +
                (z - 0.5 * lZ * (1.0 - 1.0 / nPores) + (lZ / nPores) * c) * (z - 0.5 * lZ * (1.0 - 1.0 / nPores) + (lZ / nPores) * c)
              ) < rad)
              {
                Uz_cpu[k * nX * nY + j * nX + i] = 0.0;
              }
            } // for(c)
          } // for(b)
        } // for(a)
      } // for(k)
    } // for(j)
  }
}
void EffPlast3D::calcPoreVolume() const {
  poreVolume43Pi = 0.0;
  internalPoreVolume43Pi = 0.0;
  for (int a = 0; a < nPores; a++) {
    for (int b = 0; b < nPores; b++) {
      for (int c = 0; c < nPores; c++) {
        const double cxdX = 0.5 * (nX - 1) * (1.0 - 1.0 / nPores) - (static_cast<double>(nX - 1) / nPores) * a; // cx / dX
        const double cydY = 0.5 * (nY - 1) * (1.0 - 1.0 / nPores) - (static_cast<double>(nY - 1) / nPores) * b; // cy / dY
        const double czdZ = 0.5 * (nZ - 1) * (1.0 - 1.0 / nPores) - (static_cast<double>(nZ - 1) / nPores) * c; // cz / dZ

        const size_t cxIdx = static_cast<size_t>(cxdX + 0.5 * (nX - 1));
        const size_t cyIdx = static_cast<size_t>(cydY + 0.5 * (nY - 1));
        const size_t czIdx = static_cast<size_t>(czdZ + 0.5 * (nZ - 1));

        // horizontal displacements
        // left point of a pore        
        size_t rxIdx = static_cast<size_t>(cxdX - rad / dX + 0.5 * nX);
        std::vector<double> dispXleft(5);
        //std::cout << "dispXleft:\n";
        if (rxIdx < 1) {
          throw std::out_of_range("Error in calcPoreVolume! Grid is too small or pores are too big!\n");
        }
        for (int i = 0; i < 5; i++) {
          dispXleft[i] = Ux_cpu[czIdx * (nX + 1) * nY + cyIdx * (nX + 1) + rxIdx - 1 + i];
          //std::cout << "j = " << cyIdx << " i = " << rxIdx - 1 + i << "\n";
          //std::cout << dispXleft[i] << "\n";
        }
        // right point of a pore
        rxIdx = static_cast<size_t>(cxdX + rad / dX + 0.5 * nX);
        std::vector<double> dispXright(5);
        //std::cout << "dispXright:\n";
        if (rxIdx > nX - 2) {
          throw std::out_of_range("Error in calcPoreVolume! Grid is too small or pores are too big!\n");
        }
        for (int i = 0; i < 5; i++) {
          dispXright[i] = Ux_cpu[czIdx * (nX + 1) * nY + cyIdx * (nX + 1) + rxIdx - 2 + i];
          //std::cout << dispXright[i] << "\n";
        }

        // depth displacements
        // near point of a pore
        size_t ryIdx = static_cast<size_t>(cydY - rad / dY + 0.5 * nY);
        std::vector<double> dispYnear(5);
        //std::cout << "dispYnear:\n";
        if (ryIdx < 1) {
          throw std::out_of_range("Error in calcPoreVolume! Grid is too small or pores are too big!\n");
        }
        for (int j = 0; j < 5; j++) {
          dispYnear[j] = Uy_cpu[czIdx * nX * (nY + 1) + (ryIdx - 1 + j) * nX + cxIdx];
          //std::cout << dispYnear[j] << "\n";
        }
        // far point of a hole
        ryIdx = static_cast<size_t>(cydY + rad / dY + 0.5 * nY);
        std::vector<double> dispYfar(5);
        //std::cout << "dispYfar:\n";
        if (ryIdx > nY - 2) {
          throw std::out_of_range("Error in calcPoreVolume! Grid is too small or pores are too big!\n");
        }
        for (int j = 0; j < 5; j++) {
          dispYfar[j] = Uy_cpu[czIdx * nX * (nY + 1) + (ryIdx - 2 + j) * nX + cxIdx];
          //std::cout << dispYfar[j] << "\n";
        }

        // vertical displacements
        // bottom point of a hole
        size_t rzIdx = static_cast<size_t>(czdZ - rad / dZ + 0.5 * nZ);
        std::vector<double> dispZbottom(5);
        //std::cout << "dispZbottom:\n";
        if (rzIdx < 1) {
          throw std::out_of_range("Error in calcPoreVolume! Grid is too small or pores are too big!\n");
        }
        for (int k = 0; k < 5; k++) {
          dispZbottom[k] = Uz_cpu[(rzIdx - 1 + k) * nX * nY + cyIdx * nX + cxIdx];
          //std::cout << dispZbottom[k] << "\n";
        }
        // top point of a hole
        rzIdx = static_cast<size_t>(czdZ + rad / dZ + 0.5 * nZ);
        std::vector<double> dispZtop(5);
        //std::cout << "dispYtop\n";
        if (rzIdx > nZ - 2) {
          throw std::out_of_range("Error in calcPoreVolume! Grid is too small or pores are too big!\n");
        }
        for (int k = 0; k < 5; k++) {
          dispZtop[k] = Uz_cpu[(rzIdx - 2 + k) * nX * nY + cyIdx * nX + cxIdx];
          //std::cout << dispZtop[k] << "\n";
        }

        //std::cout << "dRxLeft = " << FindMaxAbs(dispXleft) << ", dRxRight = " << FindMaxAbs(dispXright) << "\n";
        const double dRx = -0.5 * (findMaxAbs(dispXleft) - findMaxAbs(dispXright));
        const double dRy = -0.5 * (findMaxAbs(dispYnear) - findMaxAbs(dispYfar));
        const double dRz = -0.5 * (findMaxAbs(dispZbottom) - findMaxAbs(dispZtop));
        //std::cout << "dRx = " << dRx << ", dRy = " << dRy << "\n";

        poreVolume43Pi += (rad + dRx) * (rad + dRy) * (rad + dRz);
        //std::cout << poreVolume43Pi << "\n";
        if (a > 0 && b > 0 && c > 0 && a < nPores - 1 && b < nPores - 1 && c < nPores - 1) {
          internalPoreVolume43Pi += (rad + dRx) * (rad + dRy) * (rad + dRz);
        }
      } // for(c)
    } // for(b)
  } // for(a)
}

/* AVERAGING */
double EffPlast3D::getPeffNonper() const {
  if (nX <= 2 && nY <= 2 && nZ <= 2) {
    throw std::runtime_error("Error in getPeffNonper! The grid is too small!\n");
  }
  double PeffX{0.0}, PeffY{0.0}, PeffZ{0.0};
  for (int j = 1; j < nY - 1; j++) {
    for (int k = 1; k < nZ - 1; k++) {
      PeffX += P_cpu[k * nX * nY + j * nX + 0];
      PeffX += P_cpu[k * nX * nY + j * nX + nX - 1];
    }
  }
  PeffX /= 2.0 * (nY - 2) * (nZ - 2);
  for (int i = 1; i < nX - 1; i++) {
    for (int k = 1; k < nZ - 1; k++) {
      PeffY += P_cpu[k * nX * nY + 0 * nX + i];
      PeffY += P_cpu[k * nX * nY + (nY - 1) * nX + i];
    }
  }
  PeffY /= 2.0 * (nX - 2) * (nZ - 2);
  for (int i = 1; i < nX - 1; i++) {
    for (int j = 1; j < nY - 1; j++) {
      PeffZ += P_cpu[0 * nX * nY + j * nX + i];
      PeffZ += P_cpu[(nZ - 1) * nX * nY + j * nX + i];
    }
  }
  PeffZ /= 2.0 * (nX - 2) * (nY - 2);
  return (PeffX + PeffY + PeffZ) / 3.0;
}
double EffPlast3D::getPeffPer() const {
  if (nPores <= 2) {
    return 0.0;
  }
  if (nX < nPores || nY < nPores || nZ < nPores) {
    throw std::runtime_error("Error in getPeffPer! The grid is too small!\n");
  }
  double PeffX{0.0}, PeffY{0.0}, PeffZ{0.0};
  for (int j = nY / nPores; j < nY * (nPores - 1) / nPores; j++) {
    for (int k = nZ / nPores; k < nZ * (nPores - 1) / nPores; k++) {
      PeffX += P_cpu[k * nX * nY + j * nX + nX / nPores];
      PeffX += P_cpu[k * nX * nY + j * nX + nX  * (nPores - 1) / nPores];
    }
  }
  PeffX /= 2.0 * (nY - 2) * (nZ - 2) * (nPores - 2) * (nPores - 2) / nPores / nPores;
  for (int i = nX / nPores; i < nX * (nPores - 1) / nPores; i++) {
    for (int k = nZ / nPores; k < nZ * (nPores - 1) / nPores; k++) {
      PeffY += P_cpu[k * nX * nY + nY / nPores * nX + i];
      PeffY += P_cpu[k * nX * nY + nY * (nPores - 1) / nPores * nX + i];
    }
  }
  PeffY /= 2.0 * (nX - 2) * (nZ - 2) * (nPores - 2) * (nPores - 2) / nPores / nPores;
  for (int i = nX / nPores; i < nX * (nPores - 1) / nPores; i++) {
    for (int j = nY / nPores; j < nY * (nPores - 1) / nPores; j++) {
      PeffZ += P_cpu[nZ / nPores * nX * nY + j * nX + i];
      PeffZ += P_cpu[nZ * (nPores - 1) / nPores * nX * nY + j * nX + i];
    }
  }
  PeffZ /= 2.0 * (nX - 2) * (nY - 2) * (nPores - 2) * (nPores - 2) / nPores / nPores;
  return (PeffX + PeffY + PeffZ) / 3.0;
}

/* CONSOLE AND LOG FILE OUTPUT */
void EffPlast3D::printStepInfo(const size_t step) {
  std::cout << "\nLOAD STEP " << step + 1 << " FROM " << NL << ": ";
  log_file << "\nLOAD STEP " << step + 1 << " FROM " << NL << ": ";
  switch (step) {
  case 0:
    std::cout << "PRELOADING\n";
    log_file << "PRELOADING\n";
    break;
  case 1:
    std::cout << "SMALL HYDROSTATIC INCREMENT\n";
    log_file << "SMALL HYDROSTATIC INCREMENT\n";
    break;
  case 2:
    std::cout << "SMALL DEVIATORIC INCREMENT\n";
    log_file << "SMALL DEVIATORIC INCREMENT\n";
    break;
  default:
    throw std::invalid_argument("ERROR:  Wrong step index!\n");
  }
  std::cout << "Porosity is " << porosity * 100 << "%\n";
  log_file << "Porosity is " << porosity * 100 << "%\n";
  std::cout << "Grid resolution is " << nX << "x" << nY << "x" << nZ << "\n\n";
  log_file << "Grid resolution is " << nX << "x" << nY << "x" << nZ << "\n\n";
}
void EffPlast3D::printCalculationType() {
  switch (NL) {
  case 1:
    std::cout << "\nPURE ELASTIC CALCULATION\nESTIMATION OF THE EFFECTIVE BULK MODULI\n";
    log_file << "\nPURE ELASTIC CALCULATION\nESTIMATION OF THE EFFECTIVE BULK MODULI\n";
    break;
  case 2:
    std::cout << "\nELASTOPLASTIC CALCULATION\nESTIMATION OF THE EFFECTIVE BULK MODULI\n";
    log_file << "\nELASTOPLASTIC CALCULATION\nESTIMATION OF THE EFFECTIVE BULK MODULI\n";
    break;
  /*case 3:
    std::cout << "\nELASTOPLASTIC CALCULATION\nESTIMATION OF THE EFFECTIVE BULK MODULI AND THE EFFECTIVE SHEAR MODULUS\n";
    log_file << "\nELASTOPLASTIC CALCULATION\nESTIMATION OF THE EFFECTIVE BULK MODULI AND THE EFFECTIVE SHEAR MODULUS\n";
    break;*/
  default:
    throw std::invalid_argument("ERROR:  Wrong number of loads!\n");
  }
}
void EffPlast3D::printDuration(int elapsed_sec) {
  if (elapsed_sec < 60) {
    std::cout << "\nCalculation time is " << elapsed_sec << " sec\n";
    log_file << "\nCalculation time is " << elapsed_sec << " sec\n\n\n";
  }
  else {
    int elapsed_min = elapsed_sec / 60;
    elapsed_sec = elapsed_sec % 60;
    if (elapsed_min < 60) {
      std::cout << "\nCalculation time is " << elapsed_min << " min " << elapsed_sec << " sec\n";
      log_file << "\nCalculation time is " << elapsed_min << " min " << elapsed_sec << " sec\n\n\n";
    }
    else {
      int elapsed_hour = elapsed_min / 60;
      elapsed_min = elapsed_min % 60;
      if (elapsed_hour < 24) {
        std::cout << "\nCalculation time is " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n";
        log_file << "\nCalculation time is " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n\n\n";
      }
      else {
        const int elapsed_day = elapsed_hour / 24;
        elapsed_hour = elapsed_hour % 24;
        if (elapsed_day < 7) {
          std::cout << "\nCalculation time is " << elapsed_day << " days " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n";
          log_file << "\nCalculation time is " << elapsed_day << " days " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n\n\n";
        }
        else {
          std::cout << "\nCalculation time is " << elapsed_day / 7 << " weeks " << elapsed_day % 7 << " days " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n";
          log_file << "\nCalculation time is " << elapsed_day / 7 << " weeks " << elapsed_day % 7 << " days " << elapsed_hour << " hours " << elapsed_min << " min " << elapsed_sec << " sec\n\n\n";
        }
      }
    }
  }
}

/* FINAL EFFECTIVE MODULI CALCULATION */
void EffPlast3D::calcBulkModuli_PureElast() {
  eff_moduli_num_nonper.Kphi = getKphiNonper_PureElast();
  std::cout << "    ==============\n\n" << "KphiNonper = " << eff_moduli_num_nonper.Kphi << std::endl;
  log_file << "    ==============\n\n" << "KphiNonper = " << eff_moduli_num_nonper.Kphi << std::endl;

  eff_moduli_num_per.Kphi = getKphiPer_PureElast();
  std::cout << "KphiPer = " << eff_moduli_num_per.Kphi << std::endl;
  log_file << "KphiPer = " << eff_moduli_num_per.Kphi << std::endl;
}
void EffPlast3D::calcBulkModuli_ElastPlast() {
  eff_moduli_num_nonper.Kphi = getKphiNonper_ElastPlast();
  std::cout << "    ==============\n\n" << "KphiNonper = " << eff_moduli_num_nonper.Kphi << std::endl;
  log_file << "    ==============\n\n" << "KphiNonper = " << eff_moduli_num_nonper.Kphi << std::endl;

  eff_moduli_num_per.Kphi = getKphiPer_ElastPlast();
  std::cout << "KphiPer = " << eff_moduli_num_per.Kphi << std::endl;
  log_file << "KphiPer = " << eff_moduli_num_per.Kphi << std::endl;
}
// bulk moduli in the pure elastic case
double EffPlast3D::getKphiNonper_PureElast() {
  const double Pinc = PeffNonper[0][nTimeSteps_ - 1];
  const double phiInc = dPhiNonper[0][nTimeSteps_ - 1];
  return Pinc / phiInc;
}
double EffPlast3D::getKphiPer_PureElast() {
  const double Pinc = PeffPer[0][nTimeSteps_ - 1];
  const double phiInc = dPhiPer[0][nTimeSteps_ - 1];
  return Pinc / phiInc;
}
double EffPlast3D::getKphiNonper_ElastPlast() {
  const double Pinc = PeffNonper[1][0] - PeffNonper[0][nTimeSteps_ - 1];
  const double phiInc = dPhiNonper[1][0] - dPhiNonper[0][nTimeSteps_ - 1];
  return Pinc / phiInc;
}
double EffPlast3D::getKphiPer_ElastPlast() {
  const double Pinc = PeffPer[1][0] - PeffPer[0][nTimeSteps_ - 1];
  const double phiInc = dPhiPer[1][0] - dPhiPer[0][nTimeSteps_ - 1];
  return Pinc / phiInc;
}

EffPlast3D::EffPlast3D() {
  block.x = 8;
  block.y = 8;
  block.z = 8;
  grid.x = NGRID;
  grid.y = NGRID;
  grid.z = NGRID;

  nX = block.x * grid.x;
  nY = block.y * grid.y;
  nZ = block.z * grid.z;

  gpuErrchk(cudaSetDevice(DEVICE_IDX));
  //gpuErrchk(cudaDeviceReset());
  //gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

  /* PARAMETERS */
  pa_cpu = new double[NPARS];
  gpuErrchk(cudaMalloc(&pa_cuda, NPARS * sizeof(double)));
  ReadParams("data/pa.dat");

  dX = pa_cpu[0];
  dY = pa_cpu[1];
  dZ = pa_cpu[2];
  dT = pa_cpu[3];
  K0 = pa_cpu[4];
  G0 = pa_cpu[5];
  E0 = 9.0 * K0 * G0 / (3.0 * K0 + G0);
  nu0 = (1.5 * K0 - G0) / (3.0 * K0 + G0);
  //std::cout << "E = " << E0 << ", nu = " << nu0 << "\n";
  rad = pa_cpu[11];
  Y = pa_cpu[10] / sqrt(3.0);
  nPores = pa_cpu[12];

  cudaMemcpyToSymbol(coh_cuda, &pa_cpu[10], sizeof(double));

  /* SPACE ARRAYS */
  // materials
  K_cpu = new double[nX * nY * nZ];
  G_cpu = new double[nX * nY * nZ];
  gpuErrchk(cudaMalloc(&K_cuda, nX * nY * nZ * sizeof(double)));
  gpuErrchk(cudaMalloc(&G_cuda, nX * nY * nZ * sizeof(double)));
  SetMaterials();

  // stress
  P0_cpu = new double[nX * nY * nZ];
  gpuErrchk(cudaMalloc(&P0_cuda, nX * nY * nZ * sizeof(double)));
  SetInitPressure(pa_cpu[10]);

  SetTensorZero(&P_cpu, &P_cuda, nX, nY, nZ);
  SetTensorZero(&tauXX_cpu, &tauXX_cuda, nX, nY, nZ);
  SetTensorZero(&tauYY_cpu, &tauYY_cuda, nX, nY, nZ);
  SetTensorZero(&tauZZ_cpu, &tauZZ_cuda, nX, nY, nZ);
  SetTensorZero(&tauXY_cpu, &tauXY_cuda, nX - 1, nY - 1, nZ);
  SetTensorZero(&tauXZ_cpu, &tauXZ_cuda, nX - 1, nY, nZ - 1);
  SetTensorZero(&tauYZ_cpu, &tauYZ_cuda, nX, nY - 1, nZ - 1);
  if (NL > 1) {
    SetTensorZero(&tauXYav_cpu, &tauXYav_cuda, nX, nY, nZ);
    SetTensorZero(&tauXZav_cpu, &tauXZav_cuda, nX, nY, nZ);
    SetTensorZero(&tauYZav_cpu, &tauYZav_cuda, nX, nY, nZ);

    // plasticity
    SetTensorZero(&J2_cpu, &J2_cuda, nX, nY, nZ);
    SetTensorZero(&J2XY_cpu, &J2XY_cuda, nX - 1, nY - 1, nZ);
    SetTensorZero(&J2XZ_cpu, &J2XZ_cuda, nX - 1, nY, nZ - 1);
    SetTensorZero(&J2YZ_cpu, &J2YZ_cuda, nX, nY - 1, nZ - 1);
  }

  // displacement
  SetTensorZero(&Ux_cpu, &Ux_cuda, nX + 1, nY, nZ);
  SetTensorZero(&Uy_cpu, &Uy_cuda, nX, nY + 1, nZ);
  SetTensorZero(&Uz_cpu, &Uz_cuda, nX, nY, nZ + 1);

  // velocity
  SetTensorZero(&Vx_cpu, &Vx_cuda, nX + 1, nY, nZ);
  SetTensorZero(&Vy_cpu, &Vy_cuda, nX, nY + 1, nZ);
  SetTensorZero(&Vz_cpu, &Vz_cuda, nX, nY, nZ + 1);

  /* UTILITIES */
  log_file.open("EffPlast3D.log", std::ios_base::app);
  output_step = 1000;
  lX = (nX - 1) * dX;
  lY = (nY - 1) * dY;
  lZ = (nZ - 1) * dZ;
  porosity = (4.0 / 3.0) * 3.1415926 * pow(rad * nPores, 3.0) / (lX * lY * lZ);
}
EffPlast3D::~EffPlast3D() {
  // parameters
  delete[] pa_cpu;
  gpuErrchk(cudaFree(pa_cuda));

  // materials
  delete[] K_cpu;
  delete[] G_cpu;
  gpuErrchk(cudaFree(K_cuda));
  gpuErrchk(cudaFree(G_cuda));

  // stress
  delete[] P0_cpu;
  delete[] P_cpu;
  delete[] tauXX_cpu;
  delete[] tauYY_cpu;
  delete[] tauZZ_cpu;
  delete[] tauXY_cpu;
  delete[] tauXZ_cpu;
  delete[] tauYZ_cpu;
  //delete[] tauXYav_cpu;
  gpuErrchk(cudaFree(P0_cuda));
  gpuErrchk(cudaFree(P_cuda));
  gpuErrchk(cudaFree(tauXX_cuda));
  gpuErrchk(cudaFree(tauYY_cuda));
  gpuErrchk(cudaFree(tauZZ_cuda));
  gpuErrchk(cudaFree(tauXY_cuda));
  gpuErrchk(cudaFree(tauXZ_cuda));
  gpuErrchk(cudaFree(tauYZ_cuda));
  //gpuErrchk(cudaFree(tauXYav_cuda));

  // plasticity
  /*delete[] J2_cpu;
  delete[] J2XY_cpu;
  gpuErrchk(cudaFree(J2_cuda));
  gpuErrchk(cudaFree(J2XY_cuda));*/

  // displacement
  delete[] Ux_cpu;
  delete[] Uy_cpu;
  delete[] Uz_cpu;
  gpuErrchk(cudaFree(Ux_cuda));
  gpuErrchk(cudaFree(Uy_cuda));
  gpuErrchk(cudaFree(Uz_cuda));

  // velocity
  delete[] Vx_cpu;
  delete[] Vy_cpu;
  delete[] Vz_cpu;
  gpuErrchk(cudaFree(Vx_cuda));
  gpuErrchk(cudaFree(Vy_cuda));
  gpuErrchk(cudaFree(Vz_cuda));

  // log
  log_file.close();
}