from enum import IntEnum


class Conserved(IntEnum):
   RHO = 0
   MOMX = 1
   MOMY = 2
   MOMZ = 3
   ENERGY = 4
   BX = 5
   BY = 6
   BZ = 7


class Primitive(IntEnum):
   RHO = 0
   VX = 1
   VY = 2
   VZ = 3
   PRESS = 4
   BX = 5
   BY = 6
   BZ = 7
   LORENTZ = 8


class Speed(IntEnum):
   SOUND2 = 0
   ALFVEN_X2 = 1
   ALFVEN2 = 2
   FAST2 = 3


class Diagnostic(IntEnum):
   MASS = 0
   ENERGY = 1
   INTERNAL_ENERGY = 2
   MOMX = 3
   MOMY = 4
   MOMZ = 5
   MOM = 6
   GRADPX_L2 = 7
   GRADPY_L2 = 8
   LAPP_L2 = 9
   MAGNETIC_ENERGY = 10
   KINETIC_ENERGY = 11
   DIVB_L2 = 12


U = Conserved
P = Primitive
S = Speed
D = Diagnostic

NVAR_MHD = 8
NVAR_HYDRO = 5
PVAR_MHD = 8
PVAR_HYDRO = 5
PVAR_RMHD = 9

MOM_U = slice(U.MOMX, U.MOMZ + 1)
VEL_P = slice(P.VX, P.VZ + 1)
MAG_U = slice(U.BX, U.BZ + 1)
MAG_P = slice(P.BX, P.BZ + 1)

U_NAMES = ('rho', 'Mom_x', 'Mom_y', 'Mom_z', 'E', 'B_x', 'B_y', 'B_z')
P_NAMES = ('rho', 'v_x', 'v_y', 'v_z', 'P', 'B_x', 'B_y', 'B_z')
DIAGNOSTIC_NAMES = tuple(name.lower() for name in Diagnostic.__members__)
