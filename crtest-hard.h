/* inputs CORE-MATH's own files single out as hard (exceptions tables and
   the accurate-path database), used by crtest as extra test points */
static const double EXP_HARD[] = {
  0x1.fffffffffffffp-53, 0x1.ba07d73250de7p-14, 0x1.6a4d1af9cc989p-8, 0x1.5a75293a5dcdap-6,
  0x1.42ea46949b3c7p-5, 0x1.7c8bb0cf5d16p-5, 0x1.0948d39a41695p-3, 0x1.a065fefae814fp-3,
  0x1.f6e4c3ced7c72p-3, 0x1.1a0408712e00ap-2, 0x1.bcab27d05abdep-2, 0x1.005ae04256babp-1,
  0x1.273c188aa7b14p+2, 0x1.83d4bcdebb3f4p+2, 0x1.08f51434652c3p+4, 0x1.1d5c2daebe367p+4,
  0x1.c44ce0d716a1ap+4, 0x1.e07e71bfcf06fp+5, 0x1.f7216c4b435c9p+5, 0x1.54cd1fea7663ap+7,
  0x1.d6479eba7c971p+8, -0x1.664716b68a409p-14, -0x1.a2fefefd580dfp-13, -0x1.ce3f638d0c742p-12,
  -0x1.ceff32831e2c2p-12, -0x1.33accae78b371p-11, -0x1.d792b60084f92p-11, -0x1.7fb235d76cce7p-8,
  -0x1.1ff9b8e8b38bep-7, -0x1.54511e930898cp-7, -0x1.5c5ed0ec83666p-6, -0x1.8c56ff5326197p-6,
  -0x1.a4187f2ca71f9p-6, -0x1.a8f783d749a8fp-4, -0x1.bd44fdaed819fp-4, -0x1.daf693d64fadap-4,
  -0x1.290ea09e36479p-3, -0x1.8aeb636f3ce35p-3, -0x1.d3f3799439415p-3, -0x1.ea16274b0109bp-3,
  -0x1.22e24fa3d5cf9p-1, -0x1.85068c07fbbf6p-1, -0x1.bdc7955d1482cp-1, -0x1.2a9cad9998262p+0,
  -0x1.cc37ef7de7501p+0, -0x1.02393d5976769p+1, -0x1.65061daf79a78p+1, -0x1.e8bdbfcd9144ep+3,
  -0x1.8f80e06f3a04cp+4, -0x1.59f038076039cp+6, -0x1.981587ad4542fp+7,
};
static const double COS_HARD[] = {0x1.8000000000009p-23, 0x1.8000000000024p-22, 0x1.800000000009p-21, 0x1.20000000000f3p-20, 0x1.800000000024p-20};  /* 5, from cos.c */
static const double TAN_HARD[] = {0x1.dffffffffff1fp-22, 0x1.dfffffffffc7cp-21};  /* 2, from tan.c */
/* 20: the 10 distinct inputs as_atan_refine2 in atan.c lists as exceptions, both signs */
static const double ATAN_HARD[] = {0x1.0dc89a3b5501p-7, 0x1.e3fb41d2d226p-8, 0x1.7ba49f739829fp-1, 0x1.a933fe176b375p-3, 0x1.bb04a79820063p-8, 0x1.cd30a9499618bp-8, 0x1.f44aa37b8e66bp-7, 0x1.fd2ac95e57ef9p-8, 0x1.6419079bbf601p-6, 0x1.d768804487b07p-3, -0x1.0dc89a3b5501p-7, -0x1.e3fb41d2d226p-8, -0x1.7ba49f739829fp-1, -0x1.a933fe176b375p-3, -0x1.bb04a79820063p-8, -0x1.cd30a9499618bp-8, -0x1.f44aa37b8e66bp-7, -0x1.fd2ac95e57ef9p-8, -0x1.6419079bbf601p-6, -0x1.d768804487b07p-3};
