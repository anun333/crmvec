/* copied verbatim from CORE-MATH src/binary64/erfc/erfc.c (MIT; see erfc.c here): its exp_1
   polynomial Q_1 and the asymptotic polynomials T; its T1/T2 are pow's (POW_T1/POW_T2) */
static const double ERFC_Q1[5] = {0x1p0,                 /* degree 0 */
                             0x1p0,                 /* degree 1 */
                             0x1p-1,                /* degree 2 */
                             0x1.5555555995d37p-3,  /* degree 3 */
                             0x1.55555558489dcp-5   /* degree 4 */
};
static const double ERFC_T[6][13] = {
  {0x1.20dd750429b6dp-1, 0x1.1a1feb75a48a8p-57, -0x1.20dd750429b6cp-2, 0x1.b14c2f863e403p-2, -0x1.0ecf9db3af35dp+0, 0x1.d9eb53ca6eeedp+1, -0x1.0a945830d95c8p+4, 0x1.6e8a963e2f1f5p+6, -0x1.29b7ccc8f396fp+9, 0x1.15e716e83c27ep+12, -0x1.1cfdcfbcaf22ap+15, 0x1.1986cc7a7e8fep+18, -0x1.71f7540590a91p+20}, /* asympt0.sollya, [0x1.2ce37fb080c7dp-5,0x1.d5p-4], 2^-74.788, |p'/p| < 27.2 */
  {0x1.20dd750429ae7p-1, 0x1.63da89e801fd4p-55, -0x1.20dd750400795p-2, 0x1.b14c2f57c490cp-2, -0x1.0ecf95c8c9014p+0, 0x1.d9e981f2321efp+1, -0x1.0a81482de1506p+4, 0x1.6d662420a604bp+6, -0x1.233c96fff7772p+9, 0x1.f5d62018d3e37p+11, -0x1.9ae55e955445p+14, 0x1.052901e10d139p+17, -0x1.66465df1385fp+18}, /* asympt1.sollya, [0x1.d5p-4,0x1.59da6ca291ba6p-3], 2^-72.938, |p'/p| < 8.63 */
  {0x1.20dd75041e3fcp-1, -0x1.c9b491c4920fcp-56, -0x1.20dd74e5f1526p-2, 0x1.b14c1d35a40ep-2, -0x1.0ecdecd30e86bp+0, 0x1.d9b4e7f725263p+1, -0x1.0958b5ca8fb39p+4, 0x1.63e3179bf609cp+6, -0x1.06bbd1cd2d0fdp+9, 0x1.7b66eb6d1d2f2p+11, -0x1.ce5a4b1afab75p+13, 0x1.8b5c6ae6f773cp+15, -0x1.5475860326f86p+16}, /* asympt2.sollya, [0x1.59da6ca291ba6p-3,0x1.bcp-3], 2^-72.15, |p'/p| < 5.77 */
  {0x1.20dd75025cfe9p-1, 0x1.5a92eef32fb2p-58, -0x1.20dd71eb9d4e7p-2, 0x1.b14af4c25db28p-2, -0x1.0ebc78a22b3d8p+0, 0x1.d85287a0b3399p+1, -0x1.045f751e5ca1dp+4, 0x1.4a0d87ddea589p+6, -0x1.ac6a0981d1eeep+8, 0x1.f44822f567956p+10, -0x1.cba372de71349p+12, 0x1.1a4a19f550ca4p+14, -0x1.52a580455ed79p+14}, /* asympt3.sollya, [0x1.bcp-3,0x1.0cp-2], 2^-71.804, |p'/p| < 4.42 */
  {0x1.20dd74eb31d84p-1, -0x1.39c4054b7c09p-59, -0x1.20dd561af98c4p-2, 0x1.b1435165d9df1p-2, -0x1.0e6b60308e94p+0, 0x1.d3ce30c140882p+1, -0x1.f2083e404c299p+3, 0x1.20f113d89b42ap+6, -0x1.41433ebd89f19p+8, 0x1.2f35b6a3154f6p+10, -0x1.b020a4313cf3bp+11, 0x1.90f07e92da7eep+12, -0x1.6565e1d7665c3p+12}, /* asympt4.sollya, [0x1.0cp-2,0x1.38p-2], 2^-71.811, |p'/p| < 3.60 */
  {0x1.20dd744b3517bp-1, -0x1.f77ab25e01ab4p-57, -0x1.20dcc62ec4024p-2, 0x1.b125bfa4f66c1p-2, -0x1.0d80e6538197p+0, 0x1.ca11fbcfa65b2p+1, -0x1.cd9eaffb88315p+3, 0x1.e010db42e0da7p+5, -0x1.c5c85250ef6a3p+7, 0x1.5e118d9c1eeafp+9, -0x1.8d74be13d3d3p+10, 0x1.211b1b2b5ac83p+11, -0x1.900be759fc663p+10}, /* asympt5.sollya, [0x1.38p-2,0x1.63p-2], 2^-71.841, |p'/p| < 3.04 */
};
