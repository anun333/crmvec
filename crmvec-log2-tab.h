/* from CORE-MATH src/binary64/log2/log2.c (MIT; see log2.c here): cr_log2's tables;
   B[i] as two int64 columns, LOG2_B0 = c0 << 33 and LOG2_B1 = c1 (signed) */
static const long long LOG2_B0[32] = {2585570312192LL, 61753039781888LL, 114959094644736LL, 162547332284416LL, 204826990346240LL, 242098716540928LL, 274611618971648LL, 302631985610752LL, 326383154757632LL, 346071284842496LL, 361885354426368LL, 374022932004864LL, 382647226335232LL, 387921446174720LL, 389974440542208LL, 388960828260352LL, 385009458348032LL, 378231999954944LL, 368757302099968LL, 356671264129024LL, 342094145126400LL, 325111844438016LL, 305818851344384LL, 284301065191424LL, 260627205455872LL, 234891761418240LL, 207146272686080LL, 177476638605312LL, 145934398783488LL, 112588272697344LL, 77498389889024LL, 40716289966080LL};
static const long long LOG2_B1[32] = {27565LL, 24786LL, 22167LL, 19696LL, 17361LL, 15150LL, 13054LL, 11064LL, 9173LL, 7372LL, 5657LL, 4020LL, 2457LL, 962LL, -468LL, -1838LL, -3151LL, -4412LL, -5622LL, -6786LL, -7905LL, -8982LL, -10020LL, -11020LL, -11985LL, -12916LL, -13816LL, -14685LL, -15526LL, -16339LL, -17126LL, -17889LL};
static const double LOG2_R1[33] =
    {0x1.71548p+0, 0x1.696af492p+0, 0x1.61ab3fb68p+0, 0x1.5a1844168p+0,
    0x1.52adadb48p+0, 0x1.4b6b7c908p+0, 0x1.4451b0aa8p+0, 0x1.3d5ed8aep+0,
    0x1.369183468p+0, 0x1.2fe9b074p+0, 0x1.296760368p+0, 0x1.2307afe5p+0,
    0x1.1cca9f7f8p+0, 0x1.16b02f06p+0, 0x1.10b85e788p+0, 0x1.0ae04b2ep+0,
    0x1.0527f5268p+0, 0x1.ff1eb8c4p-1, 0x1.f42a1f18p-1, 0x1.e9721d49p-1,
    0x1.def6b357p-1, 0x1.d4b21bfp-1, 0x1.caa739bdp-1, 0x1.c0d32a15p-1,
    0x1.b735ecf8p-1, 0x1.adcc9fbdp-1, 0x1.a4974264p-1, 0x1.9b92f244p-1,
    0x1.92c29206p-1, 0x1.8a205c58p-1, 0x1.81af33e3p-1, 0x1.796c35fep-1, 0x1.71548p-1};
static const double LOG2_R2[33] =
    {0x1p+0, 0x1.ffa7p-1, 0x1.ff4fp-1, 0x1.fef6p-1, 0x1.fe9ep-1, 0x1.fe45p-1,
    0x1.fdedp-1, 0x1.fd94p-1, 0x1.fd3cp-1, 0x1.fce4p-1, 0x1.fc8cp-1, 0x1.fc34p-1,
    0x1.fbdcp-1, 0x1.fb84p-1, 0x1.fb2cp-1, 0x1.fad4p-1, 0x1.fa7cp-1, 0x1.fa24p-1,
    0x1.f9cdp-1, 0x1.f975p-1, 0x1.f91ep-1, 0x1.f8c6p-1, 0x1.f86fp-1, 0x1.f817p-1,
    0x1.f7cp-1, 0x1.f769p-1, 0x1.f711p-1, 0x1.f6bap-1, 0x1.f663p-1, 0x1.f60cp-1,
    0x1.f5b5p-1, 0x1.f55ep-1, 0x1.f507p-1};
static const double LOG2_L1[33][2] = {
    {0x0p+0, 0x0p+0}, {0x1.1435edc775b51p-26, 0x1.ffe38p-6},
    {0x1.c8f1cbf9e4073p-26, 0x1.000bcp-4}, {-0x1.7bf30fa53957bp-26, 0x1.7ff94p-4},
    {0x1.674d30b6276edp-27, 0x1.fffa6p-4}, {0x1.99fcf0d796acep-28, 0x1.400058p-3},
    {-0x1.5ffd8b92706d2p-26, 0x1.7ffe5p-3}, {0x1.bef90bc5a116dp-26, 0x1.bffc3p-3},
    {0x1.fca73b3d53f0dp-26, 0x1.ffffdp-3}, {-0x1.41024e560e04ep-27, 0x1.2000dp-2},
    {0x1.484024fad8461p-26, 0x1.3ffccp-2}, {0x1.2d9ab90ba7694p-26, 0x1.5ffe18p-2},
    {-0x1.2c998ea30ba7bp-26, 0x1.80013cp-2}, {-0x1.75fc8682f918ep-27, 0x1.a0024cp-2},
    {-0x1.f02f268a85fb8p-26, 0x1.bffd3p-2}, {-0x1.50030ea7fae4bp-27, 0x1.dffd7cp-2},
    {-0x1.bffb8da5b849dp-28, 0x1.ffff98p-2}, {0x1.f71993ff95475p-28, 0x1.0fffd4p-1},
    {0x1.88028e67f78fap-29, 0x1.20010cp-1}, {0x1.f915f5a0b4e89p-26, 0x1.30018ap-1},
    {0x1.c2fe288f968f8p-30, 0x1.3fff48p-1}, {0x1.7375a75ae0837p-26, 0x1.50013p-1},
    {0x1.91e48be920323p-29, 0x1.6000eap-1}, {0x1.ee7bc0d39a3dbp-29, 0x1.700112p-1},
    {-0x1.899e2ac5f778cp-29, 0x1.7fffb4p-1}, {0x1.f1f20176130a7p-26, 0x1.8fffaep-1},
    {0x1.03fc59d34a4f3p-27, 0x1.9fff1ep-1}, {-0x1.68722010e4653p-32, 0x1.b0012ap-1},
    {-0x1.58783d505a6ecp-26, 0x1.bffeb8p-1}, {0x1.b212ab9f8d51dp-27, 0x1.d0004ep-1},
    {-0x1.b3ecd767be776p-26, 0x1.dffeb2p-1}, {0x1.b1ad41f07fc1p-27, 0x1.effd52p-1},
    {0x0p+0, 0x1p+0}};
static const double LOG2_L2[33][2] = {
    {0x0p+0, 0x0p+0}, {-0x1.e2b19f9c7b84p-27, 0x1.00e4p-10},
    {-0x1.b1d68137631fep-27, 0x1.ff1p-10}, {-0x1.aa92227513fc3p-27, 0x1.8026p-9},
    {-0x1.52918e3ab6f5ep-27, 0x1.ff68p-9}, {0x1.1a0b9b9010a9cp-27, 0x1.4019p-8},
    {0x1.09b87f57867ecp-26, 0x1.7fdp-8}, {-0x1.3d6b70c673be6p-28, 0x1.c04cp-8},
    {0x1.76d340a6780abp-27, 0x1.000c8p-7}, {0x1.e4181e37d9e05p-26, 0x1.1ff88p-7},
    {-0x1.8c181f042b901p-27, 0x1.3fea8p-7}, {0x1.9890ae7761d66p-27, 0x1.5fe18p-7},
    {-0x1.55977ae613d5fp-26, 0x1.7fde8p-7}, {0x1.c75f49acf5e56p-27, 0x1.9fe08p-7},
    {-0x1.e1fc84d0d42bcp-28, 0x1.bfe88p-7}, {-0x1.5ab8a182ed279p-26, 0x1.dff6p-7},
    {-0x1.b389a7d8a21d9p-26, 0x1.00048p-6}, {-0x1.7376f1a891fa5p-26, 0x1.1010cp-6},
    {-0x1.0915581b87a8ap-28, 0x1.1ff1p-6}, {0x1.b7f2eaa894fcap-27, 0x1.3002cp-6},
    {-0x1.f8c41bdd38c23p-26, 0x1.3fe8cp-6}, {0x1.04ffb1f3c8215p-28, 0x1.5p-6},
    {0x1.a398ccf2b137dp-26, 0x1.5feb4p-6}, {0x1.373b5bce07f39p-26, 0x1.70084p-6},
    {-0x1.e42040fd9d454p-27, 0x1.7ff94p-6}, {0x1.ea00eb5770526p-27, 0x1.8feccp-6},
    {-0x1.3e2002bb54bb8p-26, 0x1.a0124p-6}, {0x1.797ec835db8d3p-26, 0x1.b00b4p-6},
    {0x1.bdcd45fedd285p-28, 0x1.c0074p-6}, {-0x1.33b8f09d1a21p-28, 0x1.d006p-6},
    {-0x1.58937a8eea36fp-27, 0x1.e0078p-6}, {-0x1.3e76203ff54cdp-27, 0x1.f00bcp-6},
    {-0x1.6560f19fc3f41p-30, 0x1.00096p-5}};
