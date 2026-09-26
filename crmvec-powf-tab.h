/* copied verbatim from CORE-MATH src/binary32/pow/powf.c (MIT; see powf.c here):
   the fast path's log2 tables POWF_IX[33], POWF_LIX[33][2] and exp2 table POWF_TB[16] */
static const double POWF_IX[33] = {
    0x1p+0, 0x1.f07c1f07cp-1, 0x1.e1e1e1e1ep-1, 0x1.d41d41d42p-1,
    0x1.c71c71c72p-1, 0x1.bacf914c2p-1, 0x1.af286bca2p-1, 0x1.a41a41a42p-1,
    0x1.99999999ap-1, 0x1.8f9c18f9cp-1, 0x1.861861862p-1, 0x1.7d05f417dp-1,
    0x1.745d1745dp-1, 0x1.6c16c16c1p-1, 0x1.642c8590bp-1, 0x1.5c9882b93p-1,
    0x1.555555555p-1, 0x1.4e5e0a72fp-1, 0x1.47ae147aep-1, 0x1.414141414p-1,
    0x1.3b13b13b1p-1, 0x1.3521cfb2bp-1, 0x1.2f684bda1p-1, 0x1.29e4129e4p-1,
    0x1.249249249p-1, 0x1.1f7047dc1p-1, 0x1.1a7b9611ap-1, 0x1.15b1e5f75p-1,
    0x1.111111111p-1, 0x1.0c9714fbdp-1, 0x1.084210842p-1, 0x1.041041041p-1, 0x1p-1
  };
static const double POWF_LIX[33][2] = {
    {0x0p+0, 0x0p+0}, {-0x1.6cp-5, 0x1.4b229b87f3f89p-15},
    {-0x1.66p-4, -0x1.fb7d654235799p-15}, {-0x1.08p-3, -0x1.8b119b2c9c87bp-12},
    {-0x1.5cp-3, -0x1.a39fa6533294dp-19}, {-0x1.acp-3, -0x1.ebc5b663dd4b8p-12},
    {-0x1.fcp-3, 0x1.f4a37fe0fa46fp-14}, {-0x1.24p-2, -0x1.01eac33103e6bp-12},
    {-0x1.4ap-2, 0x1.61ed0d15725ep-12}, {-0x1.6ep-2, -0x1.10e6ceb499ba9p-13},
    {-0x1.92p-2, 0x1.115db8ada837dp-12}, {-0x1.b4p-2, -0x1.fafdce266d7aep-12},
    {-0x1.d6p-2, -0x1.d4f80cd19906fp-12}, {-0x1.f8p-2, 0x1.5ea5ccd0a7396p-12},
    {0x1.e8p-2, -0x1.0500d67fe62ebp-13}, {0x1.c8p-2, 0x1.9dc2d41aa4626p-14},
    {0x1.a8p-2, 0x1.ff2e2ff321344p-11}, {0x1.8ap-2, 0x1.130157f4c3a3ep-11},
    {0x1.6cp-2, 0x1.61ed0cad929ccp-11}, {0x1.5p-2, -0x1.2089a632d7949p-11},
    {0x1.32p-2, 0x1.7fdc6dfb2d21ap-11}, {0x1.16p-2, 0x1.380a6c36088f3p-11},
    {0x1.f6p-3, -0x1.3ab7dc7ba81acp-18}, {0x1.cp-3, -0x1.cc2c0061ef1a2p-14},
    {0x1.8ap-3, 0x1.130157c97bbep-12}, {0x1.56p-3, 0x1.ee14ff34c4128p-14},
    {0x1.22p-3, 0x1.b5b854c4fde69p-12}, {0x1.ep-4, 0x1.635d1df7cb0b5p-13},
    {0x1.7ep-4, -0x1.3f6d2636c101ep-13}, {0x1.1cp-4, -0x1.33567f1b193a4p-14},
    {0x1.78p-5, -0x1.8d66c5313a71dp-14}, {0x1.74p-6, 0x1.f7430ee200ep-17}, {0x0p+0, 0x0p+0}
  };
static const double POWF_TB[16] =
    {0x1p+0, 0x1.0b5586cf9890fp+0, 0x1.172b83c7d517bp+0, 0x1.2387a6e756238p+0,
     0x1.306fe0a31b715p+0, 0x1.3dea64c123422p+0, 0x1.4bfdad5362a27p+0, 0x1.5ab07dd485429p+0,
     0x1.6a09e667f3bcdp+0, 0x1.7a11473eb0187p+0, 0x1.8ace5422aa0dbp+0, 0x1.9c49182a3f09p+0,
     0x1.ae89f995ad3adp+0, 0x1.c199bdd85529cp+0, 0x1.d5818dcfba487p+0, 0x1.ea4afa2a490dap+0};
