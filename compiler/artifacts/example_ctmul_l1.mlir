module {
  func.func @ct_mul_l1(%c00: tensor<16xi32>, %c01: tensor<16xi32>, %c02: tensor<16xi32>, %c03: tensor<16xi32>, %c04: tensor<16xi32>, %c05: tensor<16xi32>, %c06: tensor<16xi32>, %c10: tensor<16xi32>, %c11: tensor<16xi32>, %c12: tensor<16xi32>, %c13: tensor<16xi32>, %c14: tensor<16xi32>, %c15: tensor<16xi32>, %c16: tensor<16xi32>, %d00: tensor<16xi32>, %d01: tensor<16xi32>, %d02: tensor<16xi32>, %d03: tensor<16xi32>, %d04: tensor<16xi32>, %d05: tensor<16xi32>, %d06: tensor<16xi32>, %d10: tensor<16xi32>, %d11: tensor<16xi32>, %d12: tensor<16xi32>, %d13: tensor<16xi32>, %d14: tensor<16xi32>, %d15: tensor<16xi32>, %d16: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %tC00 = rns.ntt %c00 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC10 = rns.ntt %c10 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD00 = rns.ntt %d00 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD10 = rns.ntt %d10 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tE00 = rns.mul %tC00, %tD00 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE20 = rns.mul %tC10, %tD10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT00 = rns.mul %tC00, %tD10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT10 = rns.mul %tC10, %tD00 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE10 = rns.add %tT00, %tT10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_0 = rns.intt %tE00 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_0 = rns.intt %tE10 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_0 = rns.intt %tE20 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC01 = rns.ntt %c01 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC11 = rns.ntt %c11 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD01 = rns.ntt %d01 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD11 = rns.ntt %d11 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tE01 = rns.mul %tC01, %tD01 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE21 = rns.mul %tC11, %tD11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT01 = rns.mul %tC01, %tD11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT11 = rns.mul %tC11, %tD01 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE11 = rns.add %tT01, %tT11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_1 = rns.intt %tE01 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_1 = rns.intt %tE11 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_1 = rns.intt %tE21 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC02 = rns.ntt %c02 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC12 = rns.ntt %c12 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD02 = rns.ntt %d02 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD12 = rns.ntt %d12 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tE02 = rns.mul %tC02, %tD02 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE22 = rns.mul %tC12, %tD12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT02 = rns.mul %tC02, %tD12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT12 = rns.mul %tC12, %tD02 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE12 = rns.add %tT02, %tT12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_2 = rns.intt %tE02 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_2 = rns.intt %tE12 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_2 = rns.intt %tE22 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC03 = rns.ntt %c03 {N = 16 : i64, q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC13 = rns.ntt %c13 {N = 16 : i64, q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD03 = rns.ntt %d03 {N = 16 : i64, q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD13 = rns.ntt %d13 {N = 16 : i64, q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tE03 = rns.mul %tC03, %tD03 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE23 = rns.mul %tC13, %tD13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT03 = rns.mul %tC03, %tD13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT13 = rns.mul %tC13, %tD03 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE13 = rns.add %tT03, %tT13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_3 = rns.intt %tE03 {N = 16 : i64, q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_3 = rns.intt %tE13 {N = 16 : i64, q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_3 = rns.intt %tE23 {N = 16 : i64, q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC04 = rns.ntt %c04 {N = 16 : i64, q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC14 = rns.ntt %c14 {N = 16 : i64, q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD04 = rns.ntt %d04 {N = 16 : i64, q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD14 = rns.ntt %d14 {N = 16 : i64, q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tE04 = rns.mul %tC04, %tD04 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE24 = rns.mul %tC14, %tD14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT04 = rns.mul %tC04, %tD14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT14 = rns.mul %tC14, %tD04 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE14 = rns.add %tT04, %tT14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_4 = rns.intt %tE04 {N = 16 : i64, q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_4 = rns.intt %tE14 {N = 16 : i64, q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_4 = rns.intt %tE24 {N = 16 : i64, q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC05 = rns.ntt %c05 {N = 16 : i64, q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC15 = rns.ntt %c15 {N = 16 : i64, q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD05 = rns.ntt %d05 {N = 16 : i64, q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD15 = rns.ntt %d15 {N = 16 : i64, q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tE05 = rns.mul %tC05, %tD05 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE25 = rns.mul %tC15, %tD15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT05 = rns.mul %tC05, %tD15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT15 = rns.mul %tC15, %tD05 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE15 = rns.add %tT05, %tT15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_5 = rns.intt %tE05 {N = 16 : i64, q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_5 = rns.intt %tE15 {N = 16 : i64, q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_5 = rns.intt %tE25 {N = 16 : i64, q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC06 = rns.ntt %c06 {N = 16 : i64, q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC16 = rns.ntt %c16 {N = 16 : i64, q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD06 = rns.ntt %d06 {N = 16 : i64, q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD16 = rns.ntt %d16 {N = 16 : i64, q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tE06 = rns.mul %tC06, %tD06 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE26 = rns.mul %tC16, %tD16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT06 = rns.mul %tC06, %tD16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT16 = rns.mul %tC16, %tD06 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE16 = rns.add %tT06, %tT16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_6 = rns.intt %tE06 {N = 16 : i64, q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_6 = rns.intt %tE16 {N = 16 : i64, q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_6 = rns.intt %tE26 {N = 16 : i64, q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %scal_t = rns.const {name = "scal_t"} : tensor<1xi32>
    %rs0_fQ0 = rns.mul %e0_0, %scal_t {q = 268435361 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fQ1 = rns.mul %e0_1, %scal_t {q = 268435009 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fQ2 = rns.mul %e0_2, %scal_t {q = 268434977 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fS0 = rns.mul %e0_3, %scal_t {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fS1 = rns.mul %e0_4, %scal_t {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fS2 = rns.mul %e0_5, %scal_t {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fS3 = rns.mul %e0_6, %scal_t {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fbc0, %rs0_fbc1, %rs0_fbc2, %rs0_fbc3 = rns.base_convert %rs0_fQ0, %rs0_fQ1, %rs0_fQ2 : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    %rs0_d0 = rns.sub %rs0_fS0, %rs0_fbc0 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Qinv_268433569 = rns.const {name = "Qinv_268433569"} : tensor<1xi32>
    %rs0_etS0 = rns.mul %rs0_d0, %Qinv_268433569 {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_d1 = rns.sub %rs0_fS1, %rs0_fbc1 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Qinv_268433441 = rns.const {name = "Qinv_268433441"} : tensor<1xi32>
    %rs0_etS1 = rns.mul %rs0_d1, %Qinv_268433441 {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_d2 = rns.sub %rs0_fS2, %rs0_fbc2 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Qinv_268433377 = rns.const {name = "Qinv_268433377"} : tensor<1xi32>
    %rs0_etS2 = rns.mul %rs0_d2, %Qinv_268433377 {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_d3 = rns.sub %rs0_fS3, %rs0_fbc3 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Qinv_268433057 = rns.const {name = "Qinv_268433057"} : tensor<1xi32>
    %rs0_etS3 = rns.mul %rs0_d3, %Qinv_268433057 {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_etQ0, %rs0_etQ1, %rs0_etQ2 = rns.base_convert %rs0_etS0, %rs0_etS1, %rs0_etS2, %rs0_etS3 {exact, signed, u = 38 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    %rs1_fQ0 = rns.mul %e1_0, %scal_t {q = 268435361 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fQ1 = rns.mul %e1_1, %scal_t {q = 268435009 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fQ2 = rns.mul %e1_2, %scal_t {q = 268434977 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fS0 = rns.mul %e1_3, %scal_t {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fS1 = rns.mul %e1_4, %scal_t {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fS2 = rns.mul %e1_5, %scal_t {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fS3 = rns.mul %e1_6, %scal_t {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fbc0, %rs1_fbc1, %rs1_fbc2, %rs1_fbc3 = rns.base_convert %rs1_fQ0, %rs1_fQ1, %rs1_fQ2 : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    %rs1_d0 = rns.sub %rs1_fS0, %rs1_fbc0 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs1_etS0 = rns.mul %rs1_d0, %Qinv_268433569 {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_d1 = rns.sub %rs1_fS1, %rs1_fbc1 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs1_etS1 = rns.mul %rs1_d1, %Qinv_268433441 {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_d2 = rns.sub %rs1_fS2, %rs1_fbc2 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs1_etS2 = rns.mul %rs1_d2, %Qinv_268433377 {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_d3 = rns.sub %rs1_fS3, %rs1_fbc3 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs1_etS3 = rns.mul %rs1_d3, %Qinv_268433057 {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_etQ0, %rs1_etQ1, %rs1_etQ2 = rns.base_convert %rs1_etS0, %rs1_etS1, %rs1_etS2, %rs1_etS3 {exact, signed, u = 38 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    %rs2_fQ0 = rns.mul %e2_0, %scal_t {q = 268435361 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fQ1 = rns.mul %e2_1, %scal_t {q = 268435009 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fQ2 = rns.mul %e2_2, %scal_t {q = 268434977 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fS0 = rns.mul %e2_3, %scal_t {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fS1 = rns.mul %e2_4, %scal_t {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fS2 = rns.mul %e2_5, %scal_t {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fS3 = rns.mul %e2_6, %scal_t {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fbc0, %rs2_fbc1, %rs2_fbc2, %rs2_fbc3 = rns.base_convert %rs2_fQ0, %rs2_fQ1, %rs2_fQ2 : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    %rs2_d0 = rns.sub %rs2_fS0, %rs2_fbc0 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs2_etS0 = rns.mul %rs2_d0, %Qinv_268433569 {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_d1 = rns.sub %rs2_fS1, %rs2_fbc1 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs2_etS1 = rns.mul %rs2_d1, %Qinv_268433441 {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_d2 = rns.sub %rs2_fS2, %rs2_fbc2 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs2_etS2 = rns.mul %rs2_d2, %Qinv_268433377 {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_d3 = rns.sub %rs2_fS3, %rs2_fbc3 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs2_etS3 = rns.mul %rs2_d3, %Qinv_268433057 {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_etQ0, %rs2_etQ1, %rs2_etQ2 = rns.base_convert %rs2_etS0, %rs2_etS1, %rs2_etS2, %rs2_etS3 {exact, signed, u = 38 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    %rl_c2P0, %rl_c2P1, %rl_c2P2 = rns.base_convert %rs2_etQ0, %rs2_etQ1, %rs2_etQ2 : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    %rl_X0 = rns.ntt %rs2_etQ0 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X1 = rns.ntt %rs2_etQ1 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X2 = rns.ntt %rs2_etQ2 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X3 = rns.ntt %rl_c2P0 {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X4 = rns.ntt %rl_c2P1 {N = 16 : i64, q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X5 = rns.ntt %rl_c2P2 {N = 16 : i64, q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_0c = rns.const {name = "rl_evkA_0c", q = 268435361 : i64} : tensor<16xi32>
    %rl_evkA_0 = rns.ntt %rl_evkA_0c {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_AY0 = rns.mul %rl_X0, %rl_evkA_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa0 = rns.intt %rl_AY0 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_1c = rns.const {name = "rl_evkA_1c", q = 268435009 : i64} : tensor<16xi32>
    %rl_evkA_1 = rns.ntt %rl_evkA_1c {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_AY1 = rns.mul %rl_X1, %rl_evkA_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa1 = rns.intt %rl_AY1 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_2c = rns.const {name = "rl_evkA_2c", q = 268434977 : i64} : tensor<16xi32>
    %rl_evkA_2 = rns.ntt %rl_evkA_2c {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_AY2 = rns.mul %rl_X2, %rl_evkA_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa2 = rns.intt %rl_AY2 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_3c = rns.const {name = "rl_evkA_3c", q = 268434721 : i64} : tensor<16xi32>
    %rl_evkA_3 = rns.ntt %rl_evkA_3c {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_AY3 = rns.mul %rl_X3, %rl_evkA_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa3 = rns.intt %rl_AY3 {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_4c = rns.const {name = "rl_evkA_4c", q = 268434401 : i64} : tensor<16xi32>
    %rl_evkA_4 = rns.ntt %rl_evkA_4c {N = 16 : i64, q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_AY4 = rns.mul %rl_X4, %rl_evkA_4 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa4 = rns.intt %rl_AY4 {N = 16 : i64, q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_5c = rns.const {name = "rl_evkA_5c", q = 268433761 : i64} : tensor<16xi32>
    %rl_evkA_5 = rns.ntt %rl_evkA_5c {N = 16 : i64, q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_AY5 = rns.mul %rl_X5, %rl_evkA_5 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa5 = rns.intt %rl_AY5 {N = 16 : i64, q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_0c = rns.const {name = "rl_evkB_0c", q = 268435361 : i64} : tensor<16xi32>
    %rl_evkB_0 = rns.ntt %rl_evkB_0c {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_BY0 = rns.mul %rl_X0, %rl_evkB_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba0 = rns.intt %rl_BY0 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_1c = rns.const {name = "rl_evkB_1c", q = 268435009 : i64} : tensor<16xi32>
    %rl_evkB_1 = rns.ntt %rl_evkB_1c {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_BY1 = rns.mul %rl_X1, %rl_evkB_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba1 = rns.intt %rl_BY1 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_2c = rns.const {name = "rl_evkB_2c", q = 268434977 : i64} : tensor<16xi32>
    %rl_evkB_2 = rns.ntt %rl_evkB_2c {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_BY2 = rns.mul %rl_X2, %rl_evkB_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba2 = rns.intt %rl_BY2 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_3c = rns.const {name = "rl_evkB_3c", q = 268434721 : i64} : tensor<16xi32>
    %rl_evkB_3 = rns.ntt %rl_evkB_3c {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_BY3 = rns.mul %rl_X3, %rl_evkB_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba3 = rns.intt %rl_BY3 {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_4c = rns.const {name = "rl_evkB_4c", q = 268434401 : i64} : tensor<16xi32>
    %rl_evkB_4 = rns.ntt %rl_evkB_4c {N = 16 : i64, q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_BY4 = rns.mul %rl_X4, %rl_evkB_4 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba4 = rns.intt %rl_BY4 {N = 16 : i64, q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_5c = rns.const {name = "rl_evkB_5c", q = 268433761 : i64} : tensor<16xi32>
    %rl_evkB_5 = rns.ntt %rl_evkB_5c {N = 16 : i64, q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_BY5 = rns.mul %rl_X5, %rl_evkB_5 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba5 = rns.intt %rl_BY5 {N = 16 : i64, q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_AxP0, %rl_AxP1, %rl_AxP2 = rns.base_convert %rl_Aa3, %rl_Aa4, %rl_Aa5 : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    %rl_Ad0 = rns.sub %rl_Aa0, %rl_AxP0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Pinv_268435361 = rns.const {name = "Pinv_268435361"} : tensor<1xi32>
    %rl_Ao0 = rns.mul %rl_Ad0, %Pinv_268435361 {q = 268435361 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rl_Ad1 = rns.sub %rl_Aa1, %rl_AxP1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Pinv_268435009 = rns.const {name = "Pinv_268435009"} : tensor<1xi32>
    %rl_Ao1 = rns.mul %rl_Ad1, %Pinv_268435009 {q = 268435009 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rl_Ad2 = rns.sub %rl_Aa2, %rl_AxP2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Pinv_268434977 = rns.const {name = "Pinv_268434977"} : tensor<1xi32>
    %rl_Ao2 = rns.mul %rl_Ad2, %Pinv_268434977 {q = 268434977 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rl_BxP0, %rl_BxP1, %rl_BxP2 = rns.base_convert %rl_Ba3, %rl_Ba4, %rl_Ba5 : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    %rl_Bd0 = rns.sub %rl_Ba0, %rl_BxP0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Bo0 = rns.mul %rl_Bd0, %Pinv_268435361 {q = 268435361 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rl_Bd1 = rns.sub %rl_Ba1, %rl_BxP1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Bo1 = rns.mul %rl_Bd1, %Pinv_268435009 {q = 268435009 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rl_Bd2 = rns.sub %rl_Ba2, %rl_BxP2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Bo2 = rns.mul %rl_Bd2, %Pinv_268434977 {q = 268434977 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %out0_0 = rns.add %rs0_etQ0, %rl_Ao0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %out1_0 = rns.add %rs1_etQ0, %rl_Bo0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %out0_1 = rns.add %rs0_etQ1, %rl_Ao1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %out1_1 = rns.add %rs1_etQ1, %rl_Bo1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %out0_2 = rns.add %rs0_etQ2, %rl_Ao2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %out1_2 = rns.add %rs1_etQ2, %rl_Bo2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    return %out0_0, %out0_1, %out0_2, %out1_0, %out1_1, %out1_2 : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }
}
