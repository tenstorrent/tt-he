module {
  func.func @ct_mul_l2(%c00: tensor<16xi32>, %c01: tensor<16xi32>, %c02: tensor<16xi32>, %c03: tensor<16xi32>, %c04: tensor<16xi32>, %c05: tensor<16xi32>, %c06: tensor<16xi32>, %c10: tensor<16xi32>, %c11: tensor<16xi32>, %c12: tensor<16xi32>, %c13: tensor<16xi32>, %c14: tensor<16xi32>, %c15: tensor<16xi32>, %c16: tensor<16xi32>, %d00: tensor<16xi32>, %d01: tensor<16xi32>, %d02: tensor<16xi32>, %d03: tensor<16xi32>, %d04: tensor<16xi32>, %d05: tensor<16xi32>, %d06: tensor<16xi32>, %d10: tensor<16xi32>, %d11: tensor<16xi32>, %d12: tensor<16xi32>, %d13: tensor<16xi32>, %d14: tensor<16xi32>, %d15: tensor<16xi32>, %d16: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %W_f_268435361 = tile.const {name = "W_f_268435361", q = 268435361 : i64} : tensor<16x16xi32>
    %tC00 = tile.matmul_modq %W_f_268435361, %c00 {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tC10 = tile.matmul_modq %W_f_268435361, %c10 {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD00 = tile.matmul_modq %W_f_268435361, %d00 {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD10 = tile.matmul_modq %W_f_268435361, %d10 {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE00 = tile.mul_modq %tC00, %tD00 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE20 = tile.mul_modq %tC10, %tD10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT00 = tile.mul_modq %tC00, %tD10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT10 = tile.mul_modq %tC10, %tD00 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE10 = tile.add_modq %tT00, %tT10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268435361 = tile.const {name = "W_i_268435361", q = 268435361 : i64} : tensor<16x16xi32>
    %e0_0 = tile.matmul_modq %W_i_268435361, %tE00 {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_0 = tile.matmul_modq %W_i_268435361, %tE10 {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_0 = tile.matmul_modq %W_i_268435361, %tE20 {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268435009 = tile.const {name = "W_f_268435009", q = 268435009 : i64} : tensor<16x16xi32>
    %tC01 = tile.matmul_modq %W_f_268435009, %c01 {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tC11 = tile.matmul_modq %W_f_268435009, %c11 {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD01 = tile.matmul_modq %W_f_268435009, %d01 {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD11 = tile.matmul_modq %W_f_268435009, %d11 {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE01 = tile.mul_modq %tC01, %tD01 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE21 = tile.mul_modq %tC11, %tD11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT01 = tile.mul_modq %tC01, %tD11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT11 = tile.mul_modq %tC11, %tD01 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE11 = tile.add_modq %tT01, %tT11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268435009 = tile.const {name = "W_i_268435009", q = 268435009 : i64} : tensor<16x16xi32>
    %e0_1 = tile.matmul_modq %W_i_268435009, %tE01 {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_1 = tile.matmul_modq %W_i_268435009, %tE11 {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_1 = tile.matmul_modq %W_i_268435009, %tE21 {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268434977 = tile.const {name = "W_f_268434977", q = 268434977 : i64} : tensor<16x16xi32>
    %tC02 = tile.matmul_modq %W_f_268434977, %c02 {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tC12 = tile.matmul_modq %W_f_268434977, %c12 {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD02 = tile.matmul_modq %W_f_268434977, %d02 {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD12 = tile.matmul_modq %W_f_268434977, %d12 {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE02 = tile.mul_modq %tC02, %tD02 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE22 = tile.mul_modq %tC12, %tD12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT02 = tile.mul_modq %tC02, %tD12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT12 = tile.mul_modq %tC12, %tD02 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE12 = tile.add_modq %tT02, %tT12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268434977 = tile.const {name = "W_i_268434977", q = 268434977 : i64} : tensor<16x16xi32>
    %e0_2 = tile.matmul_modq %W_i_268434977, %tE02 {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_2 = tile.matmul_modq %W_i_268434977, %tE12 {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_2 = tile.matmul_modq %W_i_268434977, %tE22 {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268433569 = tile.const {name = "W_f_268433569", q = 268433569 : i64} : tensor<16x16xi32>
    %tC03 = tile.matmul_modq %W_f_268433569, %c03 {q = 268433569 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tC13 = tile.matmul_modq %W_f_268433569, %c13 {q = 268433569 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD03 = tile.matmul_modq %W_f_268433569, %d03 {q = 268433569 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD13 = tile.matmul_modq %W_f_268433569, %d13 {q = 268433569 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE03 = tile.mul_modq %tC03, %tD03 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE23 = tile.mul_modq %tC13, %tD13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT03 = tile.mul_modq %tC03, %tD13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT13 = tile.mul_modq %tC13, %tD03 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE13 = tile.add_modq %tT03, %tT13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433569 = tile.const {name = "W_i_268433569", q = 268433569 : i64} : tensor<16x16xi32>
    %e0_3 = tile.matmul_modq %W_i_268433569, %tE03 {q = 268433569 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_3 = tile.matmul_modq %W_i_268433569, %tE13 {q = 268433569 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_3 = tile.matmul_modq %W_i_268433569, %tE23 {q = 268433569 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268433441 = tile.const {name = "W_f_268433441", q = 268433441 : i64} : tensor<16x16xi32>
    %tC04 = tile.matmul_modq %W_f_268433441, %c04 {q = 268433441 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tC14 = tile.matmul_modq %W_f_268433441, %c14 {q = 268433441 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD04 = tile.matmul_modq %W_f_268433441, %d04 {q = 268433441 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD14 = tile.matmul_modq %W_f_268433441, %d14 {q = 268433441 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE04 = tile.mul_modq %tC04, %tD04 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE24 = tile.mul_modq %tC14, %tD14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT04 = tile.mul_modq %tC04, %tD14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT14 = tile.mul_modq %tC14, %tD04 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE14 = tile.add_modq %tT04, %tT14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433441 = tile.const {name = "W_i_268433441", q = 268433441 : i64} : tensor<16x16xi32>
    %e0_4 = tile.matmul_modq %W_i_268433441, %tE04 {q = 268433441 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_4 = tile.matmul_modq %W_i_268433441, %tE14 {q = 268433441 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_4 = tile.matmul_modq %W_i_268433441, %tE24 {q = 268433441 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268433377 = tile.const {name = "W_f_268433377", q = 268433377 : i64} : tensor<16x16xi32>
    %tC05 = tile.matmul_modq %W_f_268433377, %c05 {q = 268433377 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tC15 = tile.matmul_modq %W_f_268433377, %c15 {q = 268433377 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD05 = tile.matmul_modq %W_f_268433377, %d05 {q = 268433377 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD15 = tile.matmul_modq %W_f_268433377, %d15 {q = 268433377 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE05 = tile.mul_modq %tC05, %tD05 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE25 = tile.mul_modq %tC15, %tD15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT05 = tile.mul_modq %tC05, %tD15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT15 = tile.mul_modq %tC15, %tD05 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE15 = tile.add_modq %tT05, %tT15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433377 = tile.const {name = "W_i_268433377", q = 268433377 : i64} : tensor<16x16xi32>
    %e0_5 = tile.matmul_modq %W_i_268433377, %tE05 {q = 268433377 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_5 = tile.matmul_modq %W_i_268433377, %tE15 {q = 268433377 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_5 = tile.matmul_modq %W_i_268433377, %tE25 {q = 268433377 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268433057 = tile.const {name = "W_f_268433057", q = 268433057 : i64} : tensor<16x16xi32>
    %tC06 = tile.matmul_modq %W_f_268433057, %c06 {q = 268433057 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tC16 = tile.matmul_modq %W_f_268433057, %c16 {q = 268433057 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD06 = tile.matmul_modq %W_f_268433057, %d06 {q = 268433057 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tD16 = tile.matmul_modq %W_f_268433057, %d16 {q = 268433057 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE06 = tile.mul_modq %tC06, %tD06 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE26 = tile.mul_modq %tC16, %tD16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT06 = tile.mul_modq %tC06, %tD16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT16 = tile.mul_modq %tC16, %tD06 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE16 = tile.add_modq %tT06, %tT16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433057 = tile.const {name = "W_i_268433057", q = 268433057 : i64} : tensor<16x16xi32>
    %e0_6 = tile.matmul_modq %W_i_268433057, %tE06 {q = 268433057 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_6 = tile.matmul_modq %W_i_268433057, %tE16 {q = 268433057 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_6 = tile.matmul_modq %W_i_268433057, %tE26 {q = 268433057 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %scal_t = rns.const {name = "scal_t"} : tensor<1xi32>
    %rs0_fQ0 = tile.mul_modq %e0_0, %scal_t {q = 268435361 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fQ1 = tile.mul_modq %e0_1, %scal_t {q = 268435009 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fQ2 = tile.mul_modq %e0_2, %scal_t {q = 268434977 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fS0 = tile.mul_modq %e0_3, %scal_t {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fS1 = tile.mul_modq %e0_4, %scal_t {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fS2 = tile.mul_modq %e0_5, %scal_t {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_fS3 = tile.mul_modq %e0_6, %scal_t {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %fbcC__268435361__268435009__268434977__268435361 = tile.const {name = "fbcC_(268435361, 268435009, 268434977)_268435361", q = 268435361 : i64} : tensor<1x1xi32>
    %rs0_fbc0_o_rs0_fQ0 = tile.reshape %rs0_fQ0 {shape = [1, 16], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_y_rs0_fQ0 = tile.mul_modq %rs0_fbc0_o_rs0_fQ0, %fbcC__268435361__268435009__268434977__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %fbcC__268435361__268435009__268434977__268435009 = tile.const {name = "fbcC_(268435361, 268435009, 268434977)_268435009", q = 268435009 : i64} : tensor<1x1xi32>
    %rs0_fbc0_o_rs0_fQ1 = tile.reshape %rs0_fQ1 {shape = [1, 16], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_y_rs0_fQ1 = tile.mul_modq %rs0_fbc0_o_rs0_fQ1, %fbcC__268435361__268435009__268434977__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %fbcC__268435361__268435009__268434977__268434977 = tile.const {name = "fbcC_(268435361, 268435009, 268434977)_268434977", q = 268434977 : i64} : tensor<1x1xi32>
    %rs0_fbc0_o_rs0_fQ2 = tile.reshape %rs0_fQ2 {shape = [1, 16], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_y_rs0_fQ2 = tile.mul_modq %rs0_fbc0_o_rs0_fQ2, %fbcC__268435361__268435009__268434977__268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_Y = tile.concat %rs0_fbc0_y_rs0_fQ0, %rs0_fbc0_y_rs0_fQ1, %rs0_fbc0_y_rs0_fQ2 {axis = 0 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<3x16xi32>
    %fbcD__268435361__268435009__268434977__268433569 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433569"} : tensor<1x3xi32>
    %rs0_fbc0_fbc_268433569 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433569, %rs0_fbc0_Y {q = 268433569 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0 = tile.reshape %rs0_fbc0_fbc_268433569 {shape = [16], q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433441 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433441"} : tensor<1x3xi32>
    %rs0_fbc0_fbc_268433441 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433441, %rs0_fbc0_Y {q = 268433441 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc1 = tile.reshape %rs0_fbc0_fbc_268433441 {shape = [16], q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433377 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433377"} : tensor<1x3xi32>
    %rs0_fbc0_fbc_268433377 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433377, %rs0_fbc0_Y {q = 268433377 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc2 = tile.reshape %rs0_fbc0_fbc_268433377 {shape = [16], q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433057 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433057"} : tensor<1x3xi32>
    %rs0_fbc0_fbc_268433057 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433057, %rs0_fbc0_Y {q = 268433057 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc3 = tile.reshape %rs0_fbc0_fbc_268433057 {shape = [16], q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs0_d0 = tile.sub_modq %rs0_fS0, %rs0_fbc0 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Qinv_268433569 = rns.const {name = "Qinv_268433569"} : tensor<1xi32>
    %rs0_etS0 = tile.mul_modq %rs0_d0, %Qinv_268433569 {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_d1 = tile.sub_modq %rs0_fS1, %rs0_fbc1 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Qinv_268433441 = rns.const {name = "Qinv_268433441"} : tensor<1xi32>
    %rs0_etS1 = tile.mul_modq %rs0_d1, %Qinv_268433441 {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_d2 = tile.sub_modq %rs0_fS2, %rs0_fbc2 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Qinv_268433377 = rns.const {name = "Qinv_268433377"} : tensor<1xi32>
    %rs0_etS2 = tile.mul_modq %rs0_d2, %Qinv_268433377 {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_d3 = tile.sub_modq %rs0_fS3, %rs0_fbc3 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Qinv_268433057 = rns.const {name = "Qinv_268433057"} : tensor<1xi32>
    %rs0_etS3 = tile.mul_modq %rs0_d3, %Qinv_268433057 {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs0_etQ0_r_rs0_etS0 = tile.reshape %rs0_etS0 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_r_rs0_etS1 = tile.reshape %rs0_etS1 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_r_rs0_etS2 = tile.reshape %rs0_etS2 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_r_rs0_etS3 = tile.reshape %rs0_etS3 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_RES = tile.concat %rs0_etQ0_r_rs0_etS0, %rs0_etQ0_r_rs0_etS1, %rs0_etQ0_r_rs0_etS2, %rs0_etQ0_r_rs0_etS3 {axis = 0 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<4x16xi32>
    %crnsF__268433569__268433441__268433377__268433057__38 = tile.const {name = "crnsF_(268433569, 268433441, 268433377, 268433057)_38"} : tensor<1x4xi32>
    %rs0_etQ0_s = tile.matmul_wide %crnsF__268433569__268433441__268433377__268433057__38, %rs0_etQ0_RES {radix = 128 : i64, nlimbs_a = 6 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_k = tile.fixed_slice %rs0_etQ0_s {shift = 38 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_neg = tile.fixed_slice %rs0_etQ0_s {shift = 37 : i64, mask = 1 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsE__268433569__268433441__268433377__268433057__268435361 = tile.const {name = "crnsE_(268433569, 268433441, 268433377, 268433057)_268435361"} : tensor<1x4xi32>
    %rs0_etQ0_xE_268435361 = tile.matmul_modq %crnsE__268433569__268433441__268433377__268433057__268435361, %rs0_etQ0_RES {q = 268435361 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %crnsg__268433569__268433441__268433377__268433057__268435361 = tile.const {name = "crnsg_(268433569, 268433441, 268433377, 268433057)_268435361", q = 268435361 : i64} : tensor<1x1xi32>
    %rs0_etQ0_kg_268435361 = tile.mul_modq %rs0_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_acc_268435361 = tile.add_modq %rs0_etQ0_xE_268435361, %rs0_etQ0_kg_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsMmod__268433569__268433441__268433377__268433057__268435361 = tile.const {name = "crnsMmod_(268433569, 268433441, 268433377, 268433057)_268435361", q = 268435361 : i64} : tensor<1x1xi32>
    %rs0_etQ0_negM_268435361 = tile.mul_modq %rs0_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_accs_268435361 = tile.sub_modq %rs0_etQ0_acc_268435361, %rs0_etQ0_negM_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0 = tile.reshape %rs0_etQ0_accs_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %crnsE__268433569__268433441__268433377__268433057__268435009 = tile.const {name = "crnsE_(268433569, 268433441, 268433377, 268433057)_268435009"} : tensor<1x4xi32>
    %rs0_etQ0_xE_268435009 = tile.matmul_modq %crnsE__268433569__268433441__268433377__268433057__268435009, %rs0_etQ0_RES {q = 268435009 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %crnsg__268433569__268433441__268433377__268433057__268435009 = tile.const {name = "crnsg_(268433569, 268433441, 268433377, 268433057)_268435009", q = 268435009 : i64} : tensor<1x1xi32>
    %rs0_etQ0_kg_268435009 = tile.mul_modq %rs0_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_acc_268435009 = tile.add_modq %rs0_etQ0_xE_268435009, %rs0_etQ0_kg_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsMmod__268433569__268433441__268433377__268433057__268435009 = tile.const {name = "crnsMmod_(268433569, 268433441, 268433377, 268433057)_268435009", q = 268435009 : i64} : tensor<1x1xi32>
    %rs0_etQ0_negM_268435009 = tile.mul_modq %rs0_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_accs_268435009 = tile.sub_modq %rs0_etQ0_acc_268435009, %rs0_etQ0_negM_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ1 = tile.reshape %rs0_etQ0_accs_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %crnsE__268433569__268433441__268433377__268433057__268434977 = tile.const {name = "crnsE_(268433569, 268433441, 268433377, 268433057)_268434977"} : tensor<1x4xi32>
    %rs0_etQ0_xE_268434977 = tile.matmul_modq %crnsE__268433569__268433441__268433377__268433057__268434977, %rs0_etQ0_RES {q = 268434977 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %crnsg__268433569__268433441__268433377__268433057__268434977 = tile.const {name = "crnsg_(268433569, 268433441, 268433377, 268433057)_268434977", q = 268434977 : i64} : tensor<1x1xi32>
    %rs0_etQ0_kg_268434977 = tile.mul_modq %rs0_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_acc_268434977 = tile.add_modq %rs0_etQ0_xE_268434977, %rs0_etQ0_kg_268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsMmod__268433569__268433441__268433377__268433057__268434977 = tile.const {name = "crnsMmod_(268433569, 268433441, 268433377, 268433057)_268434977", q = 268434977 : i64} : tensor<1x1xi32>
    %rs0_etQ0_negM_268434977 = tile.mul_modq %rs0_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_accs_268434977 = tile.sub_modq %rs0_etQ0_acc_268434977, %rs0_etQ0_negM_268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ2 = tile.reshape %rs0_etQ0_accs_268434977 {shape = [16], q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs1_fQ0 = tile.mul_modq %e1_0, %scal_t {q = 268435361 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fQ1 = tile.mul_modq %e1_1, %scal_t {q = 268435009 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fQ2 = tile.mul_modq %e1_2, %scal_t {q = 268434977 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fS0 = tile.mul_modq %e1_3, %scal_t {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fS1 = tile.mul_modq %e1_4, %scal_t {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fS2 = tile.mul_modq %e1_5, %scal_t {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fS3 = tile.mul_modq %e1_6, %scal_t {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_fbc0_o_rs1_fQ0 = tile.reshape %rs1_fQ0 {shape = [1, 16], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_y_rs1_fQ0 = tile.mul_modq %rs1_fbc0_o_rs1_fQ0, %fbcC__268435361__268435009__268434977__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_o_rs1_fQ1 = tile.reshape %rs1_fQ1 {shape = [1, 16], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_y_rs1_fQ1 = tile.mul_modq %rs1_fbc0_o_rs1_fQ1, %fbcC__268435361__268435009__268434977__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_o_rs1_fQ2 = tile.reshape %rs1_fQ2 {shape = [1, 16], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_y_rs1_fQ2 = tile.mul_modq %rs1_fbc0_o_rs1_fQ2, %fbcC__268435361__268435009__268434977__268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_Y = tile.concat %rs1_fbc0_y_rs1_fQ0, %rs1_fbc0_y_rs1_fQ1, %rs1_fbc0_y_rs1_fQ2 {axis = 0 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<3x16xi32>
    %rs1_fbc0_fbc_268433569 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433569, %rs1_fbc0_Y {q = 268433569 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0 = tile.reshape %rs1_fbc0_fbc_268433569 {shape = [16], q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs1_fbc0_fbc_268433441 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433441, %rs1_fbc0_Y {q = 268433441 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc1 = tile.reshape %rs1_fbc0_fbc_268433441 {shape = [16], q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs1_fbc0_fbc_268433377 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433377, %rs1_fbc0_Y {q = 268433377 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc2 = tile.reshape %rs1_fbc0_fbc_268433377 {shape = [16], q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs1_fbc0_fbc_268433057 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433057, %rs1_fbc0_Y {q = 268433057 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc3 = tile.reshape %rs1_fbc0_fbc_268433057 {shape = [16], q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs1_d0 = tile.sub_modq %rs1_fS0, %rs1_fbc0 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs1_etS0 = tile.mul_modq %rs1_d0, %Qinv_268433569 {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_d1 = tile.sub_modq %rs1_fS1, %rs1_fbc1 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs1_etS1 = tile.mul_modq %rs1_d1, %Qinv_268433441 {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_d2 = tile.sub_modq %rs1_fS2, %rs1_fbc2 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs1_etS2 = tile.mul_modq %rs1_d2, %Qinv_268433377 {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_d3 = tile.sub_modq %rs1_fS3, %rs1_fbc3 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs1_etS3 = tile.mul_modq %rs1_d3, %Qinv_268433057 {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs1_etQ0_r_rs1_etS0 = tile.reshape %rs1_etS0 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_r_rs1_etS1 = tile.reshape %rs1_etS1 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_r_rs1_etS2 = tile.reshape %rs1_etS2 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_r_rs1_etS3 = tile.reshape %rs1_etS3 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_RES = tile.concat %rs1_etQ0_r_rs1_etS0, %rs1_etQ0_r_rs1_etS1, %rs1_etQ0_r_rs1_etS2, %rs1_etQ0_r_rs1_etS3 {axis = 0 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<4x16xi32>
    %rs1_etQ0_s = tile.matmul_wide %crnsF__268433569__268433441__268433377__268433057__38, %rs1_etQ0_RES {radix = 128 : i64, nlimbs_a = 6 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_k = tile.fixed_slice %rs1_etQ0_s {shift = 38 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_neg = tile.fixed_slice %rs1_etQ0_s {shift = 37 : i64, mask = 1 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_xE_268435361 = tile.matmul_modq %crnsE__268433569__268433441__268433377__268433057__268435361, %rs1_etQ0_RES {q = 268435361 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_kg_268435361 = tile.mul_modq %rs1_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_acc_268435361 = tile.add_modq %rs1_etQ0_xE_268435361, %rs1_etQ0_kg_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_negM_268435361 = tile.mul_modq %rs1_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_accs_268435361 = tile.sub_modq %rs1_etQ0_acc_268435361, %rs1_etQ0_negM_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0 = tile.reshape %rs1_etQ0_accs_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs1_etQ0_xE_268435009 = tile.matmul_modq %crnsE__268433569__268433441__268433377__268433057__268435009, %rs1_etQ0_RES {q = 268435009 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_kg_268435009 = tile.mul_modq %rs1_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_acc_268435009 = tile.add_modq %rs1_etQ0_xE_268435009, %rs1_etQ0_kg_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_negM_268435009 = tile.mul_modq %rs1_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_accs_268435009 = tile.sub_modq %rs1_etQ0_acc_268435009, %rs1_etQ0_negM_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ1 = tile.reshape %rs1_etQ0_accs_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs1_etQ0_xE_268434977 = tile.matmul_modq %crnsE__268433569__268433441__268433377__268433057__268434977, %rs1_etQ0_RES {q = 268434977 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_kg_268434977 = tile.mul_modq %rs1_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_acc_268434977 = tile.add_modq %rs1_etQ0_xE_268434977, %rs1_etQ0_kg_268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_negM_268434977 = tile.mul_modq %rs1_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_accs_268434977 = tile.sub_modq %rs1_etQ0_acc_268434977, %rs1_etQ0_negM_268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ2 = tile.reshape %rs1_etQ0_accs_268434977 {shape = [16], q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs2_fQ0 = tile.mul_modq %e2_0, %scal_t {q = 268435361 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fQ1 = tile.mul_modq %e2_1, %scal_t {q = 268435009 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fQ2 = tile.mul_modq %e2_2, %scal_t {q = 268434977 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fS0 = tile.mul_modq %e2_3, %scal_t {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fS1 = tile.mul_modq %e2_4, %scal_t {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fS2 = tile.mul_modq %e2_5, %scal_t {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fS3 = tile.mul_modq %e2_6, %scal_t {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_fbc0_o_rs2_fQ0 = tile.reshape %rs2_fQ0 {shape = [1, 16], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_y_rs2_fQ0 = tile.mul_modq %rs2_fbc0_o_rs2_fQ0, %fbcC__268435361__268435009__268434977__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_o_rs2_fQ1 = tile.reshape %rs2_fQ1 {shape = [1, 16], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_y_rs2_fQ1 = tile.mul_modq %rs2_fbc0_o_rs2_fQ1, %fbcC__268435361__268435009__268434977__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_o_rs2_fQ2 = tile.reshape %rs2_fQ2 {shape = [1, 16], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_y_rs2_fQ2 = tile.mul_modq %rs2_fbc0_o_rs2_fQ2, %fbcC__268435361__268435009__268434977__268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_Y = tile.concat %rs2_fbc0_y_rs2_fQ0, %rs2_fbc0_y_rs2_fQ1, %rs2_fbc0_y_rs2_fQ2 {axis = 0 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<3x16xi32>
    %rs2_fbc0_fbc_268433569 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433569, %rs2_fbc0_Y {q = 268433569 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0 = tile.reshape %rs2_fbc0_fbc_268433569 {shape = [16], q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs2_fbc0_fbc_268433441 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433441, %rs2_fbc0_Y {q = 268433441 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc1 = tile.reshape %rs2_fbc0_fbc_268433441 {shape = [16], q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs2_fbc0_fbc_268433377 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433377, %rs2_fbc0_Y {q = 268433377 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc2 = tile.reshape %rs2_fbc0_fbc_268433377 {shape = [16], q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs2_fbc0_fbc_268433057 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433057, %rs2_fbc0_Y {q = 268433057 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc3 = tile.reshape %rs2_fbc0_fbc_268433057 {shape = [16], q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs2_d0 = tile.sub_modq %rs2_fS0, %rs2_fbc0 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs2_etS0 = tile.mul_modq %rs2_d0, %Qinv_268433569 {q = 268433569 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_d1 = tile.sub_modq %rs2_fS1, %rs2_fbc1 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs2_etS1 = tile.mul_modq %rs2_d1, %Qinv_268433441 {q = 268433441 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_d2 = tile.sub_modq %rs2_fS2, %rs2_fbc2 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs2_etS2 = tile.mul_modq %rs2_d2, %Qinv_268433377 {q = 268433377 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_d3 = tile.sub_modq %rs2_fS3, %rs2_fbc3 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rs2_etS3 = tile.mul_modq %rs2_d3, %Qinv_268433057 {q = 268433057 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rs2_etQ0_r_rs2_etS0 = tile.reshape %rs2_etS0 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_r_rs2_etS1 = tile.reshape %rs2_etS1 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_r_rs2_etS2 = tile.reshape %rs2_etS2 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_r_rs2_etS3 = tile.reshape %rs2_etS3 {shape = [1, 16]} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_RES = tile.concat %rs2_etQ0_r_rs2_etS0, %rs2_etQ0_r_rs2_etS1, %rs2_etQ0_r_rs2_etS2, %rs2_etQ0_r_rs2_etS3 {axis = 0 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<4x16xi32>
    %rs2_etQ0_s = tile.matmul_wide %crnsF__268433569__268433441__268433377__268433057__38, %rs2_etQ0_RES {radix = 128 : i64, nlimbs_a = 6 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_k = tile.fixed_slice %rs2_etQ0_s {shift = 38 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_neg = tile.fixed_slice %rs2_etQ0_s {shift = 37 : i64, mask = 1 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_xE_268435361 = tile.matmul_modq %crnsE__268433569__268433441__268433377__268433057__268435361, %rs2_etQ0_RES {q = 268435361 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_kg_268435361 = tile.mul_modq %rs2_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_acc_268435361 = tile.add_modq %rs2_etQ0_xE_268435361, %rs2_etQ0_kg_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_negM_268435361 = tile.mul_modq %rs2_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_accs_268435361 = tile.sub_modq %rs2_etQ0_acc_268435361, %rs2_etQ0_negM_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0 = tile.reshape %rs2_etQ0_accs_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs2_etQ0_xE_268435009 = tile.matmul_modq %crnsE__268433569__268433441__268433377__268433057__268435009, %rs2_etQ0_RES {q = 268435009 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_kg_268435009 = tile.mul_modq %rs2_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_acc_268435009 = tile.add_modq %rs2_etQ0_xE_268435009, %rs2_etQ0_kg_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_negM_268435009 = tile.mul_modq %rs2_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_accs_268435009 = tile.sub_modq %rs2_etQ0_acc_268435009, %rs2_etQ0_negM_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ1 = tile.reshape %rs2_etQ0_accs_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rs2_etQ0_xE_268434977 = tile.matmul_modq %crnsE__268433569__268433441__268433377__268433057__268434977, %rs2_etQ0_RES {q = 268434977 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x4xi32>, tensor<4x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_kg_268434977 = tile.mul_modq %rs2_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_acc_268434977 = tile.add_modq %rs2_etQ0_xE_268434977, %rs2_etQ0_kg_268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_negM_268434977 = tile.mul_modq %rs2_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_accs_268434977 = tile.sub_modq %rs2_etQ0_acc_268434977, %rs2_etQ0_negM_268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ2 = tile.reshape %rs2_etQ0_accs_268434977 {shape = [16], q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rl_c2P0_o_rs2_etQ0 = tile.reshape %rs2_etQ0 {shape = [1, 16], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rl_c2P0_y_rs2_etQ0 = tile.mul_modq %rl_c2P0_o_rs2_etQ0, %fbcC__268435361__268435009__268434977__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rl_c2P0_o_rs2_etQ1 = tile.reshape %rs2_etQ1 {shape = [1, 16], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rl_c2P0_y_rs2_etQ1 = tile.mul_modq %rl_c2P0_o_rs2_etQ1, %fbcC__268435361__268435009__268434977__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rl_c2P0_o_rs2_etQ2 = tile.reshape %rs2_etQ2 {shape = [1, 16], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rl_c2P0_y_rs2_etQ2 = tile.mul_modq %rl_c2P0_o_rs2_etQ2, %fbcC__268435361__268435009__268434977__268434977 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rl_c2P0_Y = tile.concat %rl_c2P0_y_rs2_etQ0, %rl_c2P0_y_rs2_etQ1, %rl_c2P0_y_rs2_etQ2 {axis = 0 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<3x16xi32>
    %fbcD__268435361__268435009__268434977__268434721 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268434721"} : tensor<1x3xi32>
    %rl_c2P0_fbc_268434721 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268434721, %rl_c2P0_Y {q = 268434721 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rl_c2P0 = tile.reshape %rl_c2P0_fbc_268434721 {shape = [16], q = 268434721 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268434401 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268434401"} : tensor<1x3xi32>
    %rl_c2P0_fbc_268434401 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268434401, %rl_c2P0_Y {q = 268434401 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rl_c2P1 = tile.reshape %rl_c2P0_fbc_268434401 {shape = [16], q = 268434401 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433761 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433761"} : tensor<1x3xi32>
    %rl_c2P0_fbc_268433761 = tile.matmul_modq %fbcD__268435361__268435009__268434977__268433761, %rl_c2P0_Y {q = 268433761 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rl_c2P2 = tile.reshape %rl_c2P0_fbc_268433761 {shape = [16], q = 268433761 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rl_X0 = tile.matmul_modq %W_f_268435361, %rs2_etQ0 {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_X1 = tile.matmul_modq %W_f_268435009, %rs2_etQ1 {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_X2 = tile.matmul_modq %W_f_268434977, %rs2_etQ2 {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268434721 = tile.const {name = "W_f_268434721", q = 268434721 : i64} : tensor<16x16xi32>
    %rl_X3 = tile.matmul_modq %W_f_268434721, %rl_c2P0 {q = 268434721 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268434401 = tile.const {name = "W_f_268434401", q = 268434401 : i64} : tensor<16x16xi32>
    %rl_X4 = tile.matmul_modq %W_f_268434401, %rl_c2P1 {q = 268434401 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268433761 = tile.const {name = "W_f_268433761", q = 268433761 : i64} : tensor<16x16xi32>
    %rl_X5 = tile.matmul_modq %W_f_268433761, %rl_c2P2 {q = 268433761 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_0c = rns.const {name = "rl_evkA_0c", q = 268435361 : i64} : tensor<16xi32>
    %rl_evkA_0 = tile.matmul_modq %W_f_268435361, %rl_evkA_0c {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY0 = tile.mul_modq %rl_X0, %rl_evkA_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa0 = tile.matmul_modq %W_i_268435361, %rl_AY0 {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_1c = rns.const {name = "rl_evkA_1c", q = 268435009 : i64} : tensor<16xi32>
    %rl_evkA_1 = tile.matmul_modq %W_f_268435009, %rl_evkA_1c {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY1 = tile.mul_modq %rl_X1, %rl_evkA_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa1 = tile.matmul_modq %W_i_268435009, %rl_AY1 {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_2c = rns.const {name = "rl_evkA_2c", q = 268434977 : i64} : tensor<16xi32>
    %rl_evkA_2 = tile.matmul_modq %W_f_268434977, %rl_evkA_2c {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY2 = tile.mul_modq %rl_X2, %rl_evkA_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa2 = tile.matmul_modq %W_i_268434977, %rl_AY2 {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_3c = rns.const {name = "rl_evkA_3c", q = 268434721 : i64} : tensor<16xi32>
    %rl_evkA_3 = tile.matmul_modq %W_f_268434721, %rl_evkA_3c {q = 268434721 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY3 = tile.mul_modq %rl_X3, %rl_evkA_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268434721 = tile.const {name = "W_i_268434721", q = 268434721 : i64} : tensor<16x16xi32>
    %rl_Aa3 = tile.matmul_modq %W_i_268434721, %rl_AY3 {q = 268434721 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_4c = rns.const {name = "rl_evkA_4c", q = 268434401 : i64} : tensor<16xi32>
    %rl_evkA_4 = tile.matmul_modq %W_f_268434401, %rl_evkA_4c {q = 268434401 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY4 = tile.mul_modq %rl_X4, %rl_evkA_4 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268434401 = tile.const {name = "W_i_268434401", q = 268434401 : i64} : tensor<16x16xi32>
    %rl_Aa4 = tile.matmul_modq %W_i_268434401, %rl_AY4 {q = 268434401 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_5c = rns.const {name = "rl_evkA_5c", q = 268433761 : i64} : tensor<16xi32>
    %rl_evkA_5 = tile.matmul_modq %W_f_268433761, %rl_evkA_5c {q = 268433761 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY5 = tile.mul_modq %rl_X5, %rl_evkA_5 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433761 = tile.const {name = "W_i_268433761", q = 268433761 : i64} : tensor<16x16xi32>
    %rl_Aa5 = tile.matmul_modq %W_i_268433761, %rl_AY5 {q = 268433761 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_0c = rns.const {name = "rl_evkB_0c", q = 268435361 : i64} : tensor<16xi32>
    %rl_evkB_0 = tile.matmul_modq %W_f_268435361, %rl_evkB_0c {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY0 = tile.mul_modq %rl_X0, %rl_evkB_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba0 = tile.matmul_modq %W_i_268435361, %rl_BY0 {q = 268435361 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_1c = rns.const {name = "rl_evkB_1c", q = 268435009 : i64} : tensor<16xi32>
    %rl_evkB_1 = tile.matmul_modq %W_f_268435009, %rl_evkB_1c {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY1 = tile.mul_modq %rl_X1, %rl_evkB_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba1 = tile.matmul_modq %W_i_268435009, %rl_BY1 {q = 268435009 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_2c = rns.const {name = "rl_evkB_2c", q = 268434977 : i64} : tensor<16xi32>
    %rl_evkB_2 = tile.matmul_modq %W_f_268434977, %rl_evkB_2c {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY2 = tile.mul_modq %rl_X2, %rl_evkB_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba2 = tile.matmul_modq %W_i_268434977, %rl_BY2 {q = 268434977 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_3c = rns.const {name = "rl_evkB_3c", q = 268434721 : i64} : tensor<16xi32>
    %rl_evkB_3 = tile.matmul_modq %W_f_268434721, %rl_evkB_3c {q = 268434721 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY3 = tile.mul_modq %rl_X3, %rl_evkB_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba3 = tile.matmul_modq %W_i_268434721, %rl_BY3 {q = 268434721 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_4c = rns.const {name = "rl_evkB_4c", q = 268434401 : i64} : tensor<16xi32>
    %rl_evkB_4 = tile.matmul_modq %W_f_268434401, %rl_evkB_4c {q = 268434401 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY4 = tile.mul_modq %rl_X4, %rl_evkB_4 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba4 = tile.matmul_modq %W_i_268434401, %rl_BY4 {q = 268434401 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_5c = rns.const {name = "rl_evkB_5c", q = 268433761 : i64} : tensor<16xi32>
    %rl_evkB_5 = tile.matmul_modq %W_f_268433761, %rl_evkB_5c {q = 268433761 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY5 = tile.mul_modq %rl_X5, %rl_evkB_5 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba5 = tile.matmul_modq %W_i_268433761, %rl_BY5 {q = 268433761 : i64} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %fbcC__268434721__268434401__268433761__268434721 = tile.const {name = "fbcC_(268434721, 268434401, 268433761)_268434721", q = 268434721 : i64} : tensor<1x1xi32>
    %rl_AxP0_o_rl_Aa3 = tile.reshape %rl_Aa3 {shape = [1, 16], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rl_AxP0_y_rl_Aa3 = tile.mul_modq %rl_AxP0_o_rl_Aa3, %fbcC__268434721__268434401__268433761__268434721 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %fbcC__268434721__268434401__268433761__268434401 = tile.const {name = "fbcC_(268434721, 268434401, 268433761)_268434401", q = 268434401 : i64} : tensor<1x1xi32>
    %rl_AxP0_o_rl_Aa4 = tile.reshape %rl_Aa4 {shape = [1, 16], q = 268434401 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rl_AxP0_y_rl_Aa4 = tile.mul_modq %rl_AxP0_o_rl_Aa4, %fbcC__268434721__268434401__268433761__268434401 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %fbcC__268434721__268434401__268433761__268433761 = tile.const {name = "fbcC_(268434721, 268434401, 268433761)_268433761", q = 268433761 : i64} : tensor<1x1xi32>
    %rl_AxP0_o_rl_Aa5 = tile.reshape %rl_Aa5 {shape = [1, 16], q = 268433761 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rl_AxP0_y_rl_Aa5 = tile.mul_modq %rl_AxP0_o_rl_Aa5, %fbcC__268434721__268434401__268433761__268433761 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rl_AxP0_Y = tile.concat %rl_AxP0_y_rl_Aa3, %rl_AxP0_y_rl_Aa4, %rl_AxP0_y_rl_Aa5 {axis = 0 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<3x16xi32>
    %fbcD__268434721__268434401__268433761__268435361 = tile.const {name = "fbcD_(268434721, 268434401, 268433761)_268435361"} : tensor<1x3xi32>
    %rl_AxP0_fbc_268435361 = tile.matmul_modq %fbcD__268434721__268434401__268433761__268435361, %rl_AxP0_Y {q = 268435361 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rl_AxP0 = tile.reshape %rl_AxP0_fbc_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268434721__268434401__268433761__268435009 = tile.const {name = "fbcD_(268434721, 268434401, 268433761)_268435009"} : tensor<1x3xi32>
    %rl_AxP0_fbc_268435009 = tile.matmul_modq %fbcD__268434721__268434401__268433761__268435009, %rl_AxP0_Y {q = 268435009 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rl_AxP1 = tile.reshape %rl_AxP0_fbc_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268434721__268434401__268433761__268434977 = tile.const {name = "fbcD_(268434721, 268434401, 268433761)_268434977"} : tensor<1x3xi32>
    %rl_AxP0_fbc_268434977 = tile.matmul_modq %fbcD__268434721__268434401__268433761__268434977, %rl_AxP0_Y {q = 268434977 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rl_AxP2 = tile.reshape %rl_AxP0_fbc_268434977 {shape = [16], q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rl_Ad0 = tile.sub_modq %rl_Aa0, %rl_AxP0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Pinv_268435361 = rns.const {name = "Pinv_268435361"} : tensor<1xi32>
    %rl_Ao0 = tile.mul_modq %rl_Ad0, %Pinv_268435361 {q = 268435361 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rl_Ad1 = tile.sub_modq %rl_Aa1, %rl_AxP1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Pinv_268435009 = rns.const {name = "Pinv_268435009"} : tensor<1xi32>
    %rl_Ao1 = tile.mul_modq %rl_Ad1, %Pinv_268435009 {q = 268435009 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rl_Ad2 = tile.sub_modq %rl_Aa2, %rl_AxP2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %Pinv_268434977 = rns.const {name = "Pinv_268434977"} : tensor<1xi32>
    %rl_Ao2 = tile.mul_modq %rl_Ad2, %Pinv_268434977 {q = 268434977 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rl_BxP0_o_rl_Ba3 = tile.reshape %rl_Ba3 {shape = [1, 16], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rl_BxP0_y_rl_Ba3 = tile.mul_modq %rl_BxP0_o_rl_Ba3, %fbcC__268434721__268434401__268433761__268434721 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rl_BxP0_o_rl_Ba4 = tile.reshape %rl_Ba4 {shape = [1, 16], q = 268434401 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rl_BxP0_y_rl_Ba4 = tile.mul_modq %rl_BxP0_o_rl_Ba4, %fbcC__268434721__268434401__268433761__268434401 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rl_BxP0_o_rl_Ba5 = tile.reshape %rl_Ba5 {shape = [1, 16], q = 268433761 : i64} : (tensor<16xi32>) -> tensor<1x16xi32>
    %rl_BxP0_y_rl_Ba5 = tile.mul_modq %rl_BxP0_o_rl_Ba5, %fbcC__268434721__268434401__268433761__268433761 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rl_BxP0_Y = tile.concat %rl_BxP0_y_rl_Ba3, %rl_BxP0_y_rl_Ba4, %rl_BxP0_y_rl_Ba5 {axis = 0 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<3x16xi32>
    %rl_BxP0_fbc_268435361 = tile.matmul_modq %fbcD__268434721__268434401__268433761__268435361, %rl_BxP0_Y {q = 268435361 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rl_BxP0 = tile.reshape %rl_BxP0_fbc_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rl_BxP0_fbc_268435009 = tile.matmul_modq %fbcD__268434721__268434401__268433761__268435009, %rl_BxP0_Y {q = 268435009 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rl_BxP1 = tile.reshape %rl_BxP0_fbc_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rl_BxP0_fbc_268434977 = tile.matmul_modq %fbcD__268434721__268434401__268433761__268434977, %rl_BxP0_Y {q = 268434977 : i64, radix = 128 : i64, nlimbs_a = 4 : i64, nlimbs_b = 4 : i64} : (tensor<1x3xi32>, tensor<3x16xi32>) -> tensor<1x16xi32>
    %rl_BxP2 = tile.reshape %rl_BxP0_fbc_268434977 {shape = [16], q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %rl_Bd0 = tile.sub_modq %rl_Ba0, %rl_BxP0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Bo0 = tile.mul_modq %rl_Bd0, %Pinv_268435361 {q = 268435361 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rl_Bd1 = tile.sub_modq %rl_Ba1, %rl_BxP1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Bo1 = tile.mul_modq %rl_Bd1, %Pinv_268435009 {q = 268435009 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %rl_Bd2 = tile.sub_modq %rl_Ba2, %rl_BxP2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_Bo2 = tile.mul_modq %rl_Bd2, %Pinv_268434977 {q = 268434977 : i64} : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %out0_0 = tile.add_modq %rs0_etQ0, %rl_Ao0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %out1_0 = tile.add_modq %rs1_etQ0, %rl_Bo0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %out0_1 = tile.add_modq %rs0_etQ1, %rl_Ao1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %out1_1 = tile.add_modq %rs1_etQ1, %rl_Bo1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %out0_2 = tile.add_modq %rs0_etQ2, %rl_Ao2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %out1_2 = tile.add_modq %rs1_etQ2, %rl_Bo2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    return %out0_0, %out0_1, %out0_2, %out1_0, %out1_1, %out1_2 : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }
}
