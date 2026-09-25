module {
  func.func @ntt_int8(%x: tensor<2048xi32>) -> tensor<2048xi32> {
    %0 = ntt.reshape %x : tensor<2048xi32> to tensor<32x64xi32>
    %1 = ntt.const {name = "W0_T"} : tensor<32x32xi32>
    %0__t1, %0__t2 = ntt.split %0 {axis = 1 : index, parts = 2 : index} : tensor<32x64xi32> -> (tensor<32x32xi32>, tensor<32x32xi32>)
    %d0, %d1, %d2, %d3 = ntt.decompose %0__t1 {radix = 128 : index, nlimbs = 4 : index} : tensor<32x32xi32> -> (tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>)
    %d4 = ntt.const {name = "W0_T__i8_p0_r0"} : tensor<32x32xi8>
    %d5 = ntt.matmul_modq %d0, %d4 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d6 = ntt.const {name = "W0_T__i8_p1_r0"} : tensor<32x32xi8>
    %d7 = ntt.matmul_modq %d1, %d6 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d8 = ntt.const {name = "W0_T__i8_p2_r0"} : tensor<32x32xi8>
    %d9 = ntt.matmul_modq %d2, %d8 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d10 = ntt.const {name = "W0_T__i8_p3_r0"} : tensor<32x32xi8>
    %d11 = ntt.matmul_modq %d3, %d10 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d12 = ntt.add_modq %d5, %d7, %d9, %d11 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d13 = ntt.const {name = "W0_T__i8_p0_r1"} : tensor<32x32xi8>
    %d14 = ntt.matmul_modq %d0, %d13 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d15 = ntt.const {name = "W0_T__i8_p1_r1"} : tensor<32x32xi8>
    %d16 = ntt.matmul_modq %d1, %d15 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d17 = ntt.const {name = "W0_T__i8_p2_r1"} : tensor<32x32xi8>
    %d18 = ntt.matmul_modq %d2, %d17 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d19 = ntt.const {name = "W0_T__i8_p3_r1"} : tensor<32x32xi8>
    %d20 = ntt.matmul_modq %d3, %d19 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d21 = ntt.add_modq %d14, %d16, %d18, %d20 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d22 = ntt.const {name = "W0_T__i8_p0_r2"} : tensor<32x32xi8>
    %d23 = ntt.matmul_modq %d0, %d22 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d24 = ntt.const {name = "W0_T__i8_p1_r2"} : tensor<32x32xi8>
    %d25 = ntt.matmul_modq %d1, %d24 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d26 = ntt.const {name = "W0_T__i8_p2_r2"} : tensor<32x32xi8>
    %d27 = ntt.matmul_modq %d2, %d26 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d28 = ntt.const {name = "W0_T__i8_p3_r2"} : tensor<32x32xi8>
    %d29 = ntt.matmul_modq %d3, %d28 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d30 = ntt.add_modq %d23, %d25, %d27, %d29 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d31 = ntt.const {name = "W0_T__i8_p0_r3"} : tensor<32x32xi8>
    %d32 = ntt.matmul_modq %d0, %d31 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d33 = ntt.const {name = "W0_T__i8_p1_r3"} : tensor<32x32xi8>
    %d34 = ntt.matmul_modq %d1, %d33 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d35 = ntt.const {name = "W0_T__i8_p2_r3"} : tensor<32x32xi8>
    %d36 = ntt.matmul_modq %d2, %d35 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d37 = ntt.const {name = "W0_T__i8_p3_r3"} : tensor<32x32xi8>
    %d38 = ntt.matmul_modq %d3, %d37 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d39 = ntt.add_modq %d32, %d34, %d36, %d38 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %2__t3 = ntt.recombine_modq %d12, %d21, %d30, %d39 {radix = 128 : index, q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %3__t4 = ntt.const {name = "T0_T__r0"} : tensor<32x32xi32>
    %2__t5 = ntt.mul_modq %2__t3, %3__t4 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d40, %d41, %d42, %d43 = ntt.decompose %0__t2 {radix = 128 : index, nlimbs = 4 : index} : tensor<32x32xi32> -> (tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>)
    %d44 = ntt.const {name = "W0_T__i8_p0_r0"} : tensor<32x32xi8>
    %d45 = ntt.matmul_modq %d40, %d44 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d46 = ntt.const {name = "W0_T__i8_p1_r0"} : tensor<32x32xi8>
    %d47 = ntt.matmul_modq %d41, %d46 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d48 = ntt.const {name = "W0_T__i8_p2_r0"} : tensor<32x32xi8>
    %d49 = ntt.matmul_modq %d42, %d48 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d50 = ntt.const {name = "W0_T__i8_p3_r0"} : tensor<32x32xi8>
    %d51 = ntt.matmul_modq %d43, %d50 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d52 = ntt.add_modq %d45, %d47, %d49, %d51 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d53 = ntt.const {name = "W0_T__i8_p0_r1"} : tensor<32x32xi8>
    %d54 = ntt.matmul_modq %d40, %d53 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d55 = ntt.const {name = "W0_T__i8_p1_r1"} : tensor<32x32xi8>
    %d56 = ntt.matmul_modq %d41, %d55 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d57 = ntt.const {name = "W0_T__i8_p2_r1"} : tensor<32x32xi8>
    %d58 = ntt.matmul_modq %d42, %d57 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d59 = ntt.const {name = "W0_T__i8_p3_r1"} : tensor<32x32xi8>
    %d60 = ntt.matmul_modq %d43, %d59 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d61 = ntt.add_modq %d54, %d56, %d58, %d60 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d62 = ntt.const {name = "W0_T__i8_p0_r2"} : tensor<32x32xi8>
    %d63 = ntt.matmul_modq %d40, %d62 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d64 = ntt.const {name = "W0_T__i8_p1_r2"} : tensor<32x32xi8>
    %d65 = ntt.matmul_modq %d41, %d64 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d66 = ntt.const {name = "W0_T__i8_p2_r2"} : tensor<32x32xi8>
    %d67 = ntt.matmul_modq %d42, %d66 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d68 = ntt.const {name = "W0_T__i8_p3_r2"} : tensor<32x32xi8>
    %d69 = ntt.matmul_modq %d43, %d68 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d70 = ntt.add_modq %d63, %d65, %d67, %d69 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d71 = ntt.const {name = "W0_T__i8_p0_r3"} : tensor<32x32xi8>
    %d72 = ntt.matmul_modq %d40, %d71 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d73 = ntt.const {name = "W0_T__i8_p1_r3"} : tensor<32x32xi8>
    %d74 = ntt.matmul_modq %d41, %d73 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d75 = ntt.const {name = "W0_T__i8_p2_r3"} : tensor<32x32xi8>
    %d76 = ntt.matmul_modq %d42, %d75 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d77 = ntt.const {name = "W0_T__i8_p3_r3"} : tensor<32x32xi8>
    %d78 = ntt.matmul_modq %d43, %d77 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d79 = ntt.add_modq %d72, %d74, %d76, %d78 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %2__t6 = ntt.recombine_modq %d52, %d61, %d70, %d79 {radix = 128 : index, q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %3__t7 = ntt.const {name = "T0_T__r1"} : tensor<32x32xi32>
    %2__t8 = ntt.mul_modq %2__t6, %3__t7 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m0, %m3 = ntt.all_to_all_tiles %2__t5, %2__t8 {perm = [0, 1]} : (tensor<32x32xi32>, tensor<32x32xi32>) -> (tensor<32x32xi32>, tensor<32x32xi32>)
    %7 = ntt.const {name = "W1_f_T"} : tensor<64x64xi32>
    %m1 = ntt.const {name = "W1_f_T__0.0_1.0"} : tensor<32x32xi32>
    %d80, %d81, %d82, %d83 = ntt.decompose %m0 {radix = 128 : index, nlimbs = 4 : index} : tensor<32x32xi32> -> (tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>)
    %d84 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p0_r0"} : tensor<32x32xi8>
    %d85 = ntt.matmul_modq %d80, %d84 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d86 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p1_r0"} : tensor<32x32xi8>
    %d87 = ntt.matmul_modq %d81, %d86 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d88 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p2_r0"} : tensor<32x32xi8>
    %d89 = ntt.matmul_modq %d82, %d88 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d90 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p3_r0"} : tensor<32x32xi8>
    %d91 = ntt.matmul_modq %d83, %d90 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d92 = ntt.add_modq %d85, %d87, %d89, %d91 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d93 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p0_r1"} : tensor<32x32xi8>
    %d94 = ntt.matmul_modq %d80, %d93 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d95 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p1_r1"} : tensor<32x32xi8>
    %d96 = ntt.matmul_modq %d81, %d95 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d97 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p2_r1"} : tensor<32x32xi8>
    %d98 = ntt.matmul_modq %d82, %d97 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d99 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p3_r1"} : tensor<32x32xi8>
    %d100 = ntt.matmul_modq %d83, %d99 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d101 = ntt.add_modq %d94, %d96, %d98, %d100 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d102 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p0_r2"} : tensor<32x32xi8>
    %d103 = ntt.matmul_modq %d80, %d102 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d104 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p1_r2"} : tensor<32x32xi8>
    %d105 = ntt.matmul_modq %d81, %d104 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d106 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p2_r2"} : tensor<32x32xi8>
    %d107 = ntt.matmul_modq %d82, %d106 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d108 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p3_r2"} : tensor<32x32xi8>
    %d109 = ntt.matmul_modq %d83, %d108 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d110 = ntt.add_modq %d103, %d105, %d107, %d109 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d111 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p0_r3"} : tensor<32x32xi8>
    %d112 = ntt.matmul_modq %d80, %d111 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d113 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p1_r3"} : tensor<32x32xi8>
    %d114 = ntt.matmul_modq %d81, %d113 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d115 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p2_r3"} : tensor<32x32xi8>
    %d116 = ntt.matmul_modq %d82, %d115 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d117 = ntt.const {name = "W1_f_T__0.0_1.0__i8_p3_r3"} : tensor<32x32xi8>
    %d118 = ntt.matmul_modq %d83, %d117 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d119 = ntt.add_modq %d112, %d114, %d116, %d118 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m2 = ntt.recombine_modq %d92, %d101, %d110, %d119 {radix = 128 : index, q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m4 = ntt.const {name = "W1_f_T__0.32_1.0"} : tensor<32x32xi32>
    %d120, %d121, %d122, %d123 = ntt.decompose %m3 {radix = 128 : index, nlimbs = 4 : index} : tensor<32x32xi32> -> (tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>)
    %d124 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p0_r0"} : tensor<32x32xi8>
    %d125 = ntt.matmul_modq %d120, %d124 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d126 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p1_r0"} : tensor<32x32xi8>
    %d127 = ntt.matmul_modq %d121, %d126 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d128 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p2_r0"} : tensor<32x32xi8>
    %d129 = ntt.matmul_modq %d122, %d128 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d130 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p3_r0"} : tensor<32x32xi8>
    %d131 = ntt.matmul_modq %d123, %d130 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d132 = ntt.add_modq %d125, %d127, %d129, %d131 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d133 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p0_r1"} : tensor<32x32xi8>
    %d134 = ntt.matmul_modq %d120, %d133 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d135 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p1_r1"} : tensor<32x32xi8>
    %d136 = ntt.matmul_modq %d121, %d135 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d137 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p2_r1"} : tensor<32x32xi8>
    %d138 = ntt.matmul_modq %d122, %d137 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d139 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p3_r1"} : tensor<32x32xi8>
    %d140 = ntt.matmul_modq %d123, %d139 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d141 = ntt.add_modq %d134, %d136, %d138, %d140 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d142 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p0_r2"} : tensor<32x32xi8>
    %d143 = ntt.matmul_modq %d120, %d142 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d144 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p1_r2"} : tensor<32x32xi8>
    %d145 = ntt.matmul_modq %d121, %d144 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d146 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p2_r2"} : tensor<32x32xi8>
    %d147 = ntt.matmul_modq %d122, %d146 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d148 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p3_r2"} : tensor<32x32xi8>
    %d149 = ntt.matmul_modq %d123, %d148 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d150 = ntt.add_modq %d143, %d145, %d147, %d149 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d151 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p0_r3"} : tensor<32x32xi8>
    %d152 = ntt.matmul_modq %d120, %d151 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d153 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p1_r3"} : tensor<32x32xi8>
    %d154 = ntt.matmul_modq %d121, %d153 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d155 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p2_r3"} : tensor<32x32xi8>
    %d156 = ntt.matmul_modq %d122, %d155 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d157 = ntt.const {name = "W1_f_T__0.32_1.0__i8_p3_r3"} : tensor<32x32xi8>
    %d158 = ntt.matmul_modq %d123, %d157 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d159 = ntt.add_modq %d152, %d154, %d156, %d158 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m5 = ntt.recombine_modq %d132, %d141, %d150, %d159 {radix = 128 : index, q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m6 = ntt.add_modq %m2, %m5 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m7 = ntt.const {name = "W1_f_T__0.0_1.32"} : tensor<32x32xi32>
    %d160, %d161, %d162, %d163 = ntt.decompose %m0 {radix = 128 : index, nlimbs = 4 : index} : tensor<32x32xi32> -> (tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>)
    %d164 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p0_r0"} : tensor<32x32xi8>
    %d165 = ntt.matmul_modq %d160, %d164 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d166 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p1_r0"} : tensor<32x32xi8>
    %d167 = ntt.matmul_modq %d161, %d166 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d168 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p2_r0"} : tensor<32x32xi8>
    %d169 = ntt.matmul_modq %d162, %d168 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d170 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p3_r0"} : tensor<32x32xi8>
    %d171 = ntt.matmul_modq %d163, %d170 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d172 = ntt.add_modq %d165, %d167, %d169, %d171 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d173 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p0_r1"} : tensor<32x32xi8>
    %d174 = ntt.matmul_modq %d160, %d173 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d175 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p1_r1"} : tensor<32x32xi8>
    %d176 = ntt.matmul_modq %d161, %d175 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d177 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p2_r1"} : tensor<32x32xi8>
    %d178 = ntt.matmul_modq %d162, %d177 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d179 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p3_r1"} : tensor<32x32xi8>
    %d180 = ntt.matmul_modq %d163, %d179 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d181 = ntt.add_modq %d174, %d176, %d178, %d180 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d182 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p0_r2"} : tensor<32x32xi8>
    %d183 = ntt.matmul_modq %d160, %d182 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d184 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p1_r2"} : tensor<32x32xi8>
    %d185 = ntt.matmul_modq %d161, %d184 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d186 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p2_r2"} : tensor<32x32xi8>
    %d187 = ntt.matmul_modq %d162, %d186 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d188 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p3_r2"} : tensor<32x32xi8>
    %d189 = ntt.matmul_modq %d163, %d188 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d190 = ntt.add_modq %d183, %d185, %d187, %d189 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d191 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p0_r3"} : tensor<32x32xi8>
    %d192 = ntt.matmul_modq %d160, %d191 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d193 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p1_r3"} : tensor<32x32xi8>
    %d194 = ntt.matmul_modq %d161, %d193 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d195 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p2_r3"} : tensor<32x32xi8>
    %d196 = ntt.matmul_modq %d162, %d195 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d197 = ntt.const {name = "W1_f_T__0.0_1.32__i8_p3_r3"} : tensor<32x32xi8>
    %d198 = ntt.matmul_modq %d163, %d197 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d199 = ntt.add_modq %d192, %d194, %d196, %d198 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m8 = ntt.recombine_modq %d172, %d181, %d190, %d199 {radix = 128 : index, q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m9 = ntt.const {name = "W1_f_T__0.32_1.32"} : tensor<32x32xi32>
    %d200, %d201, %d202, %d203 = ntt.decompose %m3 {radix = 128 : index, nlimbs = 4 : index} : tensor<32x32xi32> -> (tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>)
    %d204 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p0_r0"} : tensor<32x32xi8>
    %d205 = ntt.matmul_modq %d200, %d204 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d206 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p1_r0"} : tensor<32x32xi8>
    %d207 = ntt.matmul_modq %d201, %d206 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d208 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p2_r0"} : tensor<32x32xi8>
    %d209 = ntt.matmul_modq %d202, %d208 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d210 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p3_r0"} : tensor<32x32xi8>
    %d211 = ntt.matmul_modq %d203, %d210 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d212 = ntt.add_modq %d205, %d207, %d209, %d211 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d213 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p0_r1"} : tensor<32x32xi8>
    %d214 = ntt.matmul_modq %d200, %d213 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d215 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p1_r1"} : tensor<32x32xi8>
    %d216 = ntt.matmul_modq %d201, %d215 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d217 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p2_r1"} : tensor<32x32xi8>
    %d218 = ntt.matmul_modq %d202, %d217 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d219 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p3_r1"} : tensor<32x32xi8>
    %d220 = ntt.matmul_modq %d203, %d219 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d221 = ntt.add_modq %d214, %d216, %d218, %d220 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d222 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p0_r2"} : tensor<32x32xi8>
    %d223 = ntt.matmul_modq %d200, %d222 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d224 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p1_r2"} : tensor<32x32xi8>
    %d225 = ntt.matmul_modq %d201, %d224 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d226 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p2_r2"} : tensor<32x32xi8>
    %d227 = ntt.matmul_modq %d202, %d226 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d228 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p3_r2"} : tensor<32x32xi8>
    %d229 = ntt.matmul_modq %d203, %d228 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d230 = ntt.add_modq %d223, %d225, %d227, %d229 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %d231 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p0_r3"} : tensor<32x32xi8>
    %d232 = ntt.matmul_modq %d200, %d231 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d233 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p1_r3"} : tensor<32x32xi8>
    %d234 = ntt.matmul_modq %d201, %d233 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d235 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p2_r3"} : tensor<32x32xi8>
    %d236 = ntt.matmul_modq %d202, %d235 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d237 = ntt.const {name = "W1_f_T__0.32_1.32__i8_p3_r3"} : tensor<32x32xi8>
    %d238 = ntt.matmul_modq %d203, %d237 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi8>, tensor<32x32xi8>) -> tensor<32x32xi32>
    %d239 = ntt.add_modq %d232, %d234, %d236, %d238 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m10 = ntt.recombine_modq %d212, %d221, %d230, %d239 {radix = 128 : index, q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m11 = ntt.add_modq %m8, %m10 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %8 = ntt.concat %m6, %m11 {axis = 1 : index} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x64xi32>
    %12 = ntt.reshape %8 : tensor<32x64xi32> to tensor<2048xi32>
    return %12 : tensor<2048xi32>
  }
}