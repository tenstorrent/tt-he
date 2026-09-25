module {
  func.func @matmul_modq_i8(%x: tensor<8x8xi32>) -> tensor<8x8xi32> {
    %w = ntt.const {name = "W"} : tensor<8x8xi32>
    %d0, %d1, %d2, %d3 = ntt.decompose %x {radix = 128 : index, nlimbs = 4 : index} : tensor<8x8xi32> -> (tensor<8x8xi8>, tensor<8x8xi8>, tensor<8x8xi8>, tensor<8x8xi8>)
    %d4 = ntt.const {name = "W__i8_r0_p0"} : tensor<8x8xi8>
    %d5 = ntt.matmul_modq %d4, %d0 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d6 = ntt.const {name = "W__i8_r1_p0"} : tensor<8x8xi8>
    %d7 = ntt.matmul_modq %d6, %d1 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d8 = ntt.const {name = "W__i8_r2_p0"} : tensor<8x8xi8>
    %d9 = ntt.matmul_modq %d8, %d2 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d10 = ntt.const {name = "W__i8_r3_p0"} : tensor<8x8xi8>
    %d11 = ntt.matmul_modq %d10, %d3 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d12 = ntt.add_modq %d5, %d7, %d9, %d11 {q = 167772161 : i32} : (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
    %d13 = ntt.const {name = "W__i8_r0_p1"} : tensor<8x8xi8>
    %d14 = ntt.matmul_modq %d13, %d0 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d15 = ntt.const {name = "W__i8_r1_p1"} : tensor<8x8xi8>
    %d16 = ntt.matmul_modq %d15, %d1 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d17 = ntt.const {name = "W__i8_r2_p1"} : tensor<8x8xi8>
    %d18 = ntt.matmul_modq %d17, %d2 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d19 = ntt.const {name = "W__i8_r3_p1"} : tensor<8x8xi8>
    %d20 = ntt.matmul_modq %d19, %d3 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d21 = ntt.add_modq %d14, %d16, %d18, %d20 {q = 167772161 : i32} : (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
    %d22 = ntt.const {name = "W__i8_r0_p2"} : tensor<8x8xi8>
    %d23 = ntt.matmul_modq %d22, %d0 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d24 = ntt.const {name = "W__i8_r1_p2"} : tensor<8x8xi8>
    %d25 = ntt.matmul_modq %d24, %d1 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d26 = ntt.const {name = "W__i8_r2_p2"} : tensor<8x8xi8>
    %d27 = ntt.matmul_modq %d26, %d2 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d28 = ntt.const {name = "W__i8_r3_p2"} : tensor<8x8xi8>
    %d29 = ntt.matmul_modq %d28, %d3 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d30 = ntt.add_modq %d23, %d25, %d27, %d29 {q = 167772161 : i32} : (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
    %d31 = ntt.const {name = "W__i8_r0_p3"} : tensor<8x8xi8>
    %d32 = ntt.matmul_modq %d31, %d0 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d33 = ntt.const {name = "W__i8_r1_p3"} : tensor<8x8xi8>
    %d34 = ntt.matmul_modq %d33, %d1 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d35 = ntt.const {name = "W__i8_r2_p3"} : tensor<8x8xi8>
    %d36 = ntt.matmul_modq %d35, %d2 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d37 = ntt.const {name = "W__i8_r3_p3"} : tensor<8x8xi8>
    %d38 = ntt.matmul_modq %d37, %d3 {q = 167772161 : i32} : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi32>
    %d39 = ntt.add_modq %d32, %d34, %d36, %d38 {q = 167772161 : i32} : (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
    %o = ntt.recombine_modq %d12, %d21, %d30, %d39 {radix = 128 : index, q = 167772161 : i32} : (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
    return %o : tensor<8x8xi32>
  }
}