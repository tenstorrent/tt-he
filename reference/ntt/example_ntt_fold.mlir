module {
  func.func @ntt_fold(%x: tensor<4096xi32>) -> tensor<4096xi32> {
    %0 = ntt.reshape %x : tensor<4096xi32> to tensor<8x512xi32>
    %1 = ntt.const {name = "W0_T"} : tensor<8x8xi32>
    %2 = ntt.matmul_modq %0, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x512xi32>, tensor<8x8xi32>) -> tensor<512x8xi32>
    %3 = ntt.const {name = "T0_T"} : tensor<512x8xi32>
    %4 = ntt.mul_modq %2, %3 {q = 167772161 : i32} : (tensor<512x8xi32>, tensor<512x8xi32>) -> tensor<512x8xi32>
    %6 = ntt.reshape %4 : tensor<512x8xi32> to tensor<8x512xi32>
    %7 = ntt.const {name = "W1_T"} : tensor<8x8xi32>
    %8 = ntt.matmul_modq %6, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x512xi32>, tensor<8x8xi32>) -> tensor<512x8xi32>
    %9 = ntt.const {name = "T1_T"} : tensor<512x8xi32>
    %10 = ntt.mul_modq %8, %9 {q = 167772161 : i32} : (tensor<512x8xi32>, tensor<512x8xi32>) -> tensor<512x8xi32>
    %12 = ntt.reshape %10 : tensor<512x8xi32> to tensor<8x512xi32>
    %13 = ntt.const {name = "W2_T"} : tensor<8x8xi32>
    %14 = ntt.matmul_modq %12, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x512xi32>, tensor<8x8xi32>) -> tensor<512x8xi32>
    %15 = ntt.const {name = "T2_T"} : tensor<512x8xi32>
    %16 = ntt.mul_modq %14, %15 {q = 167772161 : i32} : (tensor<512x8xi32>, tensor<512x8xi32>) -> tensor<512x8xi32>
    %18 = ntt.reshape %16 : tensor<512x8xi32> to tensor<8x512xi32>
    %19 = ntt.const {name = "W3_f_T"} : tensor<8x8xi32>
    %20 = ntt.matmul_modq %18, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x512xi32>, tensor<8x8xi32>) -> tensor<512x8xi32>
    %24 = ntt.reshape %20 : tensor<512x8xi32> to tensor<4096xi32>
    return %24 : tensor<4096xi32>
  }
}
