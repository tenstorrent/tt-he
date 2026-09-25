module {
  func.func @ntt_keep(%x: tensor<4096xi32>) -> tensor<4096xi32> {
    %0 = ntt.reshape %x : tensor<4096xi32> to tensor<8x512xi32>
    %1 = ntt.const {name = "W0"} : tensor<8x8xi32>
    %2 = ntt.matmul_modq %1, %0 {q = 167772161 : i32} : (tensor<8x8xi32>, tensor<8x512xi32>) -> tensor<8x512xi32>
    %3 = ntt.const {name = "T0"} : tensor<8x512xi32>
    %4 = ntt.mul_modq %2, %3 {q = 167772161 : i32} : (tensor<8x512xi32>, tensor<8x512xi32>) -> tensor<8x512xi32>
    %5 = ntt.transpose %4 : tensor<8x512xi32> to tensor<512x8xi32>
    %6 = ntt.reshape %5 : tensor<512x8xi32> to tensor<8x512xi32>
    %7 = ntt.const {name = "W1"} : tensor<8x8xi32>
    %8 = ntt.matmul_modq %7, %6 {q = 167772161 : i32} : (tensor<8x8xi32>, tensor<8x512xi32>) -> tensor<8x512xi32>
    %9 = ntt.const {name = "T1"} : tensor<8x512xi32>
    %10 = ntt.mul_modq %8, %9 {q = 167772161 : i32} : (tensor<8x512xi32>, tensor<8x512xi32>) -> tensor<8x512xi32>
    %11 = ntt.transpose %10 : tensor<8x512xi32> to tensor<512x8xi32>
    %12 = ntt.reshape %11 : tensor<512x8xi32> to tensor<8x512xi32>
    %13 = ntt.const {name = "W2"} : tensor<8x8xi32>
    %14 = ntt.matmul_modq %13, %12 {q = 167772161 : i32} : (tensor<8x8xi32>, tensor<8x512xi32>) -> tensor<8x512xi32>
    %15 = ntt.const {name = "T2"} : tensor<8x512xi32>
    %16 = ntt.mul_modq %14, %15 {q = 167772161 : i32} : (tensor<8x512xi32>, tensor<8x512xi32>) -> tensor<8x512xi32>
    %17 = ntt.transpose %16 : tensor<8x512xi32> to tensor<512x8xi32>
    %18 = ntt.reshape %17 : tensor<512x8xi32> to tensor<8x512xi32>
    %19 = ntt.const {name = "W3"} : tensor<8x8xi32>
    %20 = ntt.matmul_modq %19, %18 {q = 167772161 : i32} : (tensor<8x8xi32>, tensor<8x512xi32>) -> tensor<8x512xi32>
    %21 = ntt.const {name = "T3"} : tensor<8x512xi32>
    %22 = ntt.mul_modq %20, %21 {q = 167772161 : i32} : (tensor<8x512xi32>, tensor<8x512xi32>) -> tensor<8x512xi32>
    %23 = ntt.transpose %22 : tensor<8x512xi32> to tensor<512x8xi32>
    %24 = ntt.reshape %23 : tensor<512x8xi32> to tensor<4096xi32>
    return %24 : tensor<4096xi32>
  }
}
