module {
  func.func @matmul_modq(%x: tensor<8x8xi32>) -> tensor<8x8xi32> {
    %w = ntt.const {name = "W"} : tensor<8x8xi32>
    %o = ntt.matmul_modq %w, %x {q = 167772161 : i32} : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
    return %o : tensor<8x8xi32>
  }
}