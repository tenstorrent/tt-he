module {
  func.func @ntt(%x: tensor<2048xi32>) -> tensor<2048xi32> {
    %0 = ntt.reshape %x : tensor<2048xi32> to tensor<32x64xi32>
    %1 = ntt.const {name = "W0_T"} : tensor<32x32xi32>
    %0__t1, %0__t2 = ntt.split %0 {axis = 1 : index, parts = 2 : index} : tensor<32x64xi32> -> (tensor<32x32xi32>, tensor<32x32xi32>)
    %2__t3 = ntt.matmul_modq %0__t1, %1 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %3__t4 = ntt.const {name = "T0_T__r0"} : tensor<32x32xi32>
    %2__t5 = ntt.mul_modq %2__t3, %3__t4 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %2__t6 = ntt.matmul_modq %0__t2, %1 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %3__t7 = ntt.const {name = "T0_T__r1"} : tensor<32x32xi32>
    %2__t8 = ntt.mul_modq %2__t6, %3__t7 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m0, %m3 = ntt.all_to_all_tiles %2__t5, %2__t8 {perm = [0, 1]} : (tensor<32x32xi32>, tensor<32x32xi32>) -> (tensor<32x32xi32>, tensor<32x32xi32>)
    %7 = ntt.const {name = "W1_f_T"} : tensor<64x64xi32>
    %m1 = ntt.const {name = "W1_f_T__0.0_1.0"} : tensor<32x32xi32>
    %m2 = ntt.matmul_modq %m0, %m1 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m4 = ntt.const {name = "W1_f_T__0.32_1.0"} : tensor<32x32xi32>
    %m5 = ntt.matmul_modq %m3, %m4 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m6 = ntt.add_modq %m2, %m5 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m7 = ntt.const {name = "W1_f_T__0.0_1.32"} : tensor<32x32xi32>
    %m8 = ntt.matmul_modq %m0, %m7 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m9 = ntt.const {name = "W1_f_T__0.32_1.32"} : tensor<32x32xi32>
    %m10 = ntt.matmul_modq %m3, %m9 {q = 167772161 : i32, transpose_a} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %m11 = ntt.add_modq %m8, %m10 {q = 167772161 : i32} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
    %8 = ntt.concat %m6, %m11 {axis = 1 : index} : (tensor<32x32xi32>, tensor<32x32xi32>) -> tensor<32x64xi32>
    %12 = ntt.reshape %8 : tensor<32x64xi32> to tensor<2048xi32>
    return %12 : tensor<2048xi32>
  }
}