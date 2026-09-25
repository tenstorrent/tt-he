module {
  func.func @base_convert_crns_l2(%r0: tensor<8xi32>, %r1: tensor<8xi32>, %r2: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    %o0_r_r0 = tile.reshape %r0 {shape = [1, 8]} : (tensor<8xi32>) -> tensor<1x8xi32>
    %o0_r_r1 = tile.reshape %r1 {shape = [1, 8]} : (tensor<8xi32>) -> tensor<1x8xi32>
    %o0_r_r2 = tile.reshape %r2 {shape = [1, 8]} : (tensor<8xi32>) -> tensor<1x8xi32>
    %o0_RES = tile.concat %o0_r_r0, %o0_r_r1, %o0_r_r2 {axis = 0 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<3x8xi32>
    %crnsF__40961__65537__114689__68 = tile.const {name = "crnsF_(40961, 65537, 114689)_68"} : tensor<1x3xi32>
    %o0_s = tile.matmul_wide %crnsF__40961__65537__114689__68, %o0_RES {radix = 128 : i64, nlimbs_a = 10 : i64, nlimbs_b = 3 : i64} : (tensor<1x3xi32>, tensor<3x8xi32>) -> tensor<1x8xi32>
    %o0_k = tile.fixed_slice %o0_s {shift = 68 : i64} : (tensor<1x8xi32>) -> tensor<1x8xi32>
    %o0_neg = tile.fixed_slice %o0_s {shift = 67 : i64, mask = 1 : i64} : (tensor<1x8xi32>) -> tensor<1x8xi32>
    %crnsE__40961__65537__114689__147457 = tile.const {name = "crnsE_(40961, 65537, 114689)_147457"} : tensor<1x3xi32>
    %o0_xE_147457 = tile.matmul_modq %crnsE__40961__65537__114689__147457, %o0_RES {q = 147457 : i64, radix = 128 : i64, nlimbs_a = 3 : i64, nlimbs_b = 3 : i64} : (tensor<1x3xi32>, tensor<3x8xi32>) -> tensor<1x8xi32>
    %crnsg__40961__65537__114689__147457 = tile.const {name = "crnsg_(40961, 65537, 114689)_147457", q = 147457 : i64} : tensor<1x1xi32>
    %o0_kg_147457 = tile.mul_modq %o0_k, %crnsg__40961__65537__114689__147457 {q = 147457 : i64} : (tensor<1x8xi32>, tensor<1x1xi32>) -> tensor<1x8xi32>
    %o0_acc_147457 = tile.add_modq %o0_xE_147457, %o0_kg_147457 {q = 147457 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %crnsMmod__40961__65537__114689__147457 = tile.const {name = "crnsMmod_(40961, 65537, 114689)_147457", q = 147457 : i64} : tensor<1x1xi32>
    %o0_negM_147457 = tile.mul_modq %o0_neg, %crnsMmod__40961__65537__114689__147457 {q = 147457 : i64} : (tensor<1x8xi32>, tensor<1x1xi32>) -> tensor<1x8xi32>
    %o0_accs_147457 = tile.sub_modq %o0_acc_147457, %o0_negM_147457 {q = 147457 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %o0 = tile.reshape %o0_accs_147457 {shape = [8], q = 147457 : i64} : (tensor<1x8xi32>) -> tensor<8xi32>
    %crnsE__40961__65537__114689__163841 = tile.const {name = "crnsE_(40961, 65537, 114689)_163841"} : tensor<1x3xi32>
    %o0_xE_163841 = tile.matmul_modq %crnsE__40961__65537__114689__163841, %o0_RES {q = 163841 : i64, radix = 128 : i64, nlimbs_a = 3 : i64, nlimbs_b = 3 : i64} : (tensor<1x3xi32>, tensor<3x8xi32>) -> tensor<1x8xi32>
    %crnsg__40961__65537__114689__163841 = tile.const {name = "crnsg_(40961, 65537, 114689)_163841", q = 163841 : i64} : tensor<1x1xi32>
    %o0_kg_163841 = tile.mul_modq %o0_k, %crnsg__40961__65537__114689__163841 {q = 163841 : i64} : (tensor<1x8xi32>, tensor<1x1xi32>) -> tensor<1x8xi32>
    %o0_acc_163841 = tile.add_modq %o0_xE_163841, %o0_kg_163841 {q = 163841 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %crnsMmod__40961__65537__114689__163841 = tile.const {name = "crnsMmod_(40961, 65537, 114689)_163841", q = 163841 : i64} : tensor<1x1xi32>
    %o0_negM_163841 = tile.mul_modq %o0_neg, %crnsMmod__40961__65537__114689__163841 {q = 163841 : i64} : (tensor<1x8xi32>, tensor<1x1xi32>) -> tensor<1x8xi32>
    %o0_accs_163841 = tile.sub_modq %o0_acc_163841, %o0_negM_163841 {q = 163841 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %o1 = tile.reshape %o0_accs_163841 {shape = [8], q = 163841 : i64} : (tensor<1x8xi32>) -> tensor<8xi32>
    return %o0, %o1 : tensor<8xi32>, tensor<8xi32>
  }
}
