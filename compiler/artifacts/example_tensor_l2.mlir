module {
  func.func @tensor_l2(%c0_0: tensor<16xi32>, %c1_0: tensor<16xi32>, %d0_0: tensor<16xi32>, %d1_0: tensor<16xi32>, %c0_1: tensor<16xi32>, %c1_1: tensor<16xi32>, %d0_1: tensor<16xi32>, %d1_1: tensor<16xi32>, %c0_2: tensor<16xi32>, %c1_2: tensor<16xi32>, %d0_2: tensor<16xi32>, %d1_2: tensor<16xi32>, %c0_3: tensor<16xi32>, %c1_3: tensor<16xi32>, %d0_3: tensor<16xi32>, %d1_3: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %psi_268435361_16 = tile.const {name = "psi_268435361_16", q = 268435361 : i64} : tensor<16xi32>
    %C0_0_s1 = rns.mul %c0_0, %psi_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C0_0_s2 = tile.reshape %C0_0_s1 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %Wf_268435361_16_0 = tile.const {name = "Wf_268435361_16_0", q = 268435361 : i64} : tensor<4x4xi32>
    %C0_0_s3 = tile.matmul_modq %Wf_268435361_16_0, %C0_0_s2 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268435361_16_0 = tile.const {name = "Tf_268435361_16_0", q = 268435361 : i64} : tensor<4x4xi32>
    %C0_0_s4 = tile.mul_modq %C0_0_s3, %Tf_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_0_s5 = tile.transpose %C0_0_s4 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_0_s6 = tile.reshape %C0_0_s5 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wf_268435361_16_1 = tile.const {name = "Wf_268435361_16_1", q = 268435361 : i64} : tensor<4x4xi32>
    %C0_0_s7 = tile.matmul_modq %Wf_268435361_16_1, %C0_0_s6 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268435361_16_1 = tile.const {name = "Tf_268435361_16_1", q = 268435361 : i64} : tensor<4x4xi32>
    %C0_0_s8 = tile.mul_modq %C0_0_s7, %Tf_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_0_s9 = tile.transpose %C0_0_s8 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_0 = tile.reshape %C0_0_s9 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %C1_0_s1 = rns.mul %c1_0, %psi_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C1_0_s2 = tile.reshape %C1_0_s1 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %C1_0_s3 = tile.matmul_modq %Wf_268435361_16_0, %C1_0_s2 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s4 = tile.mul_modq %C1_0_s3, %Tf_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s5 = tile.transpose %C1_0_s4 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s6 = tile.reshape %C1_0_s5 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s7 = tile.matmul_modq %Wf_268435361_16_1, %C1_0_s6 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s8 = tile.mul_modq %C1_0_s7, %Tf_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s9 = tile.transpose %C1_0_s8 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0 = tile.reshape %C1_0_s9 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D0_0_s1 = rns.mul %d0_0, %psi_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D0_0_s2 = tile.reshape %D0_0_s1 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %D0_0_s3 = tile.matmul_modq %Wf_268435361_16_0, %D0_0_s2 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s4 = tile.mul_modq %D0_0_s3, %Tf_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s5 = tile.transpose %D0_0_s4 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s6 = tile.reshape %D0_0_s5 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s7 = tile.matmul_modq %Wf_268435361_16_1, %D0_0_s6 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s8 = tile.mul_modq %D0_0_s7, %Tf_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s9 = tile.transpose %D0_0_s8 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0 = tile.reshape %D0_0_s9 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D1_0_s1 = rns.mul %d1_0, %psi_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D1_0_s2 = tile.reshape %D1_0_s1 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %D1_0_s3 = tile.matmul_modq %Wf_268435361_16_0, %D1_0_s2 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s4 = tile.mul_modq %D1_0_s3, %Tf_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s5 = tile.transpose %D1_0_s4 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s6 = tile.reshape %D1_0_s5 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s7 = tile.matmul_modq %Wf_268435361_16_1, %D1_0_s6 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s8 = tile.mul_modq %D1_0_s7, %Tf_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s9 = tile.transpose %D1_0_s8 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0 = tile.reshape %D1_0_s9 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %E0_0 = tile.mul_modq %C0_0, %D0_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E2_0 = tile.mul_modq %C1_0, %D1_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T0_0 = tile.mul_modq %C0_0, %D1_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T1_0 = tile.mul_modq %C1_0, %D0_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E1_0 = tile.add_modq %T0_0, %T1_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_0_s1 = tile.reshape %E0_0 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e0_0_s2 = tile.transpose %e0_0_s1 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268435361_16_1 = tile.const {name = "Ti_268435361_16_1", q = 268435361 : i64} : tensor<4x4xi32>
    %e0_0_s3 = tile.mul_modq %e0_0_s2, %Ti_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268435361_16_1 = tile.const {name = "Wi_268435361_16_1", q = 268435361 : i64} : tensor<4x4xi32>
    %e0_0_s4 = tile.matmul_modq %Wi_268435361_16_1, %e0_0_s3 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_0_s5 = tile.reshape %e0_0_s4 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_0_s6 = tile.transpose %e0_0_s5 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268435361_16_0 = tile.const {name = "Ti_268435361_16_0", q = 268435361 : i64} : tensor<4x4xi32>
    %e0_0_s7 = tile.mul_modq %e0_0_s6, %Ti_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268435361_16_0 = tile.const {name = "Wi_268435361_16_0", q = 268435361 : i64} : tensor<4x4xi32>
    %e0_0_s8 = tile.matmul_modq %Wi_268435361_16_0, %e0_0_s7 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_0_s9 = tile.reshape %e0_0_s8 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %psinv_268435361_16 = tile.const {name = "psinv_268435361_16", q = 268435361 : i64} : tensor<16xi32>
    %e0_0 = rns.mul %e0_0_s9, %psinv_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_0_s1 = tile.reshape %E1_0 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e1_0_s2 = tile.transpose %e1_0_s1 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s3 = tile.mul_modq %e1_0_s2, %Ti_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s4 = tile.matmul_modq %Wi_268435361_16_1, %e1_0_s3 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s5 = tile.reshape %e1_0_s4 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s6 = tile.transpose %e1_0_s5 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s7 = tile.mul_modq %e1_0_s6, %Ti_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s8 = tile.matmul_modq %Wi_268435361_16_0, %e1_0_s7 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s9 = tile.reshape %e1_0_s8 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e1_0 = rns.mul %e1_0_s9, %psinv_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_0_s1 = tile.reshape %E2_0 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e2_0_s2 = tile.transpose %e2_0_s1 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s3 = tile.mul_modq %e2_0_s2, %Ti_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s4 = tile.matmul_modq %Wi_268435361_16_1, %e2_0_s3 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s5 = tile.reshape %e2_0_s4 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s6 = tile.transpose %e2_0_s5 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s7 = tile.mul_modq %e2_0_s6, %Ti_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s8 = tile.matmul_modq %Wi_268435361_16_0, %e2_0_s7 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s9 = tile.reshape %e2_0_s8 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e2_0 = rns.mul %e2_0_s9, %psinv_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %psi_268435009_16 = tile.const {name = "psi_268435009_16", q = 268435009 : i64} : tensor<16xi32>
    %C0_1_s1 = rns.mul %c0_1, %psi_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C0_1_s2 = tile.reshape %C0_1_s1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %Wf_268435009_16_0 = tile.const {name = "Wf_268435009_16_0", q = 268435009 : i64} : tensor<4x4xi32>
    %C0_1_s3 = tile.matmul_modq %Wf_268435009_16_0, %C0_1_s2 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268435009_16_0 = tile.const {name = "Tf_268435009_16_0", q = 268435009 : i64} : tensor<4x4xi32>
    %C0_1_s4 = tile.mul_modq %C0_1_s3, %Tf_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_1_s5 = tile.transpose %C0_1_s4 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_1_s6 = tile.reshape %C0_1_s5 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wf_268435009_16_1 = tile.const {name = "Wf_268435009_16_1", q = 268435009 : i64} : tensor<4x4xi32>
    %C0_1_s7 = tile.matmul_modq %Wf_268435009_16_1, %C0_1_s6 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268435009_16_1 = tile.const {name = "Tf_268435009_16_1", q = 268435009 : i64} : tensor<4x4xi32>
    %C0_1_s8 = tile.mul_modq %C0_1_s7, %Tf_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_1_s9 = tile.transpose %C0_1_s8 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_1 = tile.reshape %C0_1_s9 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %C1_1_s1 = rns.mul %c1_1, %psi_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C1_1_s2 = tile.reshape %C1_1_s1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %C1_1_s3 = tile.matmul_modq %Wf_268435009_16_0, %C1_1_s2 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s4 = tile.mul_modq %C1_1_s3, %Tf_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s5 = tile.transpose %C1_1_s4 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s6 = tile.reshape %C1_1_s5 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s7 = tile.matmul_modq %Wf_268435009_16_1, %C1_1_s6 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s8 = tile.mul_modq %C1_1_s7, %Tf_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s9 = tile.transpose %C1_1_s8 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1 = tile.reshape %C1_1_s9 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D0_1_s1 = rns.mul %d0_1, %psi_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D0_1_s2 = tile.reshape %D0_1_s1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %D0_1_s3 = tile.matmul_modq %Wf_268435009_16_0, %D0_1_s2 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s4 = tile.mul_modq %D0_1_s3, %Tf_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s5 = tile.transpose %D0_1_s4 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s6 = tile.reshape %D0_1_s5 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s7 = tile.matmul_modq %Wf_268435009_16_1, %D0_1_s6 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s8 = tile.mul_modq %D0_1_s7, %Tf_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s9 = tile.transpose %D0_1_s8 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1 = tile.reshape %D0_1_s9 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D1_1_s1 = rns.mul %d1_1, %psi_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D1_1_s2 = tile.reshape %D1_1_s1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %D1_1_s3 = tile.matmul_modq %Wf_268435009_16_0, %D1_1_s2 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s4 = tile.mul_modq %D1_1_s3, %Tf_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s5 = tile.transpose %D1_1_s4 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s6 = tile.reshape %D1_1_s5 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s7 = tile.matmul_modq %Wf_268435009_16_1, %D1_1_s6 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s8 = tile.mul_modq %D1_1_s7, %Tf_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s9 = tile.transpose %D1_1_s8 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1 = tile.reshape %D1_1_s9 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %E0_1 = tile.mul_modq %C0_1, %D0_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E2_1 = tile.mul_modq %C1_1, %D1_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T0_1 = tile.mul_modq %C0_1, %D1_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T1_1 = tile.mul_modq %C1_1, %D0_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E1_1 = tile.add_modq %T0_1, %T1_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_1_s1 = tile.reshape %E0_1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e0_1_s2 = tile.transpose %e0_1_s1 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268435009_16_1 = tile.const {name = "Ti_268435009_16_1", q = 268435009 : i64} : tensor<4x4xi32>
    %e0_1_s3 = tile.mul_modq %e0_1_s2, %Ti_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268435009_16_1 = tile.const {name = "Wi_268435009_16_1", q = 268435009 : i64} : tensor<4x4xi32>
    %e0_1_s4 = tile.matmul_modq %Wi_268435009_16_1, %e0_1_s3 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_1_s5 = tile.reshape %e0_1_s4 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_1_s6 = tile.transpose %e0_1_s5 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268435009_16_0 = tile.const {name = "Ti_268435009_16_0", q = 268435009 : i64} : tensor<4x4xi32>
    %e0_1_s7 = tile.mul_modq %e0_1_s6, %Ti_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268435009_16_0 = tile.const {name = "Wi_268435009_16_0", q = 268435009 : i64} : tensor<4x4xi32>
    %e0_1_s8 = tile.matmul_modq %Wi_268435009_16_0, %e0_1_s7 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_1_s9 = tile.reshape %e0_1_s8 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %psinv_268435009_16 = tile.const {name = "psinv_268435009_16", q = 268435009 : i64} : tensor<16xi32>
    %e0_1 = rns.mul %e0_1_s9, %psinv_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_1_s1 = tile.reshape %E1_1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e1_1_s2 = tile.transpose %e1_1_s1 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s3 = tile.mul_modq %e1_1_s2, %Ti_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s4 = tile.matmul_modq %Wi_268435009_16_1, %e1_1_s3 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s5 = tile.reshape %e1_1_s4 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s6 = tile.transpose %e1_1_s5 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s7 = tile.mul_modq %e1_1_s6, %Ti_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s8 = tile.matmul_modq %Wi_268435009_16_0, %e1_1_s7 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s9 = tile.reshape %e1_1_s8 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e1_1 = rns.mul %e1_1_s9, %psinv_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_1_s1 = tile.reshape %E2_1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e2_1_s2 = tile.transpose %e2_1_s1 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s3 = tile.mul_modq %e2_1_s2, %Ti_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s4 = tile.matmul_modq %Wi_268435009_16_1, %e2_1_s3 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s5 = tile.reshape %e2_1_s4 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s6 = tile.transpose %e2_1_s5 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s7 = tile.mul_modq %e2_1_s6, %Ti_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s8 = tile.matmul_modq %Wi_268435009_16_0, %e2_1_s7 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s9 = tile.reshape %e2_1_s8 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e2_1 = rns.mul %e2_1_s9, %psinv_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %psi_268434977_16 = tile.const {name = "psi_268434977_16", q = 268434977 : i64} : tensor<16xi32>
    %C0_2_s1 = rns.mul %c0_2, %psi_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C0_2_s2 = tile.reshape %C0_2_s1 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %Wf_268434977_16_0 = tile.const {name = "Wf_268434977_16_0", q = 268434977 : i64} : tensor<4x4xi32>
    %C0_2_s3 = tile.matmul_modq %Wf_268434977_16_0, %C0_2_s2 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268434977_16_0 = tile.const {name = "Tf_268434977_16_0", q = 268434977 : i64} : tensor<4x4xi32>
    %C0_2_s4 = tile.mul_modq %C0_2_s3, %Tf_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_2_s5 = tile.transpose %C0_2_s4 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_2_s6 = tile.reshape %C0_2_s5 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wf_268434977_16_1 = tile.const {name = "Wf_268434977_16_1", q = 268434977 : i64} : tensor<4x4xi32>
    %C0_2_s7 = tile.matmul_modq %Wf_268434977_16_1, %C0_2_s6 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268434977_16_1 = tile.const {name = "Tf_268434977_16_1", q = 268434977 : i64} : tensor<4x4xi32>
    %C0_2_s8 = tile.mul_modq %C0_2_s7, %Tf_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_2_s9 = tile.transpose %C0_2_s8 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_2 = tile.reshape %C0_2_s9 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %C1_2_s1 = rns.mul %c1_2, %psi_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C1_2_s2 = tile.reshape %C1_2_s1 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %C1_2_s3 = tile.matmul_modq %Wf_268434977_16_0, %C1_2_s2 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s4 = tile.mul_modq %C1_2_s3, %Tf_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s5 = tile.transpose %C1_2_s4 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s6 = tile.reshape %C1_2_s5 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s7 = tile.matmul_modq %Wf_268434977_16_1, %C1_2_s6 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s8 = tile.mul_modq %C1_2_s7, %Tf_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s9 = tile.transpose %C1_2_s8 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2 = tile.reshape %C1_2_s9 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D0_2_s1 = rns.mul %d0_2, %psi_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D0_2_s2 = tile.reshape %D0_2_s1 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %D0_2_s3 = tile.matmul_modq %Wf_268434977_16_0, %D0_2_s2 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s4 = tile.mul_modq %D0_2_s3, %Tf_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s5 = tile.transpose %D0_2_s4 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s6 = tile.reshape %D0_2_s5 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s7 = tile.matmul_modq %Wf_268434977_16_1, %D0_2_s6 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s8 = tile.mul_modq %D0_2_s7, %Tf_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s9 = tile.transpose %D0_2_s8 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2 = tile.reshape %D0_2_s9 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D1_2_s1 = rns.mul %d1_2, %psi_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D1_2_s2 = tile.reshape %D1_2_s1 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %D1_2_s3 = tile.matmul_modq %Wf_268434977_16_0, %D1_2_s2 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s4 = tile.mul_modq %D1_2_s3, %Tf_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s5 = tile.transpose %D1_2_s4 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s6 = tile.reshape %D1_2_s5 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s7 = tile.matmul_modq %Wf_268434977_16_1, %D1_2_s6 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s8 = tile.mul_modq %D1_2_s7, %Tf_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s9 = tile.transpose %D1_2_s8 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2 = tile.reshape %D1_2_s9 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %E0_2 = tile.mul_modq %C0_2, %D0_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E2_2 = tile.mul_modq %C1_2, %D1_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T0_2 = tile.mul_modq %C0_2, %D1_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T1_2 = tile.mul_modq %C1_2, %D0_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E1_2 = tile.add_modq %T0_2, %T1_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_2_s1 = tile.reshape %E0_2 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e0_2_s2 = tile.transpose %e0_2_s1 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268434977_16_1 = tile.const {name = "Ti_268434977_16_1", q = 268434977 : i64} : tensor<4x4xi32>
    %e0_2_s3 = tile.mul_modq %e0_2_s2, %Ti_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268434977_16_1 = tile.const {name = "Wi_268434977_16_1", q = 268434977 : i64} : tensor<4x4xi32>
    %e0_2_s4 = tile.matmul_modq %Wi_268434977_16_1, %e0_2_s3 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_2_s5 = tile.reshape %e0_2_s4 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_2_s6 = tile.transpose %e0_2_s5 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268434977_16_0 = tile.const {name = "Ti_268434977_16_0", q = 268434977 : i64} : tensor<4x4xi32>
    %e0_2_s7 = tile.mul_modq %e0_2_s6, %Ti_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268434977_16_0 = tile.const {name = "Wi_268434977_16_0", q = 268434977 : i64} : tensor<4x4xi32>
    %e0_2_s8 = tile.matmul_modq %Wi_268434977_16_0, %e0_2_s7 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_2_s9 = tile.reshape %e0_2_s8 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %psinv_268434977_16 = tile.const {name = "psinv_268434977_16", q = 268434977 : i64} : tensor<16xi32>
    %e0_2 = rns.mul %e0_2_s9, %psinv_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_2_s1 = tile.reshape %E1_2 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e1_2_s2 = tile.transpose %e1_2_s1 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s3 = tile.mul_modq %e1_2_s2, %Ti_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s4 = tile.matmul_modq %Wi_268434977_16_1, %e1_2_s3 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s5 = tile.reshape %e1_2_s4 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s6 = tile.transpose %e1_2_s5 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s7 = tile.mul_modq %e1_2_s6, %Ti_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s8 = tile.matmul_modq %Wi_268434977_16_0, %e1_2_s7 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s9 = tile.reshape %e1_2_s8 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e1_2 = rns.mul %e1_2_s9, %psinv_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_2_s1 = tile.reshape %E2_2 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e2_2_s2 = tile.transpose %e2_2_s1 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s3 = tile.mul_modq %e2_2_s2, %Ti_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s4 = tile.matmul_modq %Wi_268434977_16_1, %e2_2_s3 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s5 = tile.reshape %e2_2_s4 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s6 = tile.transpose %e2_2_s5 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s7 = tile.mul_modq %e2_2_s6, %Ti_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s8 = tile.matmul_modq %Wi_268434977_16_0, %e2_2_s7 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s9 = tile.reshape %e2_2_s8 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e2_2 = rns.mul %e2_2_s9, %psinv_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %psi_268434721_16 = tile.const {name = "psi_268434721_16", q = 268434721 : i64} : tensor<16xi32>
    %C0_3_s1 = rns.mul %c0_3, %psi_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C0_3_s2 = tile.reshape %C0_3_s1 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %Wf_268434721_16_0 = tile.const {name = "Wf_268434721_16_0", q = 268434721 : i64} : tensor<4x4xi32>
    %C0_3_s3 = tile.matmul_modq %Wf_268434721_16_0, %C0_3_s2 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268434721_16_0 = tile.const {name = "Tf_268434721_16_0", q = 268434721 : i64} : tensor<4x4xi32>
    %C0_3_s4 = tile.mul_modq %C0_3_s3, %Tf_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_3_s5 = tile.transpose %C0_3_s4 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_3_s6 = tile.reshape %C0_3_s5 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wf_268434721_16_1 = tile.const {name = "Wf_268434721_16_1", q = 268434721 : i64} : tensor<4x4xi32>
    %C0_3_s7 = tile.matmul_modq %Wf_268434721_16_1, %C0_3_s6 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268434721_16_1 = tile.const {name = "Tf_268434721_16_1", q = 268434721 : i64} : tensor<4x4xi32>
    %C0_3_s8 = tile.mul_modq %C0_3_s7, %Tf_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_3_s9 = tile.transpose %C0_3_s8 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_3 = tile.reshape %C0_3_s9 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %C1_3_s1 = rns.mul %c1_3, %psi_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C1_3_s2 = tile.reshape %C1_3_s1 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %C1_3_s3 = tile.matmul_modq %Wf_268434721_16_0, %C1_3_s2 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s4 = tile.mul_modq %C1_3_s3, %Tf_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s5 = tile.transpose %C1_3_s4 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s6 = tile.reshape %C1_3_s5 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s7 = tile.matmul_modq %Wf_268434721_16_1, %C1_3_s6 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s8 = tile.mul_modq %C1_3_s7, %Tf_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s9 = tile.transpose %C1_3_s8 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3 = tile.reshape %C1_3_s9 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D0_3_s1 = rns.mul %d0_3, %psi_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D0_3_s2 = tile.reshape %D0_3_s1 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %D0_3_s3 = tile.matmul_modq %Wf_268434721_16_0, %D0_3_s2 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s4 = tile.mul_modq %D0_3_s3, %Tf_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s5 = tile.transpose %D0_3_s4 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s6 = tile.reshape %D0_3_s5 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s7 = tile.matmul_modq %Wf_268434721_16_1, %D0_3_s6 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s8 = tile.mul_modq %D0_3_s7, %Tf_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s9 = tile.transpose %D0_3_s8 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3 = tile.reshape %D0_3_s9 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D1_3_s1 = rns.mul %d1_3, %psi_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D1_3_s2 = tile.reshape %D1_3_s1 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %D1_3_s3 = tile.matmul_modq %Wf_268434721_16_0, %D1_3_s2 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s4 = tile.mul_modq %D1_3_s3, %Tf_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s5 = tile.transpose %D1_3_s4 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s6 = tile.reshape %D1_3_s5 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s7 = tile.matmul_modq %Wf_268434721_16_1, %D1_3_s6 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s8 = tile.mul_modq %D1_3_s7, %Tf_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s9 = tile.transpose %D1_3_s8 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3 = tile.reshape %D1_3_s9 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %E0_3 = tile.mul_modq %C0_3, %D0_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E2_3 = tile.mul_modq %C1_3, %D1_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T0_3 = tile.mul_modq %C0_3, %D1_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T1_3 = tile.mul_modq %C1_3, %D0_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E1_3 = tile.add_modq %T0_3, %T1_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_3_s1 = tile.reshape %E0_3 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e0_3_s2 = tile.transpose %e0_3_s1 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268434721_16_1 = tile.const {name = "Ti_268434721_16_1", q = 268434721 : i64} : tensor<4x4xi32>
    %e0_3_s3 = tile.mul_modq %e0_3_s2, %Ti_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268434721_16_1 = tile.const {name = "Wi_268434721_16_1", q = 268434721 : i64} : tensor<4x4xi32>
    %e0_3_s4 = tile.matmul_modq %Wi_268434721_16_1, %e0_3_s3 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_3_s5 = tile.reshape %e0_3_s4 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_3_s6 = tile.transpose %e0_3_s5 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268434721_16_0 = tile.const {name = "Ti_268434721_16_0", q = 268434721 : i64} : tensor<4x4xi32>
    %e0_3_s7 = tile.mul_modq %e0_3_s6, %Ti_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268434721_16_0 = tile.const {name = "Wi_268434721_16_0", q = 268434721 : i64} : tensor<4x4xi32>
    %e0_3_s8 = tile.matmul_modq %Wi_268434721_16_0, %e0_3_s7 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_3_s9 = tile.reshape %e0_3_s8 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %psinv_268434721_16 = tile.const {name = "psinv_268434721_16", q = 268434721 : i64} : tensor<16xi32>
    %e0_3 = rns.mul %e0_3_s9, %psinv_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_3_s1 = tile.reshape %E1_3 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e1_3_s2 = tile.transpose %e1_3_s1 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s3 = tile.mul_modq %e1_3_s2, %Ti_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s4 = tile.matmul_modq %Wi_268434721_16_1, %e1_3_s3 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s5 = tile.reshape %e1_3_s4 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s6 = tile.transpose %e1_3_s5 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s7 = tile.mul_modq %e1_3_s6, %Ti_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s8 = tile.matmul_modq %Wi_268434721_16_0, %e1_3_s7 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s9 = tile.reshape %e1_3_s8 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e1_3 = rns.mul %e1_3_s9, %psinv_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_3_s1 = tile.reshape %E2_3 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e2_3_s2 = tile.transpose %e2_3_s1 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s3 = tile.mul_modq %e2_3_s2, %Ti_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s4 = tile.matmul_modq %Wi_268434721_16_1, %e2_3_s3 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s5 = tile.reshape %e2_3_s4 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s6 = tile.transpose %e2_3_s5 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s7 = tile.mul_modq %e2_3_s6, %Ti_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s8 = tile.matmul_modq %Wi_268434721_16_0, %e2_3_s7 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s9 = tile.reshape %e2_3_s8 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e2_3 = rns.mul %e2_3_s9, %psinv_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    return %e0_0, %e0_1, %e0_2, %e0_3, %e1_0, %e1_1, %e1_2, %e1_3, %e2_0, %e2_1, %e2_2, %e2_3 : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }
}
