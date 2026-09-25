module {
  func.func @base_convert_crns_l3(%r0: tensor<8xi32>, %r1: tensor<8xi32>, %r2: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    %o0_r_r0 = tile.reshape %r0 {shape = [1, 8]} : (tensor<8xi32>) -> tensor<1x8xi32>
    %o0_r_r1 = tile.reshape %r1 {shape = [1, 8]} : (tensor<8xi32>) -> tensor<1x8xi32>
    %o0_r_r2 = tile.reshape %r2 {shape = [1, 8]} : (tensor<8xi32>) -> tensor<1x8xi32>
    %o0_RES = tile.concat %o0_r_r0, %o0_r_r1, %o0_r_r2 {axis = 0 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<3x8xi32>
    %crnsF__40961__65537__114689__68 = tile.const {name = "crnsF_(40961, 65537, 114689)_68"} : tensor<1x3xi32>
    %a_1, %a_2, %a_3, %a_4, %a_5, %a_6, %a_7, %a_8, %a_9, %a_10 = tile.decompose %crnsF__40961__65537__114689__68 {radix = 128 : i64, nlimbs = 10 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_11, %b_12, %b_13 = tile.decompose %o0_RES {radix = 128 : i64, nlimbs = 3 : i64} : (tensor<3x8xi32>) -> (tensor<3x8xi8>, tensor<3x8xi8>, tensor<3x8xi8>)
    %wP_14 = tile.matmul_wide %a_1, %b_11 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_15 = tile.matmul_wide %a_1, %b_12 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_16 = tile.matmul_wide %a_1, %b_13 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_17 = tile.matmul_wide %a_2, %b_11 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_18 = tile.matmul_wide %a_2, %b_12 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_19 = tile.matmul_wide %a_2, %b_13 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_20 = tile.matmul_wide %a_3, %b_11 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_21 = tile.matmul_wide %a_3, %b_12 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_22 = tile.matmul_wide %a_3, %b_13 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_23 = tile.matmul_wide %a_4, %b_11 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_24 = tile.matmul_wide %a_4, %b_12 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_25 = tile.matmul_wide %a_4, %b_13 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_26 = tile.matmul_wide %a_5, %b_11 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_27 = tile.matmul_wide %a_5, %b_12 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_28 = tile.matmul_wide %a_5, %b_13 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_29 = tile.matmul_wide %a_6, %b_11 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_30 = tile.matmul_wide %a_6, %b_12 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_31 = tile.matmul_wide %a_6, %b_13 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_32 = tile.matmul_wide %a_7, %b_11 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_33 = tile.matmul_wide %a_7, %b_12 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_34 = tile.matmul_wide %a_7, %b_13 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_35 = tile.matmul_wide %a_8, %b_11 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_36 = tile.matmul_wide %a_8, %b_12 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_37 = tile.matmul_wide %a_8, %b_13 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_38 = tile.matmul_wide %a_9, %b_11 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_39 = tile.matmul_wide %a_9, %b_12 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_40 = tile.matmul_wide %a_9, %b_13 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_41 = tile.matmul_wide %a_10, %b_11 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_42 = tile.matmul_wide %a_10, %b_12 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %wP_43 = tile.matmul_wide %a_10, %b_13 : (tensor<1x3xi8>, tensor<3x8xi8>) -> tensor<1x8xi32>
    %o0_s = tile.recombine_wide %wP_14, %wP_15, %wP_16, %wP_17, %wP_18, %wP_19, %wP_20, %wP_21, %wP_22, %wP_23, %wP_24, %wP_25, %wP_26, %wP_27, %wP_28, %wP_29, %wP_30, %wP_31, %wP_32, %wP_33, %wP_34, %wP_35, %wP_36, %wP_37, %wP_38, %wP_39, %wP_40, %wP_41, %wP_42, %wP_43 {radix = 128 : i64, exps = [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10, 9, 10, 11]} : (tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %o0_sN = tile.reshape %o0_s {shape = [8]} : (tensor<1x8xi32>) -> tensor<8xi32>
    %o0_k = tile.fixed_slice %o0_sN {shift = 68 : i64} : (tensor<8xi32>) -> tensor<8xi32>
    %o0_kR = tile.reshape %o0_k {shape = [1, 8]} : (tensor<8xi32>) -> tensor<1x8xi32>
    %o0_neg = tile.fixed_slice %o0_sN {shift = 67 : i64, mask = 1 : i64} : (tensor<8xi32>) -> tensor<8xi32>
    %o0_negR = tile.reshape %o0_neg {shape = [1, 8]} : (tensor<8xi32>) -> tensor<1x8xi32>
    %o0_Rp = tile.concat %o0_r_r0, %o0_r_r1, %o0_r_r2, %o0_kR, %o0_negR {axis = 0 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<5x8xi32>
    %crnsW__40961__65537__114689__147457_True = tile.const {name = "crnsW_(40961, 65537, 114689)_147457_True"} : tensor<1x5xi32>
    %a_44, %a_45, %a_46 = tile.decompose %crnsW__40961__65537__114689__147457_True {radix = 128 : i64, nlimbs = 3 : i64} : (tensor<1x5xi32>) -> (tensor<1x5xi8>, tensor<1x5xi8>, tensor<1x5xi8>)
    %b_47, %b_48, %b_49 = tile.decompose %o0_Rp {radix = 128 : i64, nlimbs = 3 : i64} : (tensor<5x8xi32>) -> (tensor<5x8xi8>, tensor<5x8xi8>, tensor<5x8xi8>)
    %P_50 = tile.matmul_modq %a_44, %b_47 {q = 147457 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_51 = tile.matmul_modq %a_44, %b_48 {q = 147457 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_52 = tile.matmul_modq %a_44, %b_49 {q = 147457 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_53 = tile.matmul_modq %a_45, %b_47 {q = 147457 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_54 = tile.matmul_modq %a_45, %b_48 {q = 147457 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_55 = tile.matmul_modq %a_45, %b_49 {q = 147457 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_56 = tile.matmul_modq %a_46, %b_47 {q = 147457 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_57 = tile.matmul_modq %a_46, %b_48 {q = 147457 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_58 = tile.matmul_modq %a_46, %b_49 {q = 147457 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %C_59 = tile.add_modq %P_50 {q = 147457 : i64} : (tensor<1x8xi32>) -> tensor<1x8xi32>
    %C_60 = tile.add_modq %P_51, %P_53 {q = 147457 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %C_61 = tile.add_modq %P_52, %P_54, %P_56 {q = 147457 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %C_62 = tile.add_modq %P_55, %P_57 {q = 147457 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %C_63 = tile.add_modq %P_58 {q = 147457 : i64} : (tensor<1x8xi32>) -> tensor<1x8xi32>
    %o0_y_147457 = tile.recombine_modq %C_59, %C_60, %C_61, %C_62, %C_63 {radix = 128 : i64, q = 147457 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %o0 = tile.reshape %o0_y_147457 {shape = [8], q = 147457 : i64} : (tensor<1x8xi32>) -> tensor<8xi32>
    %crnsW__40961__65537__114689__163841_True = tile.const {name = "crnsW_(40961, 65537, 114689)_163841_True"} : tensor<1x5xi32>
    %a_64, %a_65, %a_66 = tile.decompose %crnsW__40961__65537__114689__163841_True {radix = 128 : i64, nlimbs = 3 : i64} : (tensor<1x5xi32>) -> (tensor<1x5xi8>, tensor<1x5xi8>, tensor<1x5xi8>)
    %b_67, %b_68, %b_69 = tile.decompose %o0_Rp {radix = 128 : i64, nlimbs = 3 : i64} : (tensor<5x8xi32>) -> (tensor<5x8xi8>, tensor<5x8xi8>, tensor<5x8xi8>)
    %P_70 = tile.matmul_modq %a_64, %b_67 {q = 163841 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_71 = tile.matmul_modq %a_64, %b_68 {q = 163841 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_72 = tile.matmul_modq %a_64, %b_69 {q = 163841 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_73 = tile.matmul_modq %a_65, %b_67 {q = 163841 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_74 = tile.matmul_modq %a_65, %b_68 {q = 163841 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_75 = tile.matmul_modq %a_65, %b_69 {q = 163841 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_76 = tile.matmul_modq %a_66, %b_67 {q = 163841 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_77 = tile.matmul_modq %a_66, %b_68 {q = 163841 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %P_78 = tile.matmul_modq %a_66, %b_69 {q = 163841 : i64} : (tensor<1x5xi8>, tensor<5x8xi8>) -> tensor<1x8xi32>
    %C_79 = tile.add_modq %P_70 {q = 163841 : i64} : (tensor<1x8xi32>) -> tensor<1x8xi32>
    %C_80 = tile.add_modq %P_71, %P_73 {q = 163841 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %C_81 = tile.add_modq %P_72, %P_74, %P_76 {q = 163841 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %C_82 = tile.add_modq %P_75, %P_77 {q = 163841 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %C_83 = tile.add_modq %P_78 {q = 163841 : i64} : (tensor<1x8xi32>) -> tensor<1x8xi32>
    %o0_y_163841 = tile.recombine_modq %C_79, %C_80, %C_81, %C_82, %C_83 {radix = 128 : i64, q = 163841 : i64} : (tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %o1 = tile.reshape %o0_y_163841 {shape = [8], q = 163841 : i64} : (tensor<1x8xi32>) -> tensor<8xi32>
    return %o0, %o1 : tensor<8xi32>, tensor<8xi32>
  }
}
