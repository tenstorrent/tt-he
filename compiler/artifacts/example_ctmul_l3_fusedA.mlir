module {
  func.func @ct_mul_l3_fusedA(%c00: tensor<16xi32>, %c01: tensor<16xi32>, %c02: tensor<16xi32>, %c03: tensor<16xi32>, %c04: tensor<16xi32>, %c05: tensor<16xi32>, %c06: tensor<16xi32>, %c10: tensor<16xi32>, %c11: tensor<16xi32>, %c12: tensor<16xi32>, %c13: tensor<16xi32>, %c14: tensor<16xi32>, %c15: tensor<16xi32>, %c16: tensor<16xi32>, %d00: tensor<16xi32>, %d01: tensor<16xi32>, %d02: tensor<16xi32>, %d03: tensor<16xi32>, %d04: tensor<16xi32>, %d05: tensor<16xi32>, %d06: tensor<16xi32>, %d10: tensor<16xi32>, %d11: tensor<16xi32>, %d12: tensor<16xi32>, %d13: tensor<16xi32>, %d14: tensor<16xi32>, %d15: tensor<16xi32>, %d16: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %W_f_268435361 = tile.const {name = "W_f_268435361", q = 268435361 : i64} : tensor<16x16xi32>
    %col_fa1 = tile.reshape %c00 {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa2 = tile.reshape %c10 {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa3 = tile.reshape %d00 {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa4 = tile.reshape %d10 {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa5 = tile.concat %col_fa1, %col_fa2, %col_fa3, %col_fa4 {axis = 1 : i64, q = 268435361 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x4xi32>
    %a_1, %a_2, %a_3, %a_4 = tile.decompose %W_f_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_5, %b_6, %b_7, %b_8 = tile.decompose %rhs_fa5 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>)
    %P_9 = tile.matmul_modq %a_1, %b_5 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_10 = tile.matmul_modq %a_1, %b_6 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_11 = tile.matmul_modq %a_1, %b_7 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_12 = tile.matmul_modq %a_1, %b_8 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_13 = tile.matmul_modq %a_2, %b_5 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_14 = tile.matmul_modq %a_2, %b_6 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_15 = tile.matmul_modq %a_2, %b_7 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_16 = tile.matmul_modq %a_2, %b_8 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_17 = tile.matmul_modq %a_3, %b_5 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_18 = tile.matmul_modq %a_3, %b_6 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_19 = tile.matmul_modq %a_3, %b_7 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_20 = tile.matmul_modq %a_3, %b_8 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_21 = tile.matmul_modq %a_4, %b_5 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_22 = tile.matmul_modq %a_4, %b_6 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_23 = tile.matmul_modq %a_4, %b_7 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_24 = tile.matmul_modq %a_4, %b_8 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %C_25 = tile.add_modq %P_9 {q = 268435361 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_26 = tile.add_modq %P_10, %P_13 {q = 268435361 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_27 = tile.add_modq %P_11, %P_14, %P_17 {q = 268435361 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_28 = tile.add_modq %P_12, %P_15, %P_18, %P_21 {q = 268435361 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_29 = tile.add_modq %P_16, %P_19, %P_22 {q = 268435361 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_30 = tile.add_modq %P_20, %P_23 {q = 268435361 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_31 = tile.add_modq %P_24 {q = 268435361 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %out_fa6 = tile.recombine_modq %C_25, %C_26, %C_27, %C_28, %C_29, %C_30, %C_31 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %yc_fa7, %yc_fa8, %yc_fa9, %yc_fa10 = tile.split %out_fa6 {axis = 1 : i64, parts = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %tC00 = tile.reshape %yc_fa7 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tC10 = tile.reshape %yc_fa8 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD00 = tile.reshape %yc_fa9 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD10 = tile.reshape %yc_fa10 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tE00 = tile.mul_modq %tC00, %tD00 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE20 = tile.mul_modq %tC10, %tD10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT00 = tile.mul_modq %tC00, %tD10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT10 = tile.mul_modq %tC10, %tD00 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE10 = tile.add_modq %tT00, %tT10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268435361 = tile.const {name = "W_i_268435361", q = 268435361 : i64} : tensor<16x16xi32>
    %col_fa11 = tile.reshape %tE00 {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa12 = tile.reshape %tE20 {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa13 = tile.reshape %tE10 {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa14 = tile.concat %col_fa11, %col_fa12, %col_fa13 {axis = 1 : i64, q = 268435361 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x3xi32>
    %a_32, %a_33, %a_34, %a_35 = tile.decompose %W_i_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_36, %b_37, %b_38, %b_39 = tile.decompose %rhs_fa14 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x3xi32>) -> (tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>)
    %P_40 = tile.matmul_modq %a_32, %b_36 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_41 = tile.matmul_modq %a_32, %b_37 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_42 = tile.matmul_modq %a_32, %b_38 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_43 = tile.matmul_modq %a_32, %b_39 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_44 = tile.matmul_modq %a_33, %b_36 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_45 = tile.matmul_modq %a_33, %b_37 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_46 = tile.matmul_modq %a_33, %b_38 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_47 = tile.matmul_modq %a_33, %b_39 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_48 = tile.matmul_modq %a_34, %b_36 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_49 = tile.matmul_modq %a_34, %b_37 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_50 = tile.matmul_modq %a_34, %b_38 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_51 = tile.matmul_modq %a_34, %b_39 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_52 = tile.matmul_modq %a_35, %b_36 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_53 = tile.matmul_modq %a_35, %b_37 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_54 = tile.matmul_modq %a_35, %b_38 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_55 = tile.matmul_modq %a_35, %b_39 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %C_56 = tile.add_modq %P_40 {q = 268435361 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_57 = tile.add_modq %P_41, %P_44 {q = 268435361 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_58 = tile.add_modq %P_42, %P_45, %P_48 {q = 268435361 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_59 = tile.add_modq %P_43, %P_46, %P_49, %P_52 {q = 268435361 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_60 = tile.add_modq %P_47, %P_50, %P_53 {q = 268435361 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_61 = tile.add_modq %P_51, %P_54 {q = 268435361 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_62 = tile.add_modq %P_55 {q = 268435361 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %out_fa15 = tile.recombine_modq %C_56, %C_57, %C_58, %C_59, %C_60, %C_61, %C_62 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %yc_fa16, %yc_fa17, %yc_fa18 = tile.split %out_fa15 {axis = 1 : i64, parts = 3 : i64} : (tensor<16x3xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %e0_0 = tile.reshape %yc_fa16 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e2_0 = tile.reshape %yc_fa17 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e1_0 = tile.reshape %yc_fa18 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %W_f_268435009 = tile.const {name = "W_f_268435009", q = 268435009 : i64} : tensor<16x16xi32>
    %col_fa19 = tile.reshape %c01 {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa20 = tile.reshape %c11 {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa21 = tile.reshape %d01 {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa22 = tile.reshape %d11 {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa23 = tile.concat %col_fa19, %col_fa20, %col_fa21, %col_fa22 {axis = 1 : i64, q = 268435009 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x4xi32>
    %a_63, %a_64, %a_65, %a_66 = tile.decompose %W_f_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_67, %b_68, %b_69, %b_70 = tile.decompose %rhs_fa23 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>)
    %P_71 = tile.matmul_modq %a_63, %b_67 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_72 = tile.matmul_modq %a_63, %b_68 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_73 = tile.matmul_modq %a_63, %b_69 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_74 = tile.matmul_modq %a_63, %b_70 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_75 = tile.matmul_modq %a_64, %b_67 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_76 = tile.matmul_modq %a_64, %b_68 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_77 = tile.matmul_modq %a_64, %b_69 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_78 = tile.matmul_modq %a_64, %b_70 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_79 = tile.matmul_modq %a_65, %b_67 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_80 = tile.matmul_modq %a_65, %b_68 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_81 = tile.matmul_modq %a_65, %b_69 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_82 = tile.matmul_modq %a_65, %b_70 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_83 = tile.matmul_modq %a_66, %b_67 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_84 = tile.matmul_modq %a_66, %b_68 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_85 = tile.matmul_modq %a_66, %b_69 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_86 = tile.matmul_modq %a_66, %b_70 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %C_87 = tile.add_modq %P_71 {q = 268435009 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_88 = tile.add_modq %P_72, %P_75 {q = 268435009 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_89 = tile.add_modq %P_73, %P_76, %P_79 {q = 268435009 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_90 = tile.add_modq %P_74, %P_77, %P_80, %P_83 {q = 268435009 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_91 = tile.add_modq %P_78, %P_81, %P_84 {q = 268435009 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_92 = tile.add_modq %P_82, %P_85 {q = 268435009 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_93 = tile.add_modq %P_86 {q = 268435009 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %out_fa24 = tile.recombine_modq %C_87, %C_88, %C_89, %C_90, %C_91, %C_92, %C_93 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %yc_fa25, %yc_fa26, %yc_fa27, %yc_fa28 = tile.split %out_fa24 {axis = 1 : i64, parts = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %tC01 = tile.reshape %yc_fa25 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tC11 = tile.reshape %yc_fa26 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD01 = tile.reshape %yc_fa27 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD11 = tile.reshape %yc_fa28 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tE01 = tile.mul_modq %tC01, %tD01 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE21 = tile.mul_modq %tC11, %tD11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT01 = tile.mul_modq %tC01, %tD11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT11 = tile.mul_modq %tC11, %tD01 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE11 = tile.add_modq %tT01, %tT11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268435009 = tile.const {name = "W_i_268435009", q = 268435009 : i64} : tensor<16x16xi32>
    %col_fa29 = tile.reshape %tE01 {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa30 = tile.reshape %tE21 {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa31 = tile.reshape %tE11 {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa32 = tile.concat %col_fa29, %col_fa30, %col_fa31 {axis = 1 : i64, q = 268435009 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x3xi32>
    %a_94, %a_95, %a_96, %a_97 = tile.decompose %W_i_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_98, %b_99, %b_100, %b_101 = tile.decompose %rhs_fa32 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x3xi32>) -> (tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>)
    %P_102 = tile.matmul_modq %a_94, %b_98 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_103 = tile.matmul_modq %a_94, %b_99 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_104 = tile.matmul_modq %a_94, %b_100 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_105 = tile.matmul_modq %a_94, %b_101 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_106 = tile.matmul_modq %a_95, %b_98 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_107 = tile.matmul_modq %a_95, %b_99 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_108 = tile.matmul_modq %a_95, %b_100 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_109 = tile.matmul_modq %a_95, %b_101 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_110 = tile.matmul_modq %a_96, %b_98 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_111 = tile.matmul_modq %a_96, %b_99 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_112 = tile.matmul_modq %a_96, %b_100 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_113 = tile.matmul_modq %a_96, %b_101 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_114 = tile.matmul_modq %a_97, %b_98 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_115 = tile.matmul_modq %a_97, %b_99 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_116 = tile.matmul_modq %a_97, %b_100 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_117 = tile.matmul_modq %a_97, %b_101 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %C_118 = tile.add_modq %P_102 {q = 268435009 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_119 = tile.add_modq %P_103, %P_106 {q = 268435009 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_120 = tile.add_modq %P_104, %P_107, %P_110 {q = 268435009 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_121 = tile.add_modq %P_105, %P_108, %P_111, %P_114 {q = 268435009 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_122 = tile.add_modq %P_109, %P_112, %P_115 {q = 268435009 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_123 = tile.add_modq %P_113, %P_116 {q = 268435009 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_124 = tile.add_modq %P_117 {q = 268435009 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %out_fa33 = tile.recombine_modq %C_118, %C_119, %C_120, %C_121, %C_122, %C_123, %C_124 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %yc_fa34, %yc_fa35, %yc_fa36 = tile.split %out_fa33 {axis = 1 : i64, parts = 3 : i64} : (tensor<16x3xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %e0_1 = tile.reshape %yc_fa34 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e2_1 = tile.reshape %yc_fa35 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e1_1 = tile.reshape %yc_fa36 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %W_f_268434977 = tile.const {name = "W_f_268434977", q = 268434977 : i64} : tensor<16x16xi32>
    %col_fa37 = tile.reshape %c02 {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa38 = tile.reshape %c12 {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa39 = tile.reshape %d02 {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa40 = tile.reshape %d12 {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa41 = tile.concat %col_fa37, %col_fa38, %col_fa39, %col_fa40 {axis = 1 : i64, q = 268434977 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x4xi32>
    %a_125, %a_126, %a_127, %a_128 = tile.decompose %W_f_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_129, %b_130, %b_131, %b_132 = tile.decompose %rhs_fa41 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>)
    %P_133 = tile.matmul_modq %a_125, %b_129 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_134 = tile.matmul_modq %a_125, %b_130 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_135 = tile.matmul_modq %a_125, %b_131 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_136 = tile.matmul_modq %a_125, %b_132 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_137 = tile.matmul_modq %a_126, %b_129 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_138 = tile.matmul_modq %a_126, %b_130 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_139 = tile.matmul_modq %a_126, %b_131 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_140 = tile.matmul_modq %a_126, %b_132 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_141 = tile.matmul_modq %a_127, %b_129 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_142 = tile.matmul_modq %a_127, %b_130 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_143 = tile.matmul_modq %a_127, %b_131 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_144 = tile.matmul_modq %a_127, %b_132 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_145 = tile.matmul_modq %a_128, %b_129 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_146 = tile.matmul_modq %a_128, %b_130 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_147 = tile.matmul_modq %a_128, %b_131 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_148 = tile.matmul_modq %a_128, %b_132 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %C_149 = tile.add_modq %P_133 {q = 268434977 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_150 = tile.add_modq %P_134, %P_137 {q = 268434977 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_151 = tile.add_modq %P_135, %P_138, %P_141 {q = 268434977 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_152 = tile.add_modq %P_136, %P_139, %P_142, %P_145 {q = 268434977 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_153 = tile.add_modq %P_140, %P_143, %P_146 {q = 268434977 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_154 = tile.add_modq %P_144, %P_147 {q = 268434977 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_155 = tile.add_modq %P_148 {q = 268434977 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %out_fa42 = tile.recombine_modq %C_149, %C_150, %C_151, %C_152, %C_153, %C_154, %C_155 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %yc_fa43, %yc_fa44, %yc_fa45, %yc_fa46 = tile.split %out_fa42 {axis = 1 : i64, parts = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %tC02 = tile.reshape %yc_fa43 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tC12 = tile.reshape %yc_fa44 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD02 = tile.reshape %yc_fa45 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD12 = tile.reshape %yc_fa46 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tE02 = tile.mul_modq %tC02, %tD02 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE22 = tile.mul_modq %tC12, %tD12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT02 = tile.mul_modq %tC02, %tD12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT12 = tile.mul_modq %tC12, %tD02 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE12 = tile.add_modq %tT02, %tT12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268434977 = tile.const {name = "W_i_268434977", q = 268434977 : i64} : tensor<16x16xi32>
    %col_fa47 = tile.reshape %tE02 {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa48 = tile.reshape %tE22 {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa49 = tile.reshape %tE12 {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa50 = tile.concat %col_fa47, %col_fa48, %col_fa49 {axis = 1 : i64, q = 268434977 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x3xi32>
    %a_156, %a_157, %a_158, %a_159 = tile.decompose %W_i_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_160, %b_161, %b_162, %b_163 = tile.decompose %rhs_fa50 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x3xi32>) -> (tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>)
    %P_164 = tile.matmul_modq %a_156, %b_160 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_165 = tile.matmul_modq %a_156, %b_161 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_166 = tile.matmul_modq %a_156, %b_162 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_167 = tile.matmul_modq %a_156, %b_163 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_168 = tile.matmul_modq %a_157, %b_160 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_169 = tile.matmul_modq %a_157, %b_161 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_170 = tile.matmul_modq %a_157, %b_162 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_171 = tile.matmul_modq %a_157, %b_163 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_172 = tile.matmul_modq %a_158, %b_160 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_173 = tile.matmul_modq %a_158, %b_161 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_174 = tile.matmul_modq %a_158, %b_162 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_175 = tile.matmul_modq %a_158, %b_163 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_176 = tile.matmul_modq %a_159, %b_160 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_177 = tile.matmul_modq %a_159, %b_161 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_178 = tile.matmul_modq %a_159, %b_162 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_179 = tile.matmul_modq %a_159, %b_163 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %C_180 = tile.add_modq %P_164 {q = 268434977 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_181 = tile.add_modq %P_165, %P_168 {q = 268434977 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_182 = tile.add_modq %P_166, %P_169, %P_172 {q = 268434977 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_183 = tile.add_modq %P_167, %P_170, %P_173, %P_176 {q = 268434977 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_184 = tile.add_modq %P_171, %P_174, %P_177 {q = 268434977 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_185 = tile.add_modq %P_175, %P_178 {q = 268434977 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_186 = tile.add_modq %P_179 {q = 268434977 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %out_fa51 = tile.recombine_modq %C_180, %C_181, %C_182, %C_183, %C_184, %C_185, %C_186 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %yc_fa52, %yc_fa53, %yc_fa54 = tile.split %out_fa51 {axis = 1 : i64, parts = 3 : i64} : (tensor<16x3xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %e0_2 = tile.reshape %yc_fa52 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e2_2 = tile.reshape %yc_fa53 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e1_2 = tile.reshape %yc_fa54 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %W_f_268433569 = tile.const {name = "W_f_268433569", q = 268433569 : i64} : tensor<16x16xi32>
    %col_fa55 = tile.reshape %c03 {shape = [16, 1], q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa56 = tile.reshape %c13 {shape = [16, 1], q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa57 = tile.reshape %d03 {shape = [16, 1], q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa58 = tile.reshape %d13 {shape = [16, 1], q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa59 = tile.concat %col_fa55, %col_fa56, %col_fa57, %col_fa58 {axis = 1 : i64, q = 268433569 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x4xi32>
    %a_187, %a_188, %a_189, %a_190 = tile.decompose %W_f_268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_191, %b_192, %b_193, %b_194 = tile.decompose %rhs_fa59 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>)
    %P_195 = tile.matmul_modq %a_187, %b_191 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_196 = tile.matmul_modq %a_187, %b_192 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_197 = tile.matmul_modq %a_187, %b_193 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_198 = tile.matmul_modq %a_187, %b_194 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_199 = tile.matmul_modq %a_188, %b_191 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_200 = tile.matmul_modq %a_188, %b_192 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_201 = tile.matmul_modq %a_188, %b_193 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_202 = tile.matmul_modq %a_188, %b_194 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_203 = tile.matmul_modq %a_189, %b_191 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_204 = tile.matmul_modq %a_189, %b_192 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_205 = tile.matmul_modq %a_189, %b_193 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_206 = tile.matmul_modq %a_189, %b_194 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_207 = tile.matmul_modq %a_190, %b_191 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_208 = tile.matmul_modq %a_190, %b_192 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_209 = tile.matmul_modq %a_190, %b_193 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_210 = tile.matmul_modq %a_190, %b_194 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %C_211 = tile.add_modq %P_195 {q = 268433569 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_212 = tile.add_modq %P_196, %P_199 {q = 268433569 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_213 = tile.add_modq %P_197, %P_200, %P_203 {q = 268433569 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_214 = tile.add_modq %P_198, %P_201, %P_204, %P_207 {q = 268433569 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_215 = tile.add_modq %P_202, %P_205, %P_208 {q = 268433569 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_216 = tile.add_modq %P_206, %P_209 {q = 268433569 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_217 = tile.add_modq %P_210 {q = 268433569 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %out_fa60 = tile.recombine_modq %C_211, %C_212, %C_213, %C_214, %C_215, %C_216, %C_217 {radix = 128 : i64, q = 268433569 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %yc_fa61, %yc_fa62, %yc_fa63, %yc_fa64 = tile.split %out_fa60 {axis = 1 : i64, parts = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %tC03 = tile.reshape %yc_fa61 {shape = [16], q = 268433569 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tC13 = tile.reshape %yc_fa62 {shape = [16], q = 268433569 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD03 = tile.reshape %yc_fa63 {shape = [16], q = 268433569 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD13 = tile.reshape %yc_fa64 {shape = [16], q = 268433569 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tE03 = tile.mul_modq %tC03, %tD03 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE23 = tile.mul_modq %tC13, %tD13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT03 = tile.mul_modq %tC03, %tD13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT13 = tile.mul_modq %tC13, %tD03 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE13 = tile.add_modq %tT03, %tT13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433569 = tile.const {name = "W_i_268433569", q = 268433569 : i64} : tensor<16x16xi32>
    %col_fa65 = tile.reshape %tE03 {shape = [16, 1], q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa66 = tile.reshape %tE23 {shape = [16, 1], q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa67 = tile.reshape %tE13 {shape = [16, 1], q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa68 = tile.concat %col_fa65, %col_fa66, %col_fa67 {axis = 1 : i64, q = 268433569 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x3xi32>
    %a_218, %a_219, %a_220, %a_221 = tile.decompose %W_i_268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_222, %b_223, %b_224, %b_225 = tile.decompose %rhs_fa68 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x3xi32>) -> (tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>)
    %P_226 = tile.matmul_modq %a_218, %b_222 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_227 = tile.matmul_modq %a_218, %b_223 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_228 = tile.matmul_modq %a_218, %b_224 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_229 = tile.matmul_modq %a_218, %b_225 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_230 = tile.matmul_modq %a_219, %b_222 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_231 = tile.matmul_modq %a_219, %b_223 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_232 = tile.matmul_modq %a_219, %b_224 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_233 = tile.matmul_modq %a_219, %b_225 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_234 = tile.matmul_modq %a_220, %b_222 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_235 = tile.matmul_modq %a_220, %b_223 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_236 = tile.matmul_modq %a_220, %b_224 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_237 = tile.matmul_modq %a_220, %b_225 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_238 = tile.matmul_modq %a_221, %b_222 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_239 = tile.matmul_modq %a_221, %b_223 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_240 = tile.matmul_modq %a_221, %b_224 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_241 = tile.matmul_modq %a_221, %b_225 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %C_242 = tile.add_modq %P_226 {q = 268433569 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_243 = tile.add_modq %P_227, %P_230 {q = 268433569 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_244 = tile.add_modq %P_228, %P_231, %P_234 {q = 268433569 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_245 = tile.add_modq %P_229, %P_232, %P_235, %P_238 {q = 268433569 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_246 = tile.add_modq %P_233, %P_236, %P_239 {q = 268433569 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_247 = tile.add_modq %P_237, %P_240 {q = 268433569 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_248 = tile.add_modq %P_241 {q = 268433569 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %out_fa69 = tile.recombine_modq %C_242, %C_243, %C_244, %C_245, %C_246, %C_247, %C_248 {radix = 128 : i64, q = 268433569 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %yc_fa70, %yc_fa71, %yc_fa72 = tile.split %out_fa69 {axis = 1 : i64, parts = 3 : i64} : (tensor<16x3xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %e0_3 = tile.reshape %yc_fa70 {shape = [16], q = 268433569 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e2_3 = tile.reshape %yc_fa71 {shape = [16], q = 268433569 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e1_3 = tile.reshape %yc_fa72 {shape = [16], q = 268433569 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %W_f_268433441 = tile.const {name = "W_f_268433441", q = 268433441 : i64} : tensor<16x16xi32>
    %col_fa73 = tile.reshape %c04 {shape = [16, 1], q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa74 = tile.reshape %c14 {shape = [16, 1], q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa75 = tile.reshape %d04 {shape = [16, 1], q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa76 = tile.reshape %d14 {shape = [16, 1], q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa77 = tile.concat %col_fa73, %col_fa74, %col_fa75, %col_fa76 {axis = 1 : i64, q = 268433441 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x4xi32>
    %a_249, %a_250, %a_251, %a_252 = tile.decompose %W_f_268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_253, %b_254, %b_255, %b_256 = tile.decompose %rhs_fa77 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>)
    %P_257 = tile.matmul_modq %a_249, %b_253 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_258 = tile.matmul_modq %a_249, %b_254 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_259 = tile.matmul_modq %a_249, %b_255 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_260 = tile.matmul_modq %a_249, %b_256 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_261 = tile.matmul_modq %a_250, %b_253 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_262 = tile.matmul_modq %a_250, %b_254 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_263 = tile.matmul_modq %a_250, %b_255 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_264 = tile.matmul_modq %a_250, %b_256 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_265 = tile.matmul_modq %a_251, %b_253 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_266 = tile.matmul_modq %a_251, %b_254 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_267 = tile.matmul_modq %a_251, %b_255 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_268 = tile.matmul_modq %a_251, %b_256 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_269 = tile.matmul_modq %a_252, %b_253 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_270 = tile.matmul_modq %a_252, %b_254 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_271 = tile.matmul_modq %a_252, %b_255 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_272 = tile.matmul_modq %a_252, %b_256 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %C_273 = tile.add_modq %P_257 {q = 268433441 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_274 = tile.add_modq %P_258, %P_261 {q = 268433441 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_275 = tile.add_modq %P_259, %P_262, %P_265 {q = 268433441 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_276 = tile.add_modq %P_260, %P_263, %P_266, %P_269 {q = 268433441 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_277 = tile.add_modq %P_264, %P_267, %P_270 {q = 268433441 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_278 = tile.add_modq %P_268, %P_271 {q = 268433441 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_279 = tile.add_modq %P_272 {q = 268433441 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %out_fa78 = tile.recombine_modq %C_273, %C_274, %C_275, %C_276, %C_277, %C_278, %C_279 {radix = 128 : i64, q = 268433441 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %yc_fa79, %yc_fa80, %yc_fa81, %yc_fa82 = tile.split %out_fa78 {axis = 1 : i64, parts = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %tC04 = tile.reshape %yc_fa79 {shape = [16], q = 268433441 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tC14 = tile.reshape %yc_fa80 {shape = [16], q = 268433441 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD04 = tile.reshape %yc_fa81 {shape = [16], q = 268433441 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD14 = tile.reshape %yc_fa82 {shape = [16], q = 268433441 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tE04 = tile.mul_modq %tC04, %tD04 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE24 = tile.mul_modq %tC14, %tD14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT04 = tile.mul_modq %tC04, %tD14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT14 = tile.mul_modq %tC14, %tD04 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE14 = tile.add_modq %tT04, %tT14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433441 = tile.const {name = "W_i_268433441", q = 268433441 : i64} : tensor<16x16xi32>
    %col_fa83 = tile.reshape %tE04 {shape = [16, 1], q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa84 = tile.reshape %tE24 {shape = [16, 1], q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa85 = tile.reshape %tE14 {shape = [16, 1], q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa86 = tile.concat %col_fa83, %col_fa84, %col_fa85 {axis = 1 : i64, q = 268433441 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x3xi32>
    %a_280, %a_281, %a_282, %a_283 = tile.decompose %W_i_268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_284, %b_285, %b_286, %b_287 = tile.decompose %rhs_fa86 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x3xi32>) -> (tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>)
    %P_288 = tile.matmul_modq %a_280, %b_284 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_289 = tile.matmul_modq %a_280, %b_285 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_290 = tile.matmul_modq %a_280, %b_286 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_291 = tile.matmul_modq %a_280, %b_287 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_292 = tile.matmul_modq %a_281, %b_284 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_293 = tile.matmul_modq %a_281, %b_285 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_294 = tile.matmul_modq %a_281, %b_286 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_295 = tile.matmul_modq %a_281, %b_287 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_296 = tile.matmul_modq %a_282, %b_284 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_297 = tile.matmul_modq %a_282, %b_285 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_298 = tile.matmul_modq %a_282, %b_286 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_299 = tile.matmul_modq %a_282, %b_287 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_300 = tile.matmul_modq %a_283, %b_284 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_301 = tile.matmul_modq %a_283, %b_285 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_302 = tile.matmul_modq %a_283, %b_286 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_303 = tile.matmul_modq %a_283, %b_287 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %C_304 = tile.add_modq %P_288 {q = 268433441 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_305 = tile.add_modq %P_289, %P_292 {q = 268433441 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_306 = tile.add_modq %P_290, %P_293, %P_296 {q = 268433441 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_307 = tile.add_modq %P_291, %P_294, %P_297, %P_300 {q = 268433441 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_308 = tile.add_modq %P_295, %P_298, %P_301 {q = 268433441 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_309 = tile.add_modq %P_299, %P_302 {q = 268433441 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_310 = tile.add_modq %P_303 {q = 268433441 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %out_fa87 = tile.recombine_modq %C_304, %C_305, %C_306, %C_307, %C_308, %C_309, %C_310 {radix = 128 : i64, q = 268433441 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %yc_fa88, %yc_fa89, %yc_fa90 = tile.split %out_fa87 {axis = 1 : i64, parts = 3 : i64} : (tensor<16x3xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %e0_4 = tile.reshape %yc_fa88 {shape = [16], q = 268433441 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e2_4 = tile.reshape %yc_fa89 {shape = [16], q = 268433441 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e1_4 = tile.reshape %yc_fa90 {shape = [16], q = 268433441 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %W_f_268433377 = tile.const {name = "W_f_268433377", q = 268433377 : i64} : tensor<16x16xi32>
    %col_fa91 = tile.reshape %c05 {shape = [16, 1], q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa92 = tile.reshape %c15 {shape = [16, 1], q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa93 = tile.reshape %d05 {shape = [16, 1], q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa94 = tile.reshape %d15 {shape = [16, 1], q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa95 = tile.concat %col_fa91, %col_fa92, %col_fa93, %col_fa94 {axis = 1 : i64, q = 268433377 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x4xi32>
    %a_311, %a_312, %a_313, %a_314 = tile.decompose %W_f_268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_315, %b_316, %b_317, %b_318 = tile.decompose %rhs_fa95 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>)
    %P_319 = tile.matmul_modq %a_311, %b_315 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_320 = tile.matmul_modq %a_311, %b_316 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_321 = tile.matmul_modq %a_311, %b_317 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_322 = tile.matmul_modq %a_311, %b_318 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_323 = tile.matmul_modq %a_312, %b_315 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_324 = tile.matmul_modq %a_312, %b_316 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_325 = tile.matmul_modq %a_312, %b_317 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_326 = tile.matmul_modq %a_312, %b_318 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_327 = tile.matmul_modq %a_313, %b_315 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_328 = tile.matmul_modq %a_313, %b_316 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_329 = tile.matmul_modq %a_313, %b_317 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_330 = tile.matmul_modq %a_313, %b_318 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_331 = tile.matmul_modq %a_314, %b_315 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_332 = tile.matmul_modq %a_314, %b_316 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_333 = tile.matmul_modq %a_314, %b_317 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_334 = tile.matmul_modq %a_314, %b_318 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %C_335 = tile.add_modq %P_319 {q = 268433377 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_336 = tile.add_modq %P_320, %P_323 {q = 268433377 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_337 = tile.add_modq %P_321, %P_324, %P_327 {q = 268433377 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_338 = tile.add_modq %P_322, %P_325, %P_328, %P_331 {q = 268433377 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_339 = tile.add_modq %P_326, %P_329, %P_332 {q = 268433377 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_340 = tile.add_modq %P_330, %P_333 {q = 268433377 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_341 = tile.add_modq %P_334 {q = 268433377 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %out_fa96 = tile.recombine_modq %C_335, %C_336, %C_337, %C_338, %C_339, %C_340, %C_341 {radix = 128 : i64, q = 268433377 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %yc_fa97, %yc_fa98, %yc_fa99, %yc_fa100 = tile.split %out_fa96 {axis = 1 : i64, parts = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %tC05 = tile.reshape %yc_fa97 {shape = [16], q = 268433377 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tC15 = tile.reshape %yc_fa98 {shape = [16], q = 268433377 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD05 = tile.reshape %yc_fa99 {shape = [16], q = 268433377 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD15 = tile.reshape %yc_fa100 {shape = [16], q = 268433377 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tE05 = tile.mul_modq %tC05, %tD05 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE25 = tile.mul_modq %tC15, %tD15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT05 = tile.mul_modq %tC05, %tD15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT15 = tile.mul_modq %tC15, %tD05 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE15 = tile.add_modq %tT05, %tT15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433377 = tile.const {name = "W_i_268433377", q = 268433377 : i64} : tensor<16x16xi32>
    %col_fa101 = tile.reshape %tE05 {shape = [16, 1], q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa102 = tile.reshape %tE25 {shape = [16, 1], q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa103 = tile.reshape %tE15 {shape = [16, 1], q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa104 = tile.concat %col_fa101, %col_fa102, %col_fa103 {axis = 1 : i64, q = 268433377 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x3xi32>
    %a_342, %a_343, %a_344, %a_345 = tile.decompose %W_i_268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_346, %b_347, %b_348, %b_349 = tile.decompose %rhs_fa104 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x3xi32>) -> (tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>)
    %P_350 = tile.matmul_modq %a_342, %b_346 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_351 = tile.matmul_modq %a_342, %b_347 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_352 = tile.matmul_modq %a_342, %b_348 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_353 = tile.matmul_modq %a_342, %b_349 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_354 = tile.matmul_modq %a_343, %b_346 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_355 = tile.matmul_modq %a_343, %b_347 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_356 = tile.matmul_modq %a_343, %b_348 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_357 = tile.matmul_modq %a_343, %b_349 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_358 = tile.matmul_modq %a_344, %b_346 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_359 = tile.matmul_modq %a_344, %b_347 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_360 = tile.matmul_modq %a_344, %b_348 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_361 = tile.matmul_modq %a_344, %b_349 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_362 = tile.matmul_modq %a_345, %b_346 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_363 = tile.matmul_modq %a_345, %b_347 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_364 = tile.matmul_modq %a_345, %b_348 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_365 = tile.matmul_modq %a_345, %b_349 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %C_366 = tile.add_modq %P_350 {q = 268433377 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_367 = tile.add_modq %P_351, %P_354 {q = 268433377 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_368 = tile.add_modq %P_352, %P_355, %P_358 {q = 268433377 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_369 = tile.add_modq %P_353, %P_356, %P_359, %P_362 {q = 268433377 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_370 = tile.add_modq %P_357, %P_360, %P_363 {q = 268433377 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_371 = tile.add_modq %P_361, %P_364 {q = 268433377 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_372 = tile.add_modq %P_365 {q = 268433377 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %out_fa105 = tile.recombine_modq %C_366, %C_367, %C_368, %C_369, %C_370, %C_371, %C_372 {radix = 128 : i64, q = 268433377 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %yc_fa106, %yc_fa107, %yc_fa108 = tile.split %out_fa105 {axis = 1 : i64, parts = 3 : i64} : (tensor<16x3xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %e0_5 = tile.reshape %yc_fa106 {shape = [16], q = 268433377 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e2_5 = tile.reshape %yc_fa107 {shape = [16], q = 268433377 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e1_5 = tile.reshape %yc_fa108 {shape = [16], q = 268433377 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %W_f_268433057 = tile.const {name = "W_f_268433057", q = 268433057 : i64} : tensor<16x16xi32>
    %col_fa109 = tile.reshape %c06 {shape = [16, 1], q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa110 = tile.reshape %c16 {shape = [16, 1], q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa111 = tile.reshape %d06 {shape = [16, 1], q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa112 = tile.reshape %d16 {shape = [16, 1], q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa113 = tile.concat %col_fa109, %col_fa110, %col_fa111, %col_fa112 {axis = 1 : i64, q = 268433057 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x4xi32>
    %a_373, %a_374, %a_375, %a_376 = tile.decompose %W_f_268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_377, %b_378, %b_379, %b_380 = tile.decompose %rhs_fa113 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>, tensor<16x4xi8>)
    %P_381 = tile.matmul_modq %a_373, %b_377 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_382 = tile.matmul_modq %a_373, %b_378 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_383 = tile.matmul_modq %a_373, %b_379 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_384 = tile.matmul_modq %a_373, %b_380 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_385 = tile.matmul_modq %a_374, %b_377 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_386 = tile.matmul_modq %a_374, %b_378 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_387 = tile.matmul_modq %a_374, %b_379 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_388 = tile.matmul_modq %a_374, %b_380 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_389 = tile.matmul_modq %a_375, %b_377 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_390 = tile.matmul_modq %a_375, %b_378 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_391 = tile.matmul_modq %a_375, %b_379 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_392 = tile.matmul_modq %a_375, %b_380 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_393 = tile.matmul_modq %a_376, %b_377 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_394 = tile.matmul_modq %a_376, %b_378 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_395 = tile.matmul_modq %a_376, %b_379 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %P_396 = tile.matmul_modq %a_376, %b_380 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x4xi8>) -> tensor<16x4xi32>
    %C_397 = tile.add_modq %P_381 {q = 268433057 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_398 = tile.add_modq %P_382, %P_385 {q = 268433057 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_399 = tile.add_modq %P_383, %P_386, %P_389 {q = 268433057 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_400 = tile.add_modq %P_384, %P_387, %P_390, %P_393 {q = 268433057 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_401 = tile.add_modq %P_388, %P_391, %P_394 {q = 268433057 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_402 = tile.add_modq %P_392, %P_395 {q = 268433057 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %C_403 = tile.add_modq %P_396 {q = 268433057 : i64} : (tensor<16x4xi32>) -> tensor<16x4xi32>
    %out_fa114 = tile.recombine_modq %C_397, %C_398, %C_399, %C_400, %C_401, %C_402, %C_403 {radix = 128 : i64, q = 268433057 : i64} : (tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>, tensor<16x4xi32>) -> tensor<16x4xi32>
    %yc_fa115, %yc_fa116, %yc_fa117, %yc_fa118 = tile.split %out_fa114 {axis = 1 : i64, parts = 4 : i64} : (tensor<16x4xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %tC06 = tile.reshape %yc_fa115 {shape = [16], q = 268433057 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tC16 = tile.reshape %yc_fa116 {shape = [16], q = 268433057 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD06 = tile.reshape %yc_fa117 {shape = [16], q = 268433057 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tD16 = tile.reshape %yc_fa118 {shape = [16], q = 268433057 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %tE06 = tile.mul_modq %tC06, %tD06 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE26 = tile.mul_modq %tC16, %tD16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT06 = tile.mul_modq %tC06, %tD16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT16 = tile.mul_modq %tC16, %tD06 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE16 = tile.add_modq %tT06, %tT16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433057 = tile.const {name = "W_i_268433057", q = 268433057 : i64} : tensor<16x16xi32>
    %col_fa119 = tile.reshape %tE06 {shape = [16, 1], q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa120 = tile.reshape %tE26 {shape = [16, 1], q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa121 = tile.reshape %tE16 {shape = [16, 1], q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa122 = tile.concat %col_fa119, %col_fa120, %col_fa121 {axis = 1 : i64, q = 268433057 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x3xi32>
    %a_404, %a_405, %a_406, %a_407 = tile.decompose %W_i_268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_408, %b_409, %b_410, %b_411 = tile.decompose %rhs_fa122 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x3xi32>) -> (tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>, tensor<16x3xi8>)
    %P_412 = tile.matmul_modq %a_404, %b_408 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_413 = tile.matmul_modq %a_404, %b_409 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_414 = tile.matmul_modq %a_404, %b_410 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_415 = tile.matmul_modq %a_404, %b_411 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_416 = tile.matmul_modq %a_405, %b_408 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_417 = tile.matmul_modq %a_405, %b_409 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_418 = tile.matmul_modq %a_405, %b_410 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_419 = tile.matmul_modq %a_405, %b_411 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_420 = tile.matmul_modq %a_406, %b_408 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_421 = tile.matmul_modq %a_406, %b_409 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_422 = tile.matmul_modq %a_406, %b_410 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_423 = tile.matmul_modq %a_406, %b_411 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_424 = tile.matmul_modq %a_407, %b_408 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_425 = tile.matmul_modq %a_407, %b_409 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_426 = tile.matmul_modq %a_407, %b_410 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %P_427 = tile.matmul_modq %a_407, %b_411 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16x3xi8>) -> tensor<16x3xi32>
    %C_428 = tile.add_modq %P_412 {q = 268433057 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_429 = tile.add_modq %P_413, %P_416 {q = 268433057 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_430 = tile.add_modq %P_414, %P_417, %P_420 {q = 268433057 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_431 = tile.add_modq %P_415, %P_418, %P_421, %P_424 {q = 268433057 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_432 = tile.add_modq %P_419, %P_422, %P_425 {q = 268433057 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_433 = tile.add_modq %P_423, %P_426 {q = 268433057 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %C_434 = tile.add_modq %P_427 {q = 268433057 : i64} : (tensor<16x3xi32>) -> tensor<16x3xi32>
    %out_fa123 = tile.recombine_modq %C_428, %C_429, %C_430, %C_431, %C_432, %C_433, %C_434 {radix = 128 : i64, q = 268433057 : i64} : (tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>, tensor<16x3xi32>) -> tensor<16x3xi32>
    %yc_fa124, %yc_fa125, %yc_fa126 = tile.split %out_fa123 {axis = 1 : i64, parts = 3 : i64} : (tensor<16x3xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>, tensor<16x1xi32>)
    %e0_6 = tile.reshape %yc_fa124 {shape = [16], q = 268433057 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e2_6 = tile.reshape %yc_fa125 {shape = [16], q = 268433057 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %e1_6 = tile.reshape %yc_fa126 {shape = [16], q = 268433057 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
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
    %a_435, %a_436, %a_437, %a_438 = tile.decompose %fbcD__268435361__268435009__268434977__268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_439, %b_440, %b_441, %b_442 = tile.decompose %rs0_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_443 = tile.matmul_modq %a_435, %b_439 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_444 = tile.matmul_modq %a_435, %b_440 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_445 = tile.matmul_modq %a_435, %b_441 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_446 = tile.matmul_modq %a_435, %b_442 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_447 = tile.matmul_modq %a_436, %b_439 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_448 = tile.matmul_modq %a_436, %b_440 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_449 = tile.matmul_modq %a_436, %b_441 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_450 = tile.matmul_modq %a_436, %b_442 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_451 = tile.matmul_modq %a_437, %b_439 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_452 = tile.matmul_modq %a_437, %b_440 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_453 = tile.matmul_modq %a_437, %b_441 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_454 = tile.matmul_modq %a_437, %b_442 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_455 = tile.matmul_modq %a_438, %b_439 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_456 = tile.matmul_modq %a_438, %b_440 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_457 = tile.matmul_modq %a_438, %b_441 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_458 = tile.matmul_modq %a_438, %b_442 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_459 = tile.add_modq %P_443 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_460 = tile.add_modq %P_444, %P_447 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_461 = tile.add_modq %P_445, %P_448, %P_451 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_462 = tile.add_modq %P_446, %P_449, %P_452, %P_455 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_463 = tile.add_modq %P_450, %P_453, %P_456 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_464 = tile.add_modq %P_454, %P_457 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_465 = tile.add_modq %P_458 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_fbc_268433569 = tile.recombine_modq %C_459, %C_460, %C_461, %C_462, %C_463, %C_464, %C_465 {radix = 128 : i64, q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0 = tile.reshape %rs0_fbc0_fbc_268433569 {shape = [16], q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433441 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433441"} : tensor<1x3xi32>
    %a_466, %a_467, %a_468, %a_469 = tile.decompose %fbcD__268435361__268435009__268434977__268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_470, %b_471, %b_472, %b_473 = tile.decompose %rs0_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_474 = tile.matmul_modq %a_466, %b_470 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_475 = tile.matmul_modq %a_466, %b_471 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_476 = tile.matmul_modq %a_466, %b_472 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_477 = tile.matmul_modq %a_466, %b_473 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_478 = tile.matmul_modq %a_467, %b_470 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_479 = tile.matmul_modq %a_467, %b_471 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_480 = tile.matmul_modq %a_467, %b_472 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_481 = tile.matmul_modq %a_467, %b_473 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_482 = tile.matmul_modq %a_468, %b_470 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_483 = tile.matmul_modq %a_468, %b_471 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_484 = tile.matmul_modq %a_468, %b_472 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_485 = tile.matmul_modq %a_468, %b_473 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_486 = tile.matmul_modq %a_469, %b_470 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_487 = tile.matmul_modq %a_469, %b_471 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_488 = tile.matmul_modq %a_469, %b_472 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_489 = tile.matmul_modq %a_469, %b_473 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_490 = tile.add_modq %P_474 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_491 = tile.add_modq %P_475, %P_478 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_492 = tile.add_modq %P_476, %P_479, %P_482 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_493 = tile.add_modq %P_477, %P_480, %P_483, %P_486 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_494 = tile.add_modq %P_481, %P_484, %P_487 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_495 = tile.add_modq %P_485, %P_488 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_496 = tile.add_modq %P_489 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_fbc_268433441 = tile.recombine_modq %C_490, %C_491, %C_492, %C_493, %C_494, %C_495, %C_496 {radix = 128 : i64, q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc1 = tile.reshape %rs0_fbc0_fbc_268433441 {shape = [16], q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433377 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433377"} : tensor<1x3xi32>
    %a_497, %a_498, %a_499, %a_500 = tile.decompose %fbcD__268435361__268435009__268434977__268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_501, %b_502, %b_503, %b_504 = tile.decompose %rs0_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_505 = tile.matmul_modq %a_497, %b_501 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_506 = tile.matmul_modq %a_497, %b_502 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_507 = tile.matmul_modq %a_497, %b_503 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_508 = tile.matmul_modq %a_497, %b_504 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_509 = tile.matmul_modq %a_498, %b_501 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_510 = tile.matmul_modq %a_498, %b_502 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_511 = tile.matmul_modq %a_498, %b_503 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_512 = tile.matmul_modq %a_498, %b_504 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_513 = tile.matmul_modq %a_499, %b_501 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_514 = tile.matmul_modq %a_499, %b_502 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_515 = tile.matmul_modq %a_499, %b_503 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_516 = tile.matmul_modq %a_499, %b_504 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_517 = tile.matmul_modq %a_500, %b_501 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_518 = tile.matmul_modq %a_500, %b_502 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_519 = tile.matmul_modq %a_500, %b_503 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_520 = tile.matmul_modq %a_500, %b_504 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_521 = tile.add_modq %P_505 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_522 = tile.add_modq %P_506, %P_509 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_523 = tile.add_modq %P_507, %P_510, %P_513 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_524 = tile.add_modq %P_508, %P_511, %P_514, %P_517 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_525 = tile.add_modq %P_512, %P_515, %P_518 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_526 = tile.add_modq %P_516, %P_519 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_527 = tile.add_modq %P_520 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_fbc_268433377 = tile.recombine_modq %C_521, %C_522, %C_523, %C_524, %C_525, %C_526, %C_527 {radix = 128 : i64, q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc2 = tile.reshape %rs0_fbc0_fbc_268433377 {shape = [16], q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433057 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433057"} : tensor<1x3xi32>
    %a_528, %a_529, %a_530, %a_531 = tile.decompose %fbcD__268435361__268435009__268434977__268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_532, %b_533, %b_534, %b_535 = tile.decompose %rs0_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_536 = tile.matmul_modq %a_528, %b_532 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_537 = tile.matmul_modq %a_528, %b_533 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_538 = tile.matmul_modq %a_528, %b_534 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_539 = tile.matmul_modq %a_528, %b_535 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_540 = tile.matmul_modq %a_529, %b_532 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_541 = tile.matmul_modq %a_529, %b_533 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_542 = tile.matmul_modq %a_529, %b_534 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_543 = tile.matmul_modq %a_529, %b_535 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_544 = tile.matmul_modq %a_530, %b_532 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_545 = tile.matmul_modq %a_530, %b_533 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_546 = tile.matmul_modq %a_530, %b_534 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_547 = tile.matmul_modq %a_530, %b_535 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_548 = tile.matmul_modq %a_531, %b_532 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_549 = tile.matmul_modq %a_531, %b_533 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_550 = tile.matmul_modq %a_531, %b_534 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_551 = tile.matmul_modq %a_531, %b_535 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_552 = tile.add_modq %P_536 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_553 = tile.add_modq %P_537, %P_540 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_554 = tile.add_modq %P_538, %P_541, %P_544 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_555 = tile.add_modq %P_539, %P_542, %P_545, %P_548 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_556 = tile.add_modq %P_543, %P_546, %P_549 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_557 = tile.add_modq %P_547, %P_550 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_558 = tile.add_modq %P_551 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_fbc_268433057 = tile.recombine_modq %C_552, %C_553, %C_554, %C_555, %C_556, %C_557, %C_558 {radix = 128 : i64, q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_559, %a_560, %a_561, %a_562, %a_563, %a_564 = tile.decompose %crnsF__268433569__268433441__268433377__268433057__38 {radix = 128 : i64, nlimbs = 6 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_565, %b_566, %b_567, %b_568 = tile.decompose %rs0_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %wP_569 = tile.matmul_wide %a_559, %b_565 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_570 = tile.matmul_wide %a_559, %b_566 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_571 = tile.matmul_wide %a_559, %b_567 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_572 = tile.matmul_wide %a_559, %b_568 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_573 = tile.matmul_wide %a_560, %b_565 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_574 = tile.matmul_wide %a_560, %b_566 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_575 = tile.matmul_wide %a_560, %b_567 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_576 = tile.matmul_wide %a_560, %b_568 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_577 = tile.matmul_wide %a_561, %b_565 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_578 = tile.matmul_wide %a_561, %b_566 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_579 = tile.matmul_wide %a_561, %b_567 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_580 = tile.matmul_wide %a_561, %b_568 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_581 = tile.matmul_wide %a_562, %b_565 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_582 = tile.matmul_wide %a_562, %b_566 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_583 = tile.matmul_wide %a_562, %b_567 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_584 = tile.matmul_wide %a_562, %b_568 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_585 = tile.matmul_wide %a_563, %b_565 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_586 = tile.matmul_wide %a_563, %b_566 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_587 = tile.matmul_wide %a_563, %b_567 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_588 = tile.matmul_wide %a_563, %b_568 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_589 = tile.matmul_wide %a_564, %b_565 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_590 = tile.matmul_wide %a_564, %b_566 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_591 = tile.matmul_wide %a_564, %b_567 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_592 = tile.matmul_wide %a_564, %b_568 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %rs0_etQ0_s = tile.recombine_wide %wP_569, %wP_570, %wP_571, %wP_572, %wP_573, %wP_574, %wP_575, %wP_576, %wP_577, %wP_578, %wP_579, %wP_580, %wP_581, %wP_582, %wP_583, %wP_584, %wP_585, %wP_586, %wP_587, %wP_588, %wP_589, %wP_590, %wP_591, %wP_592 {radix = 128 : i64, exps = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8]} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_k = tile.fixed_slice %rs0_etQ0_s {shift = 38 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_neg = tile.fixed_slice %rs0_etQ0_s {shift = 37 : i64, mask = 1 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsE__268433569__268433441__268433377__268433057__268435361 = tile.const {name = "crnsE_(268433569, 268433441, 268433377, 268433057)_268435361"} : tensor<1x4xi32>
    %a_593, %a_594, %a_595, %a_596 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_597, %b_598, %b_599, %b_600 = tile.decompose %rs0_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_601 = tile.matmul_modq %a_593, %b_597 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_602 = tile.matmul_modq %a_593, %b_598 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_603 = tile.matmul_modq %a_593, %b_599 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_604 = tile.matmul_modq %a_593, %b_600 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_605 = tile.matmul_modq %a_594, %b_597 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_606 = tile.matmul_modq %a_594, %b_598 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_607 = tile.matmul_modq %a_594, %b_599 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_608 = tile.matmul_modq %a_594, %b_600 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_609 = tile.matmul_modq %a_595, %b_597 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_610 = tile.matmul_modq %a_595, %b_598 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_611 = tile.matmul_modq %a_595, %b_599 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_612 = tile.matmul_modq %a_595, %b_600 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_613 = tile.matmul_modq %a_596, %b_597 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_614 = tile.matmul_modq %a_596, %b_598 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_615 = tile.matmul_modq %a_596, %b_599 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_616 = tile.matmul_modq %a_596, %b_600 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_617 = tile.add_modq %P_601 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_618 = tile.add_modq %P_602, %P_605 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_619 = tile.add_modq %P_603, %P_606, %P_609 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_620 = tile.add_modq %P_604, %P_607, %P_610, %P_613 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_621 = tile.add_modq %P_608, %P_611, %P_614 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_622 = tile.add_modq %P_612, %P_615 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_623 = tile.add_modq %P_616 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_xE_268435361 = tile.recombine_modq %C_617, %C_618, %C_619, %C_620, %C_621, %C_622, %C_623 {radix = 128 : i64, q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsg__268433569__268433441__268433377__268433057__268435361 = tile.const {name = "crnsg_(268433569, 268433441, 268433377, 268433057)_268435361", q = 268435361 : i64} : tensor<1x1xi32>
    %rs0_etQ0_kg_268435361 = tile.mul_modq %rs0_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_acc_268435361 = tile.add_modq %rs0_etQ0_xE_268435361, %rs0_etQ0_kg_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsMmod__268433569__268433441__268433377__268433057__268435361 = tile.const {name = "crnsMmod_(268433569, 268433441, 268433377, 268433057)_268435361", q = 268435361 : i64} : tensor<1x1xi32>
    %rs0_etQ0_negM_268435361 = tile.mul_modq %rs0_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_accs_268435361 = tile.sub_modq %rs0_etQ0_acc_268435361, %rs0_etQ0_negM_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0 = tile.reshape %rs0_etQ0_accs_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %crnsE__268433569__268433441__268433377__268433057__268435009 = tile.const {name = "crnsE_(268433569, 268433441, 268433377, 268433057)_268435009"} : tensor<1x4xi32>
    %a_624, %a_625, %a_626, %a_627 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_628, %b_629, %b_630, %b_631 = tile.decompose %rs0_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_632 = tile.matmul_modq %a_624, %b_628 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_633 = tile.matmul_modq %a_624, %b_629 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_634 = tile.matmul_modq %a_624, %b_630 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_635 = tile.matmul_modq %a_624, %b_631 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_636 = tile.matmul_modq %a_625, %b_628 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_637 = tile.matmul_modq %a_625, %b_629 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_638 = tile.matmul_modq %a_625, %b_630 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_639 = tile.matmul_modq %a_625, %b_631 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_640 = tile.matmul_modq %a_626, %b_628 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_641 = tile.matmul_modq %a_626, %b_629 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_642 = tile.matmul_modq %a_626, %b_630 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_643 = tile.matmul_modq %a_626, %b_631 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_644 = tile.matmul_modq %a_627, %b_628 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_645 = tile.matmul_modq %a_627, %b_629 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_646 = tile.matmul_modq %a_627, %b_630 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_647 = tile.matmul_modq %a_627, %b_631 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_648 = tile.add_modq %P_632 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_649 = tile.add_modq %P_633, %P_636 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_650 = tile.add_modq %P_634, %P_637, %P_640 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_651 = tile.add_modq %P_635, %P_638, %P_641, %P_644 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_652 = tile.add_modq %P_639, %P_642, %P_645 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_653 = tile.add_modq %P_643, %P_646 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_654 = tile.add_modq %P_647 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_xE_268435009 = tile.recombine_modq %C_648, %C_649, %C_650, %C_651, %C_652, %C_653, %C_654 {radix = 128 : i64, q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsg__268433569__268433441__268433377__268433057__268435009 = tile.const {name = "crnsg_(268433569, 268433441, 268433377, 268433057)_268435009", q = 268435009 : i64} : tensor<1x1xi32>
    %rs0_etQ0_kg_268435009 = tile.mul_modq %rs0_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_acc_268435009 = tile.add_modq %rs0_etQ0_xE_268435009, %rs0_etQ0_kg_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsMmod__268433569__268433441__268433377__268433057__268435009 = tile.const {name = "crnsMmod_(268433569, 268433441, 268433377, 268433057)_268435009", q = 268435009 : i64} : tensor<1x1xi32>
    %rs0_etQ0_negM_268435009 = tile.mul_modq %rs0_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_accs_268435009 = tile.sub_modq %rs0_etQ0_acc_268435009, %rs0_etQ0_negM_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ1 = tile.reshape %rs0_etQ0_accs_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %crnsE__268433569__268433441__268433377__268433057__268434977 = tile.const {name = "crnsE_(268433569, 268433441, 268433377, 268433057)_268434977"} : tensor<1x4xi32>
    %a_655, %a_656, %a_657, %a_658 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_659, %b_660, %b_661, %b_662 = tile.decompose %rs0_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_663 = tile.matmul_modq %a_655, %b_659 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_664 = tile.matmul_modq %a_655, %b_660 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_665 = tile.matmul_modq %a_655, %b_661 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_666 = tile.matmul_modq %a_655, %b_662 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_667 = tile.matmul_modq %a_656, %b_659 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_668 = tile.matmul_modq %a_656, %b_660 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_669 = tile.matmul_modq %a_656, %b_661 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_670 = tile.matmul_modq %a_656, %b_662 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_671 = tile.matmul_modq %a_657, %b_659 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_672 = tile.matmul_modq %a_657, %b_660 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_673 = tile.matmul_modq %a_657, %b_661 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_674 = tile.matmul_modq %a_657, %b_662 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_675 = tile.matmul_modq %a_658, %b_659 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_676 = tile.matmul_modq %a_658, %b_660 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_677 = tile.matmul_modq %a_658, %b_661 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_678 = tile.matmul_modq %a_658, %b_662 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_679 = tile.add_modq %P_663 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_680 = tile.add_modq %P_664, %P_667 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_681 = tile.add_modq %P_665, %P_668, %P_671 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_682 = tile.add_modq %P_666, %P_669, %P_672, %P_675 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_683 = tile.add_modq %P_670, %P_673, %P_676 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_684 = tile.add_modq %P_674, %P_677 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_685 = tile.add_modq %P_678 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_xE_268434977 = tile.recombine_modq %C_679, %C_680, %C_681, %C_682, %C_683, %C_684, %C_685 {radix = 128 : i64, q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_686, %a_687, %a_688, %a_689 = tile.decompose %fbcD__268435361__268435009__268434977__268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_690, %b_691, %b_692, %b_693 = tile.decompose %rs1_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_694 = tile.matmul_modq %a_686, %b_690 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_695 = tile.matmul_modq %a_686, %b_691 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_696 = tile.matmul_modq %a_686, %b_692 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_697 = tile.matmul_modq %a_686, %b_693 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_698 = tile.matmul_modq %a_687, %b_690 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_699 = tile.matmul_modq %a_687, %b_691 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_700 = tile.matmul_modq %a_687, %b_692 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_701 = tile.matmul_modq %a_687, %b_693 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_702 = tile.matmul_modq %a_688, %b_690 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_703 = tile.matmul_modq %a_688, %b_691 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_704 = tile.matmul_modq %a_688, %b_692 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_705 = tile.matmul_modq %a_688, %b_693 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_706 = tile.matmul_modq %a_689, %b_690 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_707 = tile.matmul_modq %a_689, %b_691 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_708 = tile.matmul_modq %a_689, %b_692 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_709 = tile.matmul_modq %a_689, %b_693 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_710 = tile.add_modq %P_694 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_711 = tile.add_modq %P_695, %P_698 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_712 = tile.add_modq %P_696, %P_699, %P_702 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_713 = tile.add_modq %P_697, %P_700, %P_703, %P_706 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_714 = tile.add_modq %P_701, %P_704, %P_707 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_715 = tile.add_modq %P_705, %P_708 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_716 = tile.add_modq %P_709 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_fbc_268433569 = tile.recombine_modq %C_710, %C_711, %C_712, %C_713, %C_714, %C_715, %C_716 {radix = 128 : i64, q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0 = tile.reshape %rs1_fbc0_fbc_268433569 {shape = [16], q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_717, %a_718, %a_719, %a_720 = tile.decompose %fbcD__268435361__268435009__268434977__268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_721, %b_722, %b_723, %b_724 = tile.decompose %rs1_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_725 = tile.matmul_modq %a_717, %b_721 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_726 = tile.matmul_modq %a_717, %b_722 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_727 = tile.matmul_modq %a_717, %b_723 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_728 = tile.matmul_modq %a_717, %b_724 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_729 = tile.matmul_modq %a_718, %b_721 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_730 = tile.matmul_modq %a_718, %b_722 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_731 = tile.matmul_modq %a_718, %b_723 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_732 = tile.matmul_modq %a_718, %b_724 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_733 = tile.matmul_modq %a_719, %b_721 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_734 = tile.matmul_modq %a_719, %b_722 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_735 = tile.matmul_modq %a_719, %b_723 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_736 = tile.matmul_modq %a_719, %b_724 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_737 = tile.matmul_modq %a_720, %b_721 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_738 = tile.matmul_modq %a_720, %b_722 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_739 = tile.matmul_modq %a_720, %b_723 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_740 = tile.matmul_modq %a_720, %b_724 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_741 = tile.add_modq %P_725 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_742 = tile.add_modq %P_726, %P_729 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_743 = tile.add_modq %P_727, %P_730, %P_733 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_744 = tile.add_modq %P_728, %P_731, %P_734, %P_737 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_745 = tile.add_modq %P_732, %P_735, %P_738 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_746 = tile.add_modq %P_736, %P_739 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_747 = tile.add_modq %P_740 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_fbc_268433441 = tile.recombine_modq %C_741, %C_742, %C_743, %C_744, %C_745, %C_746, %C_747 {radix = 128 : i64, q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc1 = tile.reshape %rs1_fbc0_fbc_268433441 {shape = [16], q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_748, %a_749, %a_750, %a_751 = tile.decompose %fbcD__268435361__268435009__268434977__268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_752, %b_753, %b_754, %b_755 = tile.decompose %rs1_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_756 = tile.matmul_modq %a_748, %b_752 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_757 = tile.matmul_modq %a_748, %b_753 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_758 = tile.matmul_modq %a_748, %b_754 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_759 = tile.matmul_modq %a_748, %b_755 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_760 = tile.matmul_modq %a_749, %b_752 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_761 = tile.matmul_modq %a_749, %b_753 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_762 = tile.matmul_modq %a_749, %b_754 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_763 = tile.matmul_modq %a_749, %b_755 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_764 = tile.matmul_modq %a_750, %b_752 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_765 = tile.matmul_modq %a_750, %b_753 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_766 = tile.matmul_modq %a_750, %b_754 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_767 = tile.matmul_modq %a_750, %b_755 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_768 = tile.matmul_modq %a_751, %b_752 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_769 = tile.matmul_modq %a_751, %b_753 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_770 = tile.matmul_modq %a_751, %b_754 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_771 = tile.matmul_modq %a_751, %b_755 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_772 = tile.add_modq %P_756 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_773 = tile.add_modq %P_757, %P_760 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_774 = tile.add_modq %P_758, %P_761, %P_764 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_775 = tile.add_modq %P_759, %P_762, %P_765, %P_768 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_776 = tile.add_modq %P_763, %P_766, %P_769 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_777 = tile.add_modq %P_767, %P_770 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_778 = tile.add_modq %P_771 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_fbc_268433377 = tile.recombine_modq %C_772, %C_773, %C_774, %C_775, %C_776, %C_777, %C_778 {radix = 128 : i64, q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc2 = tile.reshape %rs1_fbc0_fbc_268433377 {shape = [16], q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_779, %a_780, %a_781, %a_782 = tile.decompose %fbcD__268435361__268435009__268434977__268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_783, %b_784, %b_785, %b_786 = tile.decompose %rs1_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_787 = tile.matmul_modq %a_779, %b_783 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_788 = tile.matmul_modq %a_779, %b_784 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_789 = tile.matmul_modq %a_779, %b_785 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_790 = tile.matmul_modq %a_779, %b_786 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_791 = tile.matmul_modq %a_780, %b_783 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_792 = tile.matmul_modq %a_780, %b_784 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_793 = tile.matmul_modq %a_780, %b_785 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_794 = tile.matmul_modq %a_780, %b_786 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_795 = tile.matmul_modq %a_781, %b_783 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_796 = tile.matmul_modq %a_781, %b_784 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_797 = tile.matmul_modq %a_781, %b_785 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_798 = tile.matmul_modq %a_781, %b_786 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_799 = tile.matmul_modq %a_782, %b_783 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_800 = tile.matmul_modq %a_782, %b_784 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_801 = tile.matmul_modq %a_782, %b_785 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_802 = tile.matmul_modq %a_782, %b_786 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_803 = tile.add_modq %P_787 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_804 = tile.add_modq %P_788, %P_791 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_805 = tile.add_modq %P_789, %P_792, %P_795 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_806 = tile.add_modq %P_790, %P_793, %P_796, %P_799 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_807 = tile.add_modq %P_794, %P_797, %P_800 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_808 = tile.add_modq %P_798, %P_801 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_809 = tile.add_modq %P_802 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_fbc_268433057 = tile.recombine_modq %C_803, %C_804, %C_805, %C_806, %C_807, %C_808, %C_809 {radix = 128 : i64, q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_810, %a_811, %a_812, %a_813, %a_814, %a_815 = tile.decompose %crnsF__268433569__268433441__268433377__268433057__38 {radix = 128 : i64, nlimbs = 6 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_816, %b_817, %b_818, %b_819 = tile.decompose %rs1_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %wP_820 = tile.matmul_wide %a_810, %b_816 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_821 = tile.matmul_wide %a_810, %b_817 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_822 = tile.matmul_wide %a_810, %b_818 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_823 = tile.matmul_wide %a_810, %b_819 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_824 = tile.matmul_wide %a_811, %b_816 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_825 = tile.matmul_wide %a_811, %b_817 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_826 = tile.matmul_wide %a_811, %b_818 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_827 = tile.matmul_wide %a_811, %b_819 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_828 = tile.matmul_wide %a_812, %b_816 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_829 = tile.matmul_wide %a_812, %b_817 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_830 = tile.matmul_wide %a_812, %b_818 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_831 = tile.matmul_wide %a_812, %b_819 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_832 = tile.matmul_wide %a_813, %b_816 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_833 = tile.matmul_wide %a_813, %b_817 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_834 = tile.matmul_wide %a_813, %b_818 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_835 = tile.matmul_wide %a_813, %b_819 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_836 = tile.matmul_wide %a_814, %b_816 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_837 = tile.matmul_wide %a_814, %b_817 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_838 = tile.matmul_wide %a_814, %b_818 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_839 = tile.matmul_wide %a_814, %b_819 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_840 = tile.matmul_wide %a_815, %b_816 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_841 = tile.matmul_wide %a_815, %b_817 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_842 = tile.matmul_wide %a_815, %b_818 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_843 = tile.matmul_wide %a_815, %b_819 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %rs1_etQ0_s = tile.recombine_wide %wP_820, %wP_821, %wP_822, %wP_823, %wP_824, %wP_825, %wP_826, %wP_827, %wP_828, %wP_829, %wP_830, %wP_831, %wP_832, %wP_833, %wP_834, %wP_835, %wP_836, %wP_837, %wP_838, %wP_839, %wP_840, %wP_841, %wP_842, %wP_843 {radix = 128 : i64, exps = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8]} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_k = tile.fixed_slice %rs1_etQ0_s {shift = 38 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_neg = tile.fixed_slice %rs1_etQ0_s {shift = 37 : i64, mask = 1 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %a_844, %a_845, %a_846, %a_847 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_848, %b_849, %b_850, %b_851 = tile.decompose %rs1_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_852 = tile.matmul_modq %a_844, %b_848 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_853 = tile.matmul_modq %a_844, %b_849 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_854 = tile.matmul_modq %a_844, %b_850 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_855 = tile.matmul_modq %a_844, %b_851 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_856 = tile.matmul_modq %a_845, %b_848 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_857 = tile.matmul_modq %a_845, %b_849 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_858 = tile.matmul_modq %a_845, %b_850 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_859 = tile.matmul_modq %a_845, %b_851 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_860 = tile.matmul_modq %a_846, %b_848 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_861 = tile.matmul_modq %a_846, %b_849 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_862 = tile.matmul_modq %a_846, %b_850 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_863 = tile.matmul_modq %a_846, %b_851 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_864 = tile.matmul_modq %a_847, %b_848 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_865 = tile.matmul_modq %a_847, %b_849 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_866 = tile.matmul_modq %a_847, %b_850 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_867 = tile.matmul_modq %a_847, %b_851 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_868 = tile.add_modq %P_852 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_869 = tile.add_modq %P_853, %P_856 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_870 = tile.add_modq %P_854, %P_857, %P_860 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_871 = tile.add_modq %P_855, %P_858, %P_861, %P_864 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_872 = tile.add_modq %P_859, %P_862, %P_865 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_873 = tile.add_modq %P_863, %P_866 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_874 = tile.add_modq %P_867 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_xE_268435361 = tile.recombine_modq %C_868, %C_869, %C_870, %C_871, %C_872, %C_873, %C_874 {radix = 128 : i64, q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_kg_268435361 = tile.mul_modq %rs1_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_acc_268435361 = tile.add_modq %rs1_etQ0_xE_268435361, %rs1_etQ0_kg_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_negM_268435361 = tile.mul_modq %rs1_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_accs_268435361 = tile.sub_modq %rs1_etQ0_acc_268435361, %rs1_etQ0_negM_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0 = tile.reshape %rs1_etQ0_accs_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_875, %a_876, %a_877, %a_878 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_879, %b_880, %b_881, %b_882 = tile.decompose %rs1_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_883 = tile.matmul_modq %a_875, %b_879 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_884 = tile.matmul_modq %a_875, %b_880 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_885 = tile.matmul_modq %a_875, %b_881 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_886 = tile.matmul_modq %a_875, %b_882 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_887 = tile.matmul_modq %a_876, %b_879 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_888 = tile.matmul_modq %a_876, %b_880 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_889 = tile.matmul_modq %a_876, %b_881 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_890 = tile.matmul_modq %a_876, %b_882 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_891 = tile.matmul_modq %a_877, %b_879 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_892 = tile.matmul_modq %a_877, %b_880 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_893 = tile.matmul_modq %a_877, %b_881 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_894 = tile.matmul_modq %a_877, %b_882 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_895 = tile.matmul_modq %a_878, %b_879 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_896 = tile.matmul_modq %a_878, %b_880 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_897 = tile.matmul_modq %a_878, %b_881 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_898 = tile.matmul_modq %a_878, %b_882 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_899 = tile.add_modq %P_883 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_900 = tile.add_modq %P_884, %P_887 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_901 = tile.add_modq %P_885, %P_888, %P_891 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_902 = tile.add_modq %P_886, %P_889, %P_892, %P_895 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_903 = tile.add_modq %P_890, %P_893, %P_896 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_904 = tile.add_modq %P_894, %P_897 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_905 = tile.add_modq %P_898 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_xE_268435009 = tile.recombine_modq %C_899, %C_900, %C_901, %C_902, %C_903, %C_904, %C_905 {radix = 128 : i64, q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_kg_268435009 = tile.mul_modq %rs1_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_acc_268435009 = tile.add_modq %rs1_etQ0_xE_268435009, %rs1_etQ0_kg_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_negM_268435009 = tile.mul_modq %rs1_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_accs_268435009 = tile.sub_modq %rs1_etQ0_acc_268435009, %rs1_etQ0_negM_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ1 = tile.reshape %rs1_etQ0_accs_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_906, %a_907, %a_908, %a_909 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_910, %b_911, %b_912, %b_913 = tile.decompose %rs1_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_914 = tile.matmul_modq %a_906, %b_910 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_915 = tile.matmul_modq %a_906, %b_911 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_916 = tile.matmul_modq %a_906, %b_912 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_917 = tile.matmul_modq %a_906, %b_913 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_918 = tile.matmul_modq %a_907, %b_910 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_919 = tile.matmul_modq %a_907, %b_911 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_920 = tile.matmul_modq %a_907, %b_912 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_921 = tile.matmul_modq %a_907, %b_913 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_922 = tile.matmul_modq %a_908, %b_910 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_923 = tile.matmul_modq %a_908, %b_911 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_924 = tile.matmul_modq %a_908, %b_912 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_925 = tile.matmul_modq %a_908, %b_913 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_926 = tile.matmul_modq %a_909, %b_910 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_927 = tile.matmul_modq %a_909, %b_911 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_928 = tile.matmul_modq %a_909, %b_912 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_929 = tile.matmul_modq %a_909, %b_913 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_930 = tile.add_modq %P_914 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_931 = tile.add_modq %P_915, %P_918 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_932 = tile.add_modq %P_916, %P_919, %P_922 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_933 = tile.add_modq %P_917, %P_920, %P_923, %P_926 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_934 = tile.add_modq %P_921, %P_924, %P_927 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_935 = tile.add_modq %P_925, %P_928 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_936 = tile.add_modq %P_929 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_xE_268434977 = tile.recombine_modq %C_930, %C_931, %C_932, %C_933, %C_934, %C_935, %C_936 {radix = 128 : i64, q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_937, %a_938, %a_939, %a_940 = tile.decompose %fbcD__268435361__268435009__268434977__268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_941, %b_942, %b_943, %b_944 = tile.decompose %rs2_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_945 = tile.matmul_modq %a_937, %b_941 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_946 = tile.matmul_modq %a_937, %b_942 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_947 = tile.matmul_modq %a_937, %b_943 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_948 = tile.matmul_modq %a_937, %b_944 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_949 = tile.matmul_modq %a_938, %b_941 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_950 = tile.matmul_modq %a_938, %b_942 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_951 = tile.matmul_modq %a_938, %b_943 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_952 = tile.matmul_modq %a_938, %b_944 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_953 = tile.matmul_modq %a_939, %b_941 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_954 = tile.matmul_modq %a_939, %b_942 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_955 = tile.matmul_modq %a_939, %b_943 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_956 = tile.matmul_modq %a_939, %b_944 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_957 = tile.matmul_modq %a_940, %b_941 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_958 = tile.matmul_modq %a_940, %b_942 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_959 = tile.matmul_modq %a_940, %b_943 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_960 = tile.matmul_modq %a_940, %b_944 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_961 = tile.add_modq %P_945 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_962 = tile.add_modq %P_946, %P_949 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_963 = tile.add_modq %P_947, %P_950, %P_953 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_964 = tile.add_modq %P_948, %P_951, %P_954, %P_957 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_965 = tile.add_modq %P_952, %P_955, %P_958 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_966 = tile.add_modq %P_956, %P_959 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_967 = tile.add_modq %P_960 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_fbc_268433569 = tile.recombine_modq %C_961, %C_962, %C_963, %C_964, %C_965, %C_966, %C_967 {radix = 128 : i64, q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0 = tile.reshape %rs2_fbc0_fbc_268433569 {shape = [16], q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_968, %a_969, %a_970, %a_971 = tile.decompose %fbcD__268435361__268435009__268434977__268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_972, %b_973, %b_974, %b_975 = tile.decompose %rs2_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_976 = tile.matmul_modq %a_968, %b_972 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_977 = tile.matmul_modq %a_968, %b_973 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_978 = tile.matmul_modq %a_968, %b_974 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_979 = tile.matmul_modq %a_968, %b_975 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_980 = tile.matmul_modq %a_969, %b_972 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_981 = tile.matmul_modq %a_969, %b_973 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_982 = tile.matmul_modq %a_969, %b_974 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_983 = tile.matmul_modq %a_969, %b_975 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_984 = tile.matmul_modq %a_970, %b_972 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_985 = tile.matmul_modq %a_970, %b_973 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_986 = tile.matmul_modq %a_970, %b_974 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_987 = tile.matmul_modq %a_970, %b_975 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_988 = tile.matmul_modq %a_971, %b_972 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_989 = tile.matmul_modq %a_971, %b_973 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_990 = tile.matmul_modq %a_971, %b_974 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_991 = tile.matmul_modq %a_971, %b_975 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_992 = tile.add_modq %P_976 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_993 = tile.add_modq %P_977, %P_980 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_994 = tile.add_modq %P_978, %P_981, %P_984 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_995 = tile.add_modq %P_979, %P_982, %P_985, %P_988 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_996 = tile.add_modq %P_983, %P_986, %P_989 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_997 = tile.add_modq %P_987, %P_990 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_998 = tile.add_modq %P_991 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_fbc_268433441 = tile.recombine_modq %C_992, %C_993, %C_994, %C_995, %C_996, %C_997, %C_998 {radix = 128 : i64, q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc1 = tile.reshape %rs2_fbc0_fbc_268433441 {shape = [16], q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_999, %a_1000, %a_1001, %a_1002 = tile.decompose %fbcD__268435361__268435009__268434977__268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1003, %b_1004, %b_1005, %b_1006 = tile.decompose %rs2_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1007 = tile.matmul_modq %a_999, %b_1003 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1008 = tile.matmul_modq %a_999, %b_1004 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1009 = tile.matmul_modq %a_999, %b_1005 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1010 = tile.matmul_modq %a_999, %b_1006 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1011 = tile.matmul_modq %a_1000, %b_1003 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1012 = tile.matmul_modq %a_1000, %b_1004 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1013 = tile.matmul_modq %a_1000, %b_1005 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1014 = tile.matmul_modq %a_1000, %b_1006 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1015 = tile.matmul_modq %a_1001, %b_1003 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1016 = tile.matmul_modq %a_1001, %b_1004 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1017 = tile.matmul_modq %a_1001, %b_1005 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1018 = tile.matmul_modq %a_1001, %b_1006 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1019 = tile.matmul_modq %a_1002, %b_1003 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1020 = tile.matmul_modq %a_1002, %b_1004 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1021 = tile.matmul_modq %a_1002, %b_1005 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1022 = tile.matmul_modq %a_1002, %b_1006 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1023 = tile.add_modq %P_1007 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1024 = tile.add_modq %P_1008, %P_1011 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1025 = tile.add_modq %P_1009, %P_1012, %P_1015 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1026 = tile.add_modq %P_1010, %P_1013, %P_1016, %P_1019 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1027 = tile.add_modq %P_1014, %P_1017, %P_1020 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1028 = tile.add_modq %P_1018, %P_1021 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1029 = tile.add_modq %P_1022 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_fbc_268433377 = tile.recombine_modq %C_1023, %C_1024, %C_1025, %C_1026, %C_1027, %C_1028, %C_1029 {radix = 128 : i64, q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc2 = tile.reshape %rs2_fbc0_fbc_268433377 {shape = [16], q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_1030, %a_1031, %a_1032, %a_1033 = tile.decompose %fbcD__268435361__268435009__268434977__268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1034, %b_1035, %b_1036, %b_1037 = tile.decompose %rs2_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1038 = tile.matmul_modq %a_1030, %b_1034 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1039 = tile.matmul_modq %a_1030, %b_1035 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1040 = tile.matmul_modq %a_1030, %b_1036 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1041 = tile.matmul_modq %a_1030, %b_1037 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1042 = tile.matmul_modq %a_1031, %b_1034 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1043 = tile.matmul_modq %a_1031, %b_1035 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1044 = tile.matmul_modq %a_1031, %b_1036 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1045 = tile.matmul_modq %a_1031, %b_1037 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1046 = tile.matmul_modq %a_1032, %b_1034 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1047 = tile.matmul_modq %a_1032, %b_1035 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1048 = tile.matmul_modq %a_1032, %b_1036 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1049 = tile.matmul_modq %a_1032, %b_1037 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1050 = tile.matmul_modq %a_1033, %b_1034 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1051 = tile.matmul_modq %a_1033, %b_1035 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1052 = tile.matmul_modq %a_1033, %b_1036 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1053 = tile.matmul_modq %a_1033, %b_1037 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1054 = tile.add_modq %P_1038 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1055 = tile.add_modq %P_1039, %P_1042 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1056 = tile.add_modq %P_1040, %P_1043, %P_1046 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1057 = tile.add_modq %P_1041, %P_1044, %P_1047, %P_1050 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1058 = tile.add_modq %P_1045, %P_1048, %P_1051 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1059 = tile.add_modq %P_1049, %P_1052 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1060 = tile.add_modq %P_1053 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_fbc_268433057 = tile.recombine_modq %C_1054, %C_1055, %C_1056, %C_1057, %C_1058, %C_1059, %C_1060 {radix = 128 : i64, q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_1061, %a_1062, %a_1063, %a_1064, %a_1065, %a_1066 = tile.decompose %crnsF__268433569__268433441__268433377__268433057__38 {radix = 128 : i64, nlimbs = 6 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1067, %b_1068, %b_1069, %b_1070 = tile.decompose %rs2_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %wP_1071 = tile.matmul_wide %a_1061, %b_1067 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1072 = tile.matmul_wide %a_1061, %b_1068 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1073 = tile.matmul_wide %a_1061, %b_1069 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1074 = tile.matmul_wide %a_1061, %b_1070 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1075 = tile.matmul_wide %a_1062, %b_1067 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1076 = tile.matmul_wide %a_1062, %b_1068 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1077 = tile.matmul_wide %a_1062, %b_1069 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1078 = tile.matmul_wide %a_1062, %b_1070 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1079 = tile.matmul_wide %a_1063, %b_1067 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1080 = tile.matmul_wide %a_1063, %b_1068 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1081 = tile.matmul_wide %a_1063, %b_1069 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1082 = tile.matmul_wide %a_1063, %b_1070 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1083 = tile.matmul_wide %a_1064, %b_1067 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1084 = tile.matmul_wide %a_1064, %b_1068 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1085 = tile.matmul_wide %a_1064, %b_1069 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1086 = tile.matmul_wide %a_1064, %b_1070 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1087 = tile.matmul_wide %a_1065, %b_1067 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1088 = tile.matmul_wide %a_1065, %b_1068 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1089 = tile.matmul_wide %a_1065, %b_1069 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1090 = tile.matmul_wide %a_1065, %b_1070 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1091 = tile.matmul_wide %a_1066, %b_1067 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1092 = tile.matmul_wide %a_1066, %b_1068 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1093 = tile.matmul_wide %a_1066, %b_1069 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1094 = tile.matmul_wide %a_1066, %b_1070 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %rs2_etQ0_s = tile.recombine_wide %wP_1071, %wP_1072, %wP_1073, %wP_1074, %wP_1075, %wP_1076, %wP_1077, %wP_1078, %wP_1079, %wP_1080, %wP_1081, %wP_1082, %wP_1083, %wP_1084, %wP_1085, %wP_1086, %wP_1087, %wP_1088, %wP_1089, %wP_1090, %wP_1091, %wP_1092, %wP_1093, %wP_1094 {radix = 128 : i64, exps = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8]} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_k = tile.fixed_slice %rs2_etQ0_s {shift = 38 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_neg = tile.fixed_slice %rs2_etQ0_s {shift = 37 : i64, mask = 1 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %a_1095, %a_1096, %a_1097, %a_1098 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1099, %b_1100, %b_1101, %b_1102 = tile.decompose %rs2_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_1103 = tile.matmul_modq %a_1095, %b_1099 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1104 = tile.matmul_modq %a_1095, %b_1100 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1105 = tile.matmul_modq %a_1095, %b_1101 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1106 = tile.matmul_modq %a_1095, %b_1102 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1107 = tile.matmul_modq %a_1096, %b_1099 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1108 = tile.matmul_modq %a_1096, %b_1100 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1109 = tile.matmul_modq %a_1096, %b_1101 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1110 = tile.matmul_modq %a_1096, %b_1102 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1111 = tile.matmul_modq %a_1097, %b_1099 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1112 = tile.matmul_modq %a_1097, %b_1100 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1113 = tile.matmul_modq %a_1097, %b_1101 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1114 = tile.matmul_modq %a_1097, %b_1102 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1115 = tile.matmul_modq %a_1098, %b_1099 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1116 = tile.matmul_modq %a_1098, %b_1100 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1117 = tile.matmul_modq %a_1098, %b_1101 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1118 = tile.matmul_modq %a_1098, %b_1102 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_1119 = tile.add_modq %P_1103 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1120 = tile.add_modq %P_1104, %P_1107 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1121 = tile.add_modq %P_1105, %P_1108, %P_1111 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1122 = tile.add_modq %P_1106, %P_1109, %P_1112, %P_1115 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1123 = tile.add_modq %P_1110, %P_1113, %P_1116 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1124 = tile.add_modq %P_1114, %P_1117 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1125 = tile.add_modq %P_1118 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_xE_268435361 = tile.recombine_modq %C_1119, %C_1120, %C_1121, %C_1122, %C_1123, %C_1124, %C_1125 {radix = 128 : i64, q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_kg_268435361 = tile.mul_modq %rs2_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_acc_268435361 = tile.add_modq %rs2_etQ0_xE_268435361, %rs2_etQ0_kg_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_negM_268435361 = tile.mul_modq %rs2_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_accs_268435361 = tile.sub_modq %rs2_etQ0_acc_268435361, %rs2_etQ0_negM_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0 = tile.reshape %rs2_etQ0_accs_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_1126, %a_1127, %a_1128, %a_1129 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1130, %b_1131, %b_1132, %b_1133 = tile.decompose %rs2_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_1134 = tile.matmul_modq %a_1126, %b_1130 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1135 = tile.matmul_modq %a_1126, %b_1131 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1136 = tile.matmul_modq %a_1126, %b_1132 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1137 = tile.matmul_modq %a_1126, %b_1133 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1138 = tile.matmul_modq %a_1127, %b_1130 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1139 = tile.matmul_modq %a_1127, %b_1131 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1140 = tile.matmul_modq %a_1127, %b_1132 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1141 = tile.matmul_modq %a_1127, %b_1133 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1142 = tile.matmul_modq %a_1128, %b_1130 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1143 = tile.matmul_modq %a_1128, %b_1131 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1144 = tile.matmul_modq %a_1128, %b_1132 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1145 = tile.matmul_modq %a_1128, %b_1133 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1146 = tile.matmul_modq %a_1129, %b_1130 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1147 = tile.matmul_modq %a_1129, %b_1131 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1148 = tile.matmul_modq %a_1129, %b_1132 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1149 = tile.matmul_modq %a_1129, %b_1133 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_1150 = tile.add_modq %P_1134 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1151 = tile.add_modq %P_1135, %P_1138 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1152 = tile.add_modq %P_1136, %P_1139, %P_1142 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1153 = tile.add_modq %P_1137, %P_1140, %P_1143, %P_1146 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1154 = tile.add_modq %P_1141, %P_1144, %P_1147 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1155 = tile.add_modq %P_1145, %P_1148 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1156 = tile.add_modq %P_1149 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_xE_268435009 = tile.recombine_modq %C_1150, %C_1151, %C_1152, %C_1153, %C_1154, %C_1155, %C_1156 {radix = 128 : i64, q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_kg_268435009 = tile.mul_modq %rs2_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_acc_268435009 = tile.add_modq %rs2_etQ0_xE_268435009, %rs2_etQ0_kg_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_negM_268435009 = tile.mul_modq %rs2_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_accs_268435009 = tile.sub_modq %rs2_etQ0_acc_268435009, %rs2_etQ0_negM_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ1 = tile.reshape %rs2_etQ0_accs_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_1157, %a_1158, %a_1159, %a_1160 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1161, %b_1162, %b_1163, %b_1164 = tile.decompose %rs2_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_1165 = tile.matmul_modq %a_1157, %b_1161 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1166 = tile.matmul_modq %a_1157, %b_1162 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1167 = tile.matmul_modq %a_1157, %b_1163 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1168 = tile.matmul_modq %a_1157, %b_1164 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1169 = tile.matmul_modq %a_1158, %b_1161 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1170 = tile.matmul_modq %a_1158, %b_1162 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1171 = tile.matmul_modq %a_1158, %b_1163 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1172 = tile.matmul_modq %a_1158, %b_1164 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1173 = tile.matmul_modq %a_1159, %b_1161 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1174 = tile.matmul_modq %a_1159, %b_1162 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1175 = tile.matmul_modq %a_1159, %b_1163 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1176 = tile.matmul_modq %a_1159, %b_1164 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1177 = tile.matmul_modq %a_1160, %b_1161 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1178 = tile.matmul_modq %a_1160, %b_1162 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1179 = tile.matmul_modq %a_1160, %b_1163 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1180 = tile.matmul_modq %a_1160, %b_1164 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_1181 = tile.add_modq %P_1165 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1182 = tile.add_modq %P_1166, %P_1169 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1183 = tile.add_modq %P_1167, %P_1170, %P_1173 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1184 = tile.add_modq %P_1168, %P_1171, %P_1174, %P_1177 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1185 = tile.add_modq %P_1172, %P_1175, %P_1178 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1186 = tile.add_modq %P_1176, %P_1179 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1187 = tile.add_modq %P_1180 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_xE_268434977 = tile.recombine_modq %C_1181, %C_1182, %C_1183, %C_1184, %C_1185, %C_1186, %C_1187 {radix = 128 : i64, q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_1188, %a_1189, %a_1190, %a_1191 = tile.decompose %fbcD__268435361__268435009__268434977__268434721 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1192, %b_1193, %b_1194, %b_1195 = tile.decompose %rl_c2P0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1196 = tile.matmul_modq %a_1188, %b_1192 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1197 = tile.matmul_modq %a_1188, %b_1193 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1198 = tile.matmul_modq %a_1188, %b_1194 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1199 = tile.matmul_modq %a_1188, %b_1195 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1200 = tile.matmul_modq %a_1189, %b_1192 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1201 = tile.matmul_modq %a_1189, %b_1193 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1202 = tile.matmul_modq %a_1189, %b_1194 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1203 = tile.matmul_modq %a_1189, %b_1195 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1204 = tile.matmul_modq %a_1190, %b_1192 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1205 = tile.matmul_modq %a_1190, %b_1193 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1206 = tile.matmul_modq %a_1190, %b_1194 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1207 = tile.matmul_modq %a_1190, %b_1195 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1208 = tile.matmul_modq %a_1191, %b_1192 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1209 = tile.matmul_modq %a_1191, %b_1193 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1210 = tile.matmul_modq %a_1191, %b_1194 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1211 = tile.matmul_modq %a_1191, %b_1195 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1212 = tile.add_modq %P_1196 {q = 268434721 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1213 = tile.add_modq %P_1197, %P_1200 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1214 = tile.add_modq %P_1198, %P_1201, %P_1204 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1215 = tile.add_modq %P_1199, %P_1202, %P_1205, %P_1208 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1216 = tile.add_modq %P_1203, %P_1206, %P_1209 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1217 = tile.add_modq %P_1207, %P_1210 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1218 = tile.add_modq %P_1211 {q = 268434721 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P0_fbc_268434721 = tile.recombine_modq %C_1212, %C_1213, %C_1214, %C_1215, %C_1216, %C_1217, %C_1218 {radix = 128 : i64, q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P0 = tile.reshape %rl_c2P0_fbc_268434721 {shape = [16], q = 268434721 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268434401 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268434401"} : tensor<1x3xi32>
    %a_1219, %a_1220, %a_1221, %a_1222 = tile.decompose %fbcD__268435361__268435009__268434977__268434401 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1223, %b_1224, %b_1225, %b_1226 = tile.decompose %rl_c2P0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1227 = tile.matmul_modq %a_1219, %b_1223 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1228 = tile.matmul_modq %a_1219, %b_1224 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1229 = tile.matmul_modq %a_1219, %b_1225 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1230 = tile.matmul_modq %a_1219, %b_1226 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1231 = tile.matmul_modq %a_1220, %b_1223 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1232 = tile.matmul_modq %a_1220, %b_1224 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1233 = tile.matmul_modq %a_1220, %b_1225 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1234 = tile.matmul_modq %a_1220, %b_1226 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1235 = tile.matmul_modq %a_1221, %b_1223 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1236 = tile.matmul_modq %a_1221, %b_1224 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1237 = tile.matmul_modq %a_1221, %b_1225 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1238 = tile.matmul_modq %a_1221, %b_1226 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1239 = tile.matmul_modq %a_1222, %b_1223 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1240 = tile.matmul_modq %a_1222, %b_1224 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1241 = tile.matmul_modq %a_1222, %b_1225 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1242 = tile.matmul_modq %a_1222, %b_1226 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1243 = tile.add_modq %P_1227 {q = 268434401 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1244 = tile.add_modq %P_1228, %P_1231 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1245 = tile.add_modq %P_1229, %P_1232, %P_1235 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1246 = tile.add_modq %P_1230, %P_1233, %P_1236, %P_1239 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1247 = tile.add_modq %P_1234, %P_1237, %P_1240 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1248 = tile.add_modq %P_1238, %P_1241 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1249 = tile.add_modq %P_1242 {q = 268434401 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P0_fbc_268434401 = tile.recombine_modq %C_1243, %C_1244, %C_1245, %C_1246, %C_1247, %C_1248, %C_1249 {radix = 128 : i64, q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P1 = tile.reshape %rl_c2P0_fbc_268434401 {shape = [16], q = 268434401 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433761 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433761"} : tensor<1x3xi32>
    %a_1250, %a_1251, %a_1252, %a_1253 = tile.decompose %fbcD__268435361__268435009__268434977__268433761 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1254, %b_1255, %b_1256, %b_1257 = tile.decompose %rl_c2P0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1258 = tile.matmul_modq %a_1250, %b_1254 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1259 = tile.matmul_modq %a_1250, %b_1255 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1260 = tile.matmul_modq %a_1250, %b_1256 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1261 = tile.matmul_modq %a_1250, %b_1257 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1262 = tile.matmul_modq %a_1251, %b_1254 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1263 = tile.matmul_modq %a_1251, %b_1255 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1264 = tile.matmul_modq %a_1251, %b_1256 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1265 = tile.matmul_modq %a_1251, %b_1257 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1266 = tile.matmul_modq %a_1252, %b_1254 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1267 = tile.matmul_modq %a_1252, %b_1255 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1268 = tile.matmul_modq %a_1252, %b_1256 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1269 = tile.matmul_modq %a_1252, %b_1257 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1270 = tile.matmul_modq %a_1253, %b_1254 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1271 = tile.matmul_modq %a_1253, %b_1255 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1272 = tile.matmul_modq %a_1253, %b_1256 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1273 = tile.matmul_modq %a_1253, %b_1257 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1274 = tile.add_modq %P_1258 {q = 268433761 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1275 = tile.add_modq %P_1259, %P_1262 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1276 = tile.add_modq %P_1260, %P_1263, %P_1266 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1277 = tile.add_modq %P_1261, %P_1264, %P_1267, %P_1270 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1278 = tile.add_modq %P_1265, %P_1268, %P_1271 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1279 = tile.add_modq %P_1269, %P_1272 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1280 = tile.add_modq %P_1273 {q = 268433761 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P0_fbc_268433761 = tile.recombine_modq %C_1274, %C_1275, %C_1276, %C_1277, %C_1278, %C_1279, %C_1280 {radix = 128 : i64, q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P2 = tile.reshape %rl_c2P0_fbc_268433761 {shape = [16], q = 268433761 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %W_f_268434721 = tile.const {name = "W_f_268434721", q = 268434721 : i64} : tensor<16x16xi32>
    %W_f_268434401 = tile.const {name = "W_f_268434401", q = 268434401 : i64} : tensor<16x16xi32>
    %W_f_268433761 = tile.const {name = "W_f_268433761", q = 268433761 : i64} : tensor<16x16xi32>
    %rl_evkA_0c = rns.const {name = "rl_evkA_0c", q = 268435361 : i64} : tensor<16xi32>
    %col_fa127 = tile.reshape %rs2_etQ0 {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa128 = tile.reshape %rl_evkA_0c {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa129 = tile.concat %col_fa127, %col_fa128 {axis = 1 : i64, q = 268435361 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1281, %a_1282, %a_1283, %a_1284 = tile.decompose %W_f_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1285, %b_1286, %b_1287, %b_1288 = tile.decompose %rhs_fa129 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1289 = tile.matmul_modq %a_1281, %b_1285 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1290 = tile.matmul_modq %a_1281, %b_1286 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1291 = tile.matmul_modq %a_1281, %b_1287 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1292 = tile.matmul_modq %a_1281, %b_1288 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1293 = tile.matmul_modq %a_1282, %b_1285 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1294 = tile.matmul_modq %a_1282, %b_1286 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1295 = tile.matmul_modq %a_1282, %b_1287 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1296 = tile.matmul_modq %a_1282, %b_1288 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1297 = tile.matmul_modq %a_1283, %b_1285 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1298 = tile.matmul_modq %a_1283, %b_1286 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1299 = tile.matmul_modq %a_1283, %b_1287 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1300 = tile.matmul_modq %a_1283, %b_1288 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1301 = tile.matmul_modq %a_1284, %b_1285 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1302 = tile.matmul_modq %a_1284, %b_1286 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1303 = tile.matmul_modq %a_1284, %b_1287 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1304 = tile.matmul_modq %a_1284, %b_1288 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1305 = tile.add_modq %P_1289 {q = 268435361 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1306 = tile.add_modq %P_1290, %P_1293 {q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1307 = tile.add_modq %P_1291, %P_1294, %P_1297 {q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1308 = tile.add_modq %P_1292, %P_1295, %P_1298, %P_1301 {q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1309 = tile.add_modq %P_1296, %P_1299, %P_1302 {q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1310 = tile.add_modq %P_1300, %P_1303 {q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1311 = tile.add_modq %P_1304 {q = 268435361 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa130 = tile.recombine_modq %C_1305, %C_1306, %C_1307, %C_1308, %C_1309, %C_1310, %C_1311 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa131, %yc_fa132 = tile.split %out_fa130 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_X0 = tile.reshape %yc_fa131 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkA_0 = tile.reshape %yc_fa132 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_AY0 = tile.mul_modq %rl_X0, %rl_evkA_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_1c = rns.const {name = "rl_evkA_1c", q = 268435009 : i64} : tensor<16xi32>
    %col_fa133 = tile.reshape %rs2_etQ1 {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa134 = tile.reshape %rl_evkA_1c {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa135 = tile.concat %col_fa133, %col_fa134 {axis = 1 : i64, q = 268435009 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1312, %a_1313, %a_1314, %a_1315 = tile.decompose %W_f_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1316, %b_1317, %b_1318, %b_1319 = tile.decompose %rhs_fa135 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1320 = tile.matmul_modq %a_1312, %b_1316 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1321 = tile.matmul_modq %a_1312, %b_1317 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1322 = tile.matmul_modq %a_1312, %b_1318 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1323 = tile.matmul_modq %a_1312, %b_1319 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1324 = tile.matmul_modq %a_1313, %b_1316 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1325 = tile.matmul_modq %a_1313, %b_1317 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1326 = tile.matmul_modq %a_1313, %b_1318 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1327 = tile.matmul_modq %a_1313, %b_1319 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1328 = tile.matmul_modq %a_1314, %b_1316 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1329 = tile.matmul_modq %a_1314, %b_1317 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1330 = tile.matmul_modq %a_1314, %b_1318 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1331 = tile.matmul_modq %a_1314, %b_1319 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1332 = tile.matmul_modq %a_1315, %b_1316 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1333 = tile.matmul_modq %a_1315, %b_1317 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1334 = tile.matmul_modq %a_1315, %b_1318 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1335 = tile.matmul_modq %a_1315, %b_1319 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1336 = tile.add_modq %P_1320 {q = 268435009 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1337 = tile.add_modq %P_1321, %P_1324 {q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1338 = tile.add_modq %P_1322, %P_1325, %P_1328 {q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1339 = tile.add_modq %P_1323, %P_1326, %P_1329, %P_1332 {q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1340 = tile.add_modq %P_1327, %P_1330, %P_1333 {q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1341 = tile.add_modq %P_1331, %P_1334 {q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1342 = tile.add_modq %P_1335 {q = 268435009 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa136 = tile.recombine_modq %C_1336, %C_1337, %C_1338, %C_1339, %C_1340, %C_1341, %C_1342 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa137, %yc_fa138 = tile.split %out_fa136 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_X1 = tile.reshape %yc_fa137 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkA_1 = tile.reshape %yc_fa138 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_AY1 = tile.mul_modq %rl_X1, %rl_evkA_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_2c = rns.const {name = "rl_evkA_2c", q = 268434977 : i64} : tensor<16xi32>
    %col_fa139 = tile.reshape %rs2_etQ2 {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa140 = tile.reshape %rl_evkA_2c {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa141 = tile.concat %col_fa139, %col_fa140 {axis = 1 : i64, q = 268434977 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1343, %a_1344, %a_1345, %a_1346 = tile.decompose %W_f_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1347, %b_1348, %b_1349, %b_1350 = tile.decompose %rhs_fa141 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1351 = tile.matmul_modq %a_1343, %b_1347 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1352 = tile.matmul_modq %a_1343, %b_1348 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1353 = tile.matmul_modq %a_1343, %b_1349 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1354 = tile.matmul_modq %a_1343, %b_1350 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1355 = tile.matmul_modq %a_1344, %b_1347 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1356 = tile.matmul_modq %a_1344, %b_1348 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1357 = tile.matmul_modq %a_1344, %b_1349 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1358 = tile.matmul_modq %a_1344, %b_1350 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1359 = tile.matmul_modq %a_1345, %b_1347 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1360 = tile.matmul_modq %a_1345, %b_1348 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1361 = tile.matmul_modq %a_1345, %b_1349 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1362 = tile.matmul_modq %a_1345, %b_1350 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1363 = tile.matmul_modq %a_1346, %b_1347 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1364 = tile.matmul_modq %a_1346, %b_1348 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1365 = tile.matmul_modq %a_1346, %b_1349 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1366 = tile.matmul_modq %a_1346, %b_1350 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1367 = tile.add_modq %P_1351 {q = 268434977 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1368 = tile.add_modq %P_1352, %P_1355 {q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1369 = tile.add_modq %P_1353, %P_1356, %P_1359 {q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1370 = tile.add_modq %P_1354, %P_1357, %P_1360, %P_1363 {q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1371 = tile.add_modq %P_1358, %P_1361, %P_1364 {q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1372 = tile.add_modq %P_1362, %P_1365 {q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1373 = tile.add_modq %P_1366 {q = 268434977 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa142 = tile.recombine_modq %C_1367, %C_1368, %C_1369, %C_1370, %C_1371, %C_1372, %C_1373 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa143, %yc_fa144 = tile.split %out_fa142 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_X2 = tile.reshape %yc_fa143 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkA_2 = tile.reshape %yc_fa144 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_AY2 = tile.mul_modq %rl_X2, %rl_evkA_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_3c = rns.const {name = "rl_evkA_3c", q = 268434721 : i64} : tensor<16xi32>
    %col_fa145 = tile.reshape %rl_c2P0 {shape = [16, 1], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa146 = tile.reshape %rl_evkA_3c {shape = [16, 1], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa147 = tile.concat %col_fa145, %col_fa146 {axis = 1 : i64, q = 268434721 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1374, %a_1375, %a_1376, %a_1377 = tile.decompose %W_f_268434721 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1378, %b_1379, %b_1380, %b_1381 = tile.decompose %rhs_fa147 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1382 = tile.matmul_modq %a_1374, %b_1378 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1383 = tile.matmul_modq %a_1374, %b_1379 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1384 = tile.matmul_modq %a_1374, %b_1380 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1385 = tile.matmul_modq %a_1374, %b_1381 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1386 = tile.matmul_modq %a_1375, %b_1378 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1387 = tile.matmul_modq %a_1375, %b_1379 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1388 = tile.matmul_modq %a_1375, %b_1380 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1389 = tile.matmul_modq %a_1375, %b_1381 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1390 = tile.matmul_modq %a_1376, %b_1378 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1391 = tile.matmul_modq %a_1376, %b_1379 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1392 = tile.matmul_modq %a_1376, %b_1380 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1393 = tile.matmul_modq %a_1376, %b_1381 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1394 = tile.matmul_modq %a_1377, %b_1378 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1395 = tile.matmul_modq %a_1377, %b_1379 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1396 = tile.matmul_modq %a_1377, %b_1380 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1397 = tile.matmul_modq %a_1377, %b_1381 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1398 = tile.add_modq %P_1382 {q = 268434721 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1399 = tile.add_modq %P_1383, %P_1386 {q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1400 = tile.add_modq %P_1384, %P_1387, %P_1390 {q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1401 = tile.add_modq %P_1385, %P_1388, %P_1391, %P_1394 {q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1402 = tile.add_modq %P_1389, %P_1392, %P_1395 {q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1403 = tile.add_modq %P_1393, %P_1396 {q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1404 = tile.add_modq %P_1397 {q = 268434721 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa148 = tile.recombine_modq %C_1398, %C_1399, %C_1400, %C_1401, %C_1402, %C_1403, %C_1404 {radix = 128 : i64, q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa149, %yc_fa150 = tile.split %out_fa148 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_X3 = tile.reshape %yc_fa149 {shape = [16], q = 268434721 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkA_3 = tile.reshape %yc_fa150 {shape = [16], q = 268434721 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_AY3 = tile.mul_modq %rl_X3, %rl_evkA_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268434721 = tile.const {name = "W_i_268434721", q = 268434721 : i64} : tensor<16x16xi32>
    %rl_evkA_4c = rns.const {name = "rl_evkA_4c", q = 268434401 : i64} : tensor<16xi32>
    %col_fa151 = tile.reshape %rl_c2P1 {shape = [16, 1], q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa152 = tile.reshape %rl_evkA_4c {shape = [16, 1], q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa153 = tile.concat %col_fa151, %col_fa152 {axis = 1 : i64, q = 268434401 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1405, %a_1406, %a_1407, %a_1408 = tile.decompose %W_f_268434401 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1409, %b_1410, %b_1411, %b_1412 = tile.decompose %rhs_fa153 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1413 = tile.matmul_modq %a_1405, %b_1409 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1414 = tile.matmul_modq %a_1405, %b_1410 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1415 = tile.matmul_modq %a_1405, %b_1411 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1416 = tile.matmul_modq %a_1405, %b_1412 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1417 = tile.matmul_modq %a_1406, %b_1409 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1418 = tile.matmul_modq %a_1406, %b_1410 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1419 = tile.matmul_modq %a_1406, %b_1411 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1420 = tile.matmul_modq %a_1406, %b_1412 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1421 = tile.matmul_modq %a_1407, %b_1409 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1422 = tile.matmul_modq %a_1407, %b_1410 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1423 = tile.matmul_modq %a_1407, %b_1411 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1424 = tile.matmul_modq %a_1407, %b_1412 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1425 = tile.matmul_modq %a_1408, %b_1409 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1426 = tile.matmul_modq %a_1408, %b_1410 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1427 = tile.matmul_modq %a_1408, %b_1411 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1428 = tile.matmul_modq %a_1408, %b_1412 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1429 = tile.add_modq %P_1413 {q = 268434401 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1430 = tile.add_modq %P_1414, %P_1417 {q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1431 = tile.add_modq %P_1415, %P_1418, %P_1421 {q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1432 = tile.add_modq %P_1416, %P_1419, %P_1422, %P_1425 {q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1433 = tile.add_modq %P_1420, %P_1423, %P_1426 {q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1434 = tile.add_modq %P_1424, %P_1427 {q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1435 = tile.add_modq %P_1428 {q = 268434401 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa154 = tile.recombine_modq %C_1429, %C_1430, %C_1431, %C_1432, %C_1433, %C_1434, %C_1435 {radix = 128 : i64, q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa155, %yc_fa156 = tile.split %out_fa154 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_X4 = tile.reshape %yc_fa155 {shape = [16], q = 268434401 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkA_4 = tile.reshape %yc_fa156 {shape = [16], q = 268434401 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_AY4 = tile.mul_modq %rl_X4, %rl_evkA_4 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268434401 = tile.const {name = "W_i_268434401", q = 268434401 : i64} : tensor<16x16xi32>
    %rl_evkA_5c = rns.const {name = "rl_evkA_5c", q = 268433761 : i64} : tensor<16xi32>
    %col_fa157 = tile.reshape %rl_c2P2 {shape = [16, 1], q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa158 = tile.reshape %rl_evkA_5c {shape = [16, 1], q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa159 = tile.concat %col_fa157, %col_fa158 {axis = 1 : i64, q = 268433761 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1436, %a_1437, %a_1438, %a_1439 = tile.decompose %W_f_268433761 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1440, %b_1441, %b_1442, %b_1443 = tile.decompose %rhs_fa159 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1444 = tile.matmul_modq %a_1436, %b_1440 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1445 = tile.matmul_modq %a_1436, %b_1441 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1446 = tile.matmul_modq %a_1436, %b_1442 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1447 = tile.matmul_modq %a_1436, %b_1443 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1448 = tile.matmul_modq %a_1437, %b_1440 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1449 = tile.matmul_modq %a_1437, %b_1441 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1450 = tile.matmul_modq %a_1437, %b_1442 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1451 = tile.matmul_modq %a_1437, %b_1443 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1452 = tile.matmul_modq %a_1438, %b_1440 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1453 = tile.matmul_modq %a_1438, %b_1441 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1454 = tile.matmul_modq %a_1438, %b_1442 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1455 = tile.matmul_modq %a_1438, %b_1443 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1456 = tile.matmul_modq %a_1439, %b_1440 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1457 = tile.matmul_modq %a_1439, %b_1441 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1458 = tile.matmul_modq %a_1439, %b_1442 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1459 = tile.matmul_modq %a_1439, %b_1443 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1460 = tile.add_modq %P_1444 {q = 268433761 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1461 = tile.add_modq %P_1445, %P_1448 {q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1462 = tile.add_modq %P_1446, %P_1449, %P_1452 {q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1463 = tile.add_modq %P_1447, %P_1450, %P_1453, %P_1456 {q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1464 = tile.add_modq %P_1451, %P_1454, %P_1457 {q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1465 = tile.add_modq %P_1455, %P_1458 {q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1466 = tile.add_modq %P_1459 {q = 268433761 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa160 = tile.recombine_modq %C_1460, %C_1461, %C_1462, %C_1463, %C_1464, %C_1465, %C_1466 {radix = 128 : i64, q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa161, %yc_fa162 = tile.split %out_fa160 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_X5 = tile.reshape %yc_fa161 {shape = [16], q = 268433761 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkA_5 = tile.reshape %yc_fa162 {shape = [16], q = 268433761 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_AY5 = tile.mul_modq %rl_X5, %rl_evkA_5 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433761 = tile.const {name = "W_i_268433761", q = 268433761 : i64} : tensor<16x16xi32>
    %rl_evkB_0c = rns.const {name = "rl_evkB_0c", q = 268435361 : i64} : tensor<16xi32>
    %a_1467, %a_1468, %a_1469, %a_1470 = tile.decompose %W_f_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1471, %b_1472, %b_1473, %b_1474 = tile.decompose %rl_evkB_0c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1475 = tile.matmul_modq %a_1467, %b_1471 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1476 = tile.matmul_modq %a_1467, %b_1472 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1477 = tile.matmul_modq %a_1467, %b_1473 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1478 = tile.matmul_modq %a_1467, %b_1474 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1479 = tile.matmul_modq %a_1468, %b_1471 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1480 = tile.matmul_modq %a_1468, %b_1472 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1481 = tile.matmul_modq %a_1468, %b_1473 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1482 = tile.matmul_modq %a_1468, %b_1474 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1483 = tile.matmul_modq %a_1469, %b_1471 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1484 = tile.matmul_modq %a_1469, %b_1472 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1485 = tile.matmul_modq %a_1469, %b_1473 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1486 = tile.matmul_modq %a_1469, %b_1474 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1487 = tile.matmul_modq %a_1470, %b_1471 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1488 = tile.matmul_modq %a_1470, %b_1472 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1489 = tile.matmul_modq %a_1470, %b_1473 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1490 = tile.matmul_modq %a_1470, %b_1474 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1491 = tile.add_modq %P_1475 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1492 = tile.add_modq %P_1476, %P_1479 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1493 = tile.add_modq %P_1477, %P_1480, %P_1483 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1494 = tile.add_modq %P_1478, %P_1481, %P_1484, %P_1487 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1495 = tile.add_modq %P_1482, %P_1485, %P_1488 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1496 = tile.add_modq %P_1486, %P_1489 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1497 = tile.add_modq %P_1490 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_0 = tile.recombine_modq %C_1491, %C_1492, %C_1493, %C_1494, %C_1495, %C_1496, %C_1497 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY0 = tile.mul_modq %rl_X0, %rl_evkB_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %col_fa163 = tile.reshape %rl_AY0 {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa164 = tile.reshape %rl_BY0 {shape = [16, 1], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa165 = tile.concat %col_fa163, %col_fa164 {axis = 1 : i64, q = 268435361 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1498, %a_1499, %a_1500, %a_1501 = tile.decompose %W_i_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1502, %b_1503, %b_1504, %b_1505 = tile.decompose %rhs_fa165 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1506 = tile.matmul_modq %a_1498, %b_1502 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1507 = tile.matmul_modq %a_1498, %b_1503 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1508 = tile.matmul_modq %a_1498, %b_1504 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1509 = tile.matmul_modq %a_1498, %b_1505 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1510 = tile.matmul_modq %a_1499, %b_1502 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1511 = tile.matmul_modq %a_1499, %b_1503 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1512 = tile.matmul_modq %a_1499, %b_1504 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1513 = tile.matmul_modq %a_1499, %b_1505 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1514 = tile.matmul_modq %a_1500, %b_1502 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1515 = tile.matmul_modq %a_1500, %b_1503 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1516 = tile.matmul_modq %a_1500, %b_1504 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1517 = tile.matmul_modq %a_1500, %b_1505 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1518 = tile.matmul_modq %a_1501, %b_1502 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1519 = tile.matmul_modq %a_1501, %b_1503 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1520 = tile.matmul_modq %a_1501, %b_1504 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1521 = tile.matmul_modq %a_1501, %b_1505 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1522 = tile.add_modq %P_1506 {q = 268435361 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1523 = tile.add_modq %P_1507, %P_1510 {q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1524 = tile.add_modq %P_1508, %P_1511, %P_1514 {q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1525 = tile.add_modq %P_1509, %P_1512, %P_1515, %P_1518 {q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1526 = tile.add_modq %P_1513, %P_1516, %P_1519 {q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1527 = tile.add_modq %P_1517, %P_1520 {q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1528 = tile.add_modq %P_1521 {q = 268435361 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa166 = tile.recombine_modq %C_1522, %C_1523, %C_1524, %C_1525, %C_1526, %C_1527, %C_1528 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa167, %yc_fa168 = tile.split %out_fa166 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_Aa0 = tile.reshape %yc_fa167 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_Ba0 = tile.reshape %yc_fa168 {shape = [16], q = 268435361 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkB_1c = rns.const {name = "rl_evkB_1c", q = 268435009 : i64} : tensor<16xi32>
    %a_1529, %a_1530, %a_1531, %a_1532 = tile.decompose %W_f_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1533, %b_1534, %b_1535, %b_1536 = tile.decompose %rl_evkB_1c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1537 = tile.matmul_modq %a_1529, %b_1533 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1538 = tile.matmul_modq %a_1529, %b_1534 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1539 = tile.matmul_modq %a_1529, %b_1535 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1540 = tile.matmul_modq %a_1529, %b_1536 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1541 = tile.matmul_modq %a_1530, %b_1533 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1542 = tile.matmul_modq %a_1530, %b_1534 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1543 = tile.matmul_modq %a_1530, %b_1535 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1544 = tile.matmul_modq %a_1530, %b_1536 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1545 = tile.matmul_modq %a_1531, %b_1533 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1546 = tile.matmul_modq %a_1531, %b_1534 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1547 = tile.matmul_modq %a_1531, %b_1535 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1548 = tile.matmul_modq %a_1531, %b_1536 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1549 = tile.matmul_modq %a_1532, %b_1533 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1550 = tile.matmul_modq %a_1532, %b_1534 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1551 = tile.matmul_modq %a_1532, %b_1535 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1552 = tile.matmul_modq %a_1532, %b_1536 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1553 = tile.add_modq %P_1537 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1554 = tile.add_modq %P_1538, %P_1541 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1555 = tile.add_modq %P_1539, %P_1542, %P_1545 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1556 = tile.add_modq %P_1540, %P_1543, %P_1546, %P_1549 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1557 = tile.add_modq %P_1544, %P_1547, %P_1550 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1558 = tile.add_modq %P_1548, %P_1551 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1559 = tile.add_modq %P_1552 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_1 = tile.recombine_modq %C_1553, %C_1554, %C_1555, %C_1556, %C_1557, %C_1558, %C_1559 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY1 = tile.mul_modq %rl_X1, %rl_evkB_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %col_fa169 = tile.reshape %rl_AY1 {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa170 = tile.reshape %rl_BY1 {shape = [16, 1], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa171 = tile.concat %col_fa169, %col_fa170 {axis = 1 : i64, q = 268435009 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1560, %a_1561, %a_1562, %a_1563 = tile.decompose %W_i_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1564, %b_1565, %b_1566, %b_1567 = tile.decompose %rhs_fa171 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1568 = tile.matmul_modq %a_1560, %b_1564 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1569 = tile.matmul_modq %a_1560, %b_1565 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1570 = tile.matmul_modq %a_1560, %b_1566 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1571 = tile.matmul_modq %a_1560, %b_1567 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1572 = tile.matmul_modq %a_1561, %b_1564 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1573 = tile.matmul_modq %a_1561, %b_1565 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1574 = tile.matmul_modq %a_1561, %b_1566 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1575 = tile.matmul_modq %a_1561, %b_1567 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1576 = tile.matmul_modq %a_1562, %b_1564 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1577 = tile.matmul_modq %a_1562, %b_1565 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1578 = tile.matmul_modq %a_1562, %b_1566 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1579 = tile.matmul_modq %a_1562, %b_1567 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1580 = tile.matmul_modq %a_1563, %b_1564 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1581 = tile.matmul_modq %a_1563, %b_1565 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1582 = tile.matmul_modq %a_1563, %b_1566 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1583 = tile.matmul_modq %a_1563, %b_1567 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1584 = tile.add_modq %P_1568 {q = 268435009 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1585 = tile.add_modq %P_1569, %P_1572 {q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1586 = tile.add_modq %P_1570, %P_1573, %P_1576 {q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1587 = tile.add_modq %P_1571, %P_1574, %P_1577, %P_1580 {q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1588 = tile.add_modq %P_1575, %P_1578, %P_1581 {q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1589 = tile.add_modq %P_1579, %P_1582 {q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1590 = tile.add_modq %P_1583 {q = 268435009 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa172 = tile.recombine_modq %C_1584, %C_1585, %C_1586, %C_1587, %C_1588, %C_1589, %C_1590 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa173, %yc_fa174 = tile.split %out_fa172 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_Aa1 = tile.reshape %yc_fa173 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_Ba1 = tile.reshape %yc_fa174 {shape = [16], q = 268435009 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkB_2c = rns.const {name = "rl_evkB_2c", q = 268434977 : i64} : tensor<16xi32>
    %a_1591, %a_1592, %a_1593, %a_1594 = tile.decompose %W_f_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1595, %b_1596, %b_1597, %b_1598 = tile.decompose %rl_evkB_2c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1599 = tile.matmul_modq %a_1591, %b_1595 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1600 = tile.matmul_modq %a_1591, %b_1596 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1601 = tile.matmul_modq %a_1591, %b_1597 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1602 = tile.matmul_modq %a_1591, %b_1598 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1603 = tile.matmul_modq %a_1592, %b_1595 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1604 = tile.matmul_modq %a_1592, %b_1596 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1605 = tile.matmul_modq %a_1592, %b_1597 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1606 = tile.matmul_modq %a_1592, %b_1598 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1607 = tile.matmul_modq %a_1593, %b_1595 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1608 = tile.matmul_modq %a_1593, %b_1596 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1609 = tile.matmul_modq %a_1593, %b_1597 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1610 = tile.matmul_modq %a_1593, %b_1598 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1611 = tile.matmul_modq %a_1594, %b_1595 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1612 = tile.matmul_modq %a_1594, %b_1596 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1613 = tile.matmul_modq %a_1594, %b_1597 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1614 = tile.matmul_modq %a_1594, %b_1598 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1615 = tile.add_modq %P_1599 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1616 = tile.add_modq %P_1600, %P_1603 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1617 = tile.add_modq %P_1601, %P_1604, %P_1607 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1618 = tile.add_modq %P_1602, %P_1605, %P_1608, %P_1611 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1619 = tile.add_modq %P_1606, %P_1609, %P_1612 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1620 = tile.add_modq %P_1610, %P_1613 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1621 = tile.add_modq %P_1614 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_2 = tile.recombine_modq %C_1615, %C_1616, %C_1617, %C_1618, %C_1619, %C_1620, %C_1621 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY2 = tile.mul_modq %rl_X2, %rl_evkB_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %col_fa175 = tile.reshape %rl_AY2 {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa176 = tile.reshape %rl_BY2 {shape = [16, 1], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa177 = tile.concat %col_fa175, %col_fa176 {axis = 1 : i64, q = 268434977 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1622, %a_1623, %a_1624, %a_1625 = tile.decompose %W_i_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1626, %b_1627, %b_1628, %b_1629 = tile.decompose %rhs_fa177 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1630 = tile.matmul_modq %a_1622, %b_1626 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1631 = tile.matmul_modq %a_1622, %b_1627 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1632 = tile.matmul_modq %a_1622, %b_1628 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1633 = tile.matmul_modq %a_1622, %b_1629 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1634 = tile.matmul_modq %a_1623, %b_1626 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1635 = tile.matmul_modq %a_1623, %b_1627 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1636 = tile.matmul_modq %a_1623, %b_1628 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1637 = tile.matmul_modq %a_1623, %b_1629 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1638 = tile.matmul_modq %a_1624, %b_1626 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1639 = tile.matmul_modq %a_1624, %b_1627 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1640 = tile.matmul_modq %a_1624, %b_1628 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1641 = tile.matmul_modq %a_1624, %b_1629 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1642 = tile.matmul_modq %a_1625, %b_1626 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1643 = tile.matmul_modq %a_1625, %b_1627 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1644 = tile.matmul_modq %a_1625, %b_1628 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1645 = tile.matmul_modq %a_1625, %b_1629 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1646 = tile.add_modq %P_1630 {q = 268434977 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1647 = tile.add_modq %P_1631, %P_1634 {q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1648 = tile.add_modq %P_1632, %P_1635, %P_1638 {q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1649 = tile.add_modq %P_1633, %P_1636, %P_1639, %P_1642 {q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1650 = tile.add_modq %P_1637, %P_1640, %P_1643 {q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1651 = tile.add_modq %P_1641, %P_1644 {q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1652 = tile.add_modq %P_1645 {q = 268434977 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa178 = tile.recombine_modq %C_1646, %C_1647, %C_1648, %C_1649, %C_1650, %C_1651, %C_1652 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa179, %yc_fa180 = tile.split %out_fa178 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_Aa2 = tile.reshape %yc_fa179 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_Ba2 = tile.reshape %yc_fa180 {shape = [16], q = 268434977 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkB_3c = rns.const {name = "rl_evkB_3c", q = 268434721 : i64} : tensor<16xi32>
    %a_1653, %a_1654, %a_1655, %a_1656 = tile.decompose %W_f_268434721 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1657, %b_1658, %b_1659, %b_1660 = tile.decompose %rl_evkB_3c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1661 = tile.matmul_modq %a_1653, %b_1657 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1662 = tile.matmul_modq %a_1653, %b_1658 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1663 = tile.matmul_modq %a_1653, %b_1659 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1664 = tile.matmul_modq %a_1653, %b_1660 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1665 = tile.matmul_modq %a_1654, %b_1657 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1666 = tile.matmul_modq %a_1654, %b_1658 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1667 = tile.matmul_modq %a_1654, %b_1659 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1668 = tile.matmul_modq %a_1654, %b_1660 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1669 = tile.matmul_modq %a_1655, %b_1657 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1670 = tile.matmul_modq %a_1655, %b_1658 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1671 = tile.matmul_modq %a_1655, %b_1659 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1672 = tile.matmul_modq %a_1655, %b_1660 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1673 = tile.matmul_modq %a_1656, %b_1657 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1674 = tile.matmul_modq %a_1656, %b_1658 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1675 = tile.matmul_modq %a_1656, %b_1659 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1676 = tile.matmul_modq %a_1656, %b_1660 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1677 = tile.add_modq %P_1661 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1678 = tile.add_modq %P_1662, %P_1665 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1679 = tile.add_modq %P_1663, %P_1666, %P_1669 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1680 = tile.add_modq %P_1664, %P_1667, %P_1670, %P_1673 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1681 = tile.add_modq %P_1668, %P_1671, %P_1674 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1682 = tile.add_modq %P_1672, %P_1675 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1683 = tile.add_modq %P_1676 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_3 = tile.recombine_modq %C_1677, %C_1678, %C_1679, %C_1680, %C_1681, %C_1682, %C_1683 {radix = 128 : i64, q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY3 = tile.mul_modq %rl_X3, %rl_evkB_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %col_fa181 = tile.reshape %rl_AY3 {shape = [16, 1], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa182 = tile.reshape %rl_BY3 {shape = [16, 1], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa183 = tile.concat %col_fa181, %col_fa182 {axis = 1 : i64, q = 268434721 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1684, %a_1685, %a_1686, %a_1687 = tile.decompose %W_i_268434721 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1688, %b_1689, %b_1690, %b_1691 = tile.decompose %rhs_fa183 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1692 = tile.matmul_modq %a_1684, %b_1688 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1693 = tile.matmul_modq %a_1684, %b_1689 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1694 = tile.matmul_modq %a_1684, %b_1690 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1695 = tile.matmul_modq %a_1684, %b_1691 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1696 = tile.matmul_modq %a_1685, %b_1688 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1697 = tile.matmul_modq %a_1685, %b_1689 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1698 = tile.matmul_modq %a_1685, %b_1690 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1699 = tile.matmul_modq %a_1685, %b_1691 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1700 = tile.matmul_modq %a_1686, %b_1688 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1701 = tile.matmul_modq %a_1686, %b_1689 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1702 = tile.matmul_modq %a_1686, %b_1690 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1703 = tile.matmul_modq %a_1686, %b_1691 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1704 = tile.matmul_modq %a_1687, %b_1688 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1705 = tile.matmul_modq %a_1687, %b_1689 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1706 = tile.matmul_modq %a_1687, %b_1690 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1707 = tile.matmul_modq %a_1687, %b_1691 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1708 = tile.add_modq %P_1692 {q = 268434721 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1709 = tile.add_modq %P_1693, %P_1696 {q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1710 = tile.add_modq %P_1694, %P_1697, %P_1700 {q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1711 = tile.add_modq %P_1695, %P_1698, %P_1701, %P_1704 {q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1712 = tile.add_modq %P_1699, %P_1702, %P_1705 {q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1713 = tile.add_modq %P_1703, %P_1706 {q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1714 = tile.add_modq %P_1707 {q = 268434721 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa184 = tile.recombine_modq %C_1708, %C_1709, %C_1710, %C_1711, %C_1712, %C_1713, %C_1714 {radix = 128 : i64, q = 268434721 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa185, %yc_fa186 = tile.split %out_fa184 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_Aa3 = tile.reshape %yc_fa185 {shape = [16], q = 268434721 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_Ba3 = tile.reshape %yc_fa186 {shape = [16], q = 268434721 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkB_4c = rns.const {name = "rl_evkB_4c", q = 268434401 : i64} : tensor<16xi32>
    %a_1715, %a_1716, %a_1717, %a_1718 = tile.decompose %W_f_268434401 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1719, %b_1720, %b_1721, %b_1722 = tile.decompose %rl_evkB_4c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1723 = tile.matmul_modq %a_1715, %b_1719 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1724 = tile.matmul_modq %a_1715, %b_1720 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1725 = tile.matmul_modq %a_1715, %b_1721 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1726 = tile.matmul_modq %a_1715, %b_1722 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1727 = tile.matmul_modq %a_1716, %b_1719 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1728 = tile.matmul_modq %a_1716, %b_1720 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1729 = tile.matmul_modq %a_1716, %b_1721 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1730 = tile.matmul_modq %a_1716, %b_1722 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1731 = tile.matmul_modq %a_1717, %b_1719 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1732 = tile.matmul_modq %a_1717, %b_1720 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1733 = tile.matmul_modq %a_1717, %b_1721 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1734 = tile.matmul_modq %a_1717, %b_1722 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1735 = tile.matmul_modq %a_1718, %b_1719 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1736 = tile.matmul_modq %a_1718, %b_1720 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1737 = tile.matmul_modq %a_1718, %b_1721 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1738 = tile.matmul_modq %a_1718, %b_1722 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1739 = tile.add_modq %P_1723 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1740 = tile.add_modq %P_1724, %P_1727 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1741 = tile.add_modq %P_1725, %P_1728, %P_1731 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1742 = tile.add_modq %P_1726, %P_1729, %P_1732, %P_1735 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1743 = tile.add_modq %P_1730, %P_1733, %P_1736 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1744 = tile.add_modq %P_1734, %P_1737 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1745 = tile.add_modq %P_1738 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_4 = tile.recombine_modq %C_1739, %C_1740, %C_1741, %C_1742, %C_1743, %C_1744, %C_1745 {radix = 128 : i64, q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY4 = tile.mul_modq %rl_X4, %rl_evkB_4 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %col_fa187 = tile.reshape %rl_AY4 {shape = [16, 1], q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa188 = tile.reshape %rl_BY4 {shape = [16, 1], q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa189 = tile.concat %col_fa187, %col_fa188 {axis = 1 : i64, q = 268434401 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1746, %a_1747, %a_1748, %a_1749 = tile.decompose %W_i_268434401 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1750, %b_1751, %b_1752, %b_1753 = tile.decompose %rhs_fa189 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1754 = tile.matmul_modq %a_1746, %b_1750 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1755 = tile.matmul_modq %a_1746, %b_1751 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1756 = tile.matmul_modq %a_1746, %b_1752 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1757 = tile.matmul_modq %a_1746, %b_1753 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1758 = tile.matmul_modq %a_1747, %b_1750 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1759 = tile.matmul_modq %a_1747, %b_1751 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1760 = tile.matmul_modq %a_1747, %b_1752 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1761 = tile.matmul_modq %a_1747, %b_1753 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1762 = tile.matmul_modq %a_1748, %b_1750 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1763 = tile.matmul_modq %a_1748, %b_1751 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1764 = tile.matmul_modq %a_1748, %b_1752 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1765 = tile.matmul_modq %a_1748, %b_1753 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1766 = tile.matmul_modq %a_1749, %b_1750 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1767 = tile.matmul_modq %a_1749, %b_1751 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1768 = tile.matmul_modq %a_1749, %b_1752 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1769 = tile.matmul_modq %a_1749, %b_1753 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1770 = tile.add_modq %P_1754 {q = 268434401 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1771 = tile.add_modq %P_1755, %P_1758 {q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1772 = tile.add_modq %P_1756, %P_1759, %P_1762 {q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1773 = tile.add_modq %P_1757, %P_1760, %P_1763, %P_1766 {q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1774 = tile.add_modq %P_1761, %P_1764, %P_1767 {q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1775 = tile.add_modq %P_1765, %P_1768 {q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1776 = tile.add_modq %P_1769 {q = 268434401 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa190 = tile.recombine_modq %C_1770, %C_1771, %C_1772, %C_1773, %C_1774, %C_1775, %C_1776 {radix = 128 : i64, q = 268434401 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa191, %yc_fa192 = tile.split %out_fa190 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_Aa4 = tile.reshape %yc_fa191 {shape = [16], q = 268434401 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_Ba4 = tile.reshape %yc_fa192 {shape = [16], q = 268434401 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_evkB_5c = rns.const {name = "rl_evkB_5c", q = 268433761 : i64} : tensor<16xi32>
    %a_1777, %a_1778, %a_1779, %a_1780 = tile.decompose %W_f_268433761 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1781, %b_1782, %b_1783, %b_1784 = tile.decompose %rl_evkB_5c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1785 = tile.matmul_modq %a_1777, %b_1781 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1786 = tile.matmul_modq %a_1777, %b_1782 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1787 = tile.matmul_modq %a_1777, %b_1783 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1788 = tile.matmul_modq %a_1777, %b_1784 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1789 = tile.matmul_modq %a_1778, %b_1781 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1790 = tile.matmul_modq %a_1778, %b_1782 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1791 = tile.matmul_modq %a_1778, %b_1783 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1792 = tile.matmul_modq %a_1778, %b_1784 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1793 = tile.matmul_modq %a_1779, %b_1781 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1794 = tile.matmul_modq %a_1779, %b_1782 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1795 = tile.matmul_modq %a_1779, %b_1783 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1796 = tile.matmul_modq %a_1779, %b_1784 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1797 = tile.matmul_modq %a_1780, %b_1781 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1798 = tile.matmul_modq %a_1780, %b_1782 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1799 = tile.matmul_modq %a_1780, %b_1783 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1800 = tile.matmul_modq %a_1780, %b_1784 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1801 = tile.add_modq %P_1785 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1802 = tile.add_modq %P_1786, %P_1789 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1803 = tile.add_modq %P_1787, %P_1790, %P_1793 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1804 = tile.add_modq %P_1788, %P_1791, %P_1794, %P_1797 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1805 = tile.add_modq %P_1792, %P_1795, %P_1798 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1806 = tile.add_modq %P_1796, %P_1799 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1807 = tile.add_modq %P_1800 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_5 = tile.recombine_modq %C_1801, %C_1802, %C_1803, %C_1804, %C_1805, %C_1806, %C_1807 {radix = 128 : i64, q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY5 = tile.mul_modq %rl_X5, %rl_evkB_5 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %col_fa193 = tile.reshape %rl_AY5 {shape = [16, 1], q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %col_fa194 = tile.reshape %rl_BY5 {shape = [16, 1], q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16x1xi32>
    %rhs_fa195 = tile.concat %col_fa193, %col_fa194 {axis = 1 : i64, q = 268433761 : i64} : (tensor<16x1xi32>, tensor<16x1xi32>) -> tensor<16x2xi32>
    %a_1808, %a_1809, %a_1810, %a_1811 = tile.decompose %W_i_268433761 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1812, %b_1813, %b_1814, %b_1815 = tile.decompose %rhs_fa195 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x2xi32>) -> (tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>, tensor<16x2xi8>)
    %P_1816 = tile.matmul_modq %a_1808, %b_1812 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1817 = tile.matmul_modq %a_1808, %b_1813 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1818 = tile.matmul_modq %a_1808, %b_1814 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1819 = tile.matmul_modq %a_1808, %b_1815 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1820 = tile.matmul_modq %a_1809, %b_1812 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1821 = tile.matmul_modq %a_1809, %b_1813 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1822 = tile.matmul_modq %a_1809, %b_1814 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1823 = tile.matmul_modq %a_1809, %b_1815 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1824 = tile.matmul_modq %a_1810, %b_1812 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1825 = tile.matmul_modq %a_1810, %b_1813 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1826 = tile.matmul_modq %a_1810, %b_1814 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1827 = tile.matmul_modq %a_1810, %b_1815 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1828 = tile.matmul_modq %a_1811, %b_1812 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1829 = tile.matmul_modq %a_1811, %b_1813 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1830 = tile.matmul_modq %a_1811, %b_1814 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %P_1831 = tile.matmul_modq %a_1811, %b_1815 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16x2xi8>) -> tensor<16x2xi32>
    %C_1832 = tile.add_modq %P_1816 {q = 268433761 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1833 = tile.add_modq %P_1817, %P_1820 {q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1834 = tile.add_modq %P_1818, %P_1821, %P_1824 {q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1835 = tile.add_modq %P_1819, %P_1822, %P_1825, %P_1828 {q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1836 = tile.add_modq %P_1823, %P_1826, %P_1829 {q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1837 = tile.add_modq %P_1827, %P_1830 {q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %C_1838 = tile.add_modq %P_1831 {q = 268433761 : i64} : (tensor<16x2xi32>) -> tensor<16x2xi32>
    %out_fa196 = tile.recombine_modq %C_1832, %C_1833, %C_1834, %C_1835, %C_1836, %C_1837, %C_1838 {radix = 128 : i64, q = 268433761 : i64} : (tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>, tensor<16x2xi32>) -> tensor<16x2xi32>
    %yc_fa197, %yc_fa198 = tile.split %out_fa196 {axis = 1 : i64, parts = 2 : i64} : (tensor<16x2xi32>) -> (tensor<16x1xi32>, tensor<16x1xi32>)
    %rl_Aa5 = tile.reshape %yc_fa197 {shape = [16], q = 268433761 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
    %rl_Ba5 = tile.reshape %yc_fa198 {shape = [16], q = 268433761 : i64} : (tensor<16x1xi32>) -> tensor<16xi32>
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
    %a_1839, %a_1840, %a_1841, %a_1842 = tile.decompose %fbcD__268434721__268434401__268433761__268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1843, %b_1844, %b_1845, %b_1846 = tile.decompose %rl_AxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1847 = tile.matmul_modq %a_1839, %b_1843 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1848 = tile.matmul_modq %a_1839, %b_1844 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1849 = tile.matmul_modq %a_1839, %b_1845 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1850 = tile.matmul_modq %a_1839, %b_1846 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1851 = tile.matmul_modq %a_1840, %b_1843 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1852 = tile.matmul_modq %a_1840, %b_1844 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1853 = tile.matmul_modq %a_1840, %b_1845 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1854 = tile.matmul_modq %a_1840, %b_1846 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1855 = tile.matmul_modq %a_1841, %b_1843 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1856 = tile.matmul_modq %a_1841, %b_1844 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1857 = tile.matmul_modq %a_1841, %b_1845 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1858 = tile.matmul_modq %a_1841, %b_1846 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1859 = tile.matmul_modq %a_1842, %b_1843 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1860 = tile.matmul_modq %a_1842, %b_1844 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1861 = tile.matmul_modq %a_1842, %b_1845 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1862 = tile.matmul_modq %a_1842, %b_1846 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1863 = tile.add_modq %P_1847 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1864 = tile.add_modq %P_1848, %P_1851 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1865 = tile.add_modq %P_1849, %P_1852, %P_1855 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1866 = tile.add_modq %P_1850, %P_1853, %P_1856, %P_1859 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1867 = tile.add_modq %P_1854, %P_1857, %P_1860 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1868 = tile.add_modq %P_1858, %P_1861 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1869 = tile.add_modq %P_1862 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_AxP0_fbc_268435361 = tile.recombine_modq %C_1863, %C_1864, %C_1865, %C_1866, %C_1867, %C_1868, %C_1869 {radix = 128 : i64, q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_AxP0 = tile.reshape %rl_AxP0_fbc_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268434721__268434401__268433761__268435009 = tile.const {name = "fbcD_(268434721, 268434401, 268433761)_268435009"} : tensor<1x3xi32>
    %a_1870, %a_1871, %a_1872, %a_1873 = tile.decompose %fbcD__268434721__268434401__268433761__268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1874, %b_1875, %b_1876, %b_1877 = tile.decompose %rl_AxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1878 = tile.matmul_modq %a_1870, %b_1874 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1879 = tile.matmul_modq %a_1870, %b_1875 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1880 = tile.matmul_modq %a_1870, %b_1876 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1881 = tile.matmul_modq %a_1870, %b_1877 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1882 = tile.matmul_modq %a_1871, %b_1874 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1883 = tile.matmul_modq %a_1871, %b_1875 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1884 = tile.matmul_modq %a_1871, %b_1876 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1885 = tile.matmul_modq %a_1871, %b_1877 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1886 = tile.matmul_modq %a_1872, %b_1874 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1887 = tile.matmul_modq %a_1872, %b_1875 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1888 = tile.matmul_modq %a_1872, %b_1876 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1889 = tile.matmul_modq %a_1872, %b_1877 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1890 = tile.matmul_modq %a_1873, %b_1874 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1891 = tile.matmul_modq %a_1873, %b_1875 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1892 = tile.matmul_modq %a_1873, %b_1876 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1893 = tile.matmul_modq %a_1873, %b_1877 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1894 = tile.add_modq %P_1878 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1895 = tile.add_modq %P_1879, %P_1882 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1896 = tile.add_modq %P_1880, %P_1883, %P_1886 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1897 = tile.add_modq %P_1881, %P_1884, %P_1887, %P_1890 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1898 = tile.add_modq %P_1885, %P_1888, %P_1891 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1899 = tile.add_modq %P_1889, %P_1892 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1900 = tile.add_modq %P_1893 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_AxP0_fbc_268435009 = tile.recombine_modq %C_1894, %C_1895, %C_1896, %C_1897, %C_1898, %C_1899, %C_1900 {radix = 128 : i64, q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_AxP1 = tile.reshape %rl_AxP0_fbc_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268434721__268434401__268433761__268434977 = tile.const {name = "fbcD_(268434721, 268434401, 268433761)_268434977"} : tensor<1x3xi32>
    %a_1901, %a_1902, %a_1903, %a_1904 = tile.decompose %fbcD__268434721__268434401__268433761__268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1905, %b_1906, %b_1907, %b_1908 = tile.decompose %rl_AxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1909 = tile.matmul_modq %a_1901, %b_1905 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1910 = tile.matmul_modq %a_1901, %b_1906 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1911 = tile.matmul_modq %a_1901, %b_1907 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1912 = tile.matmul_modq %a_1901, %b_1908 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1913 = tile.matmul_modq %a_1902, %b_1905 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1914 = tile.matmul_modq %a_1902, %b_1906 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1915 = tile.matmul_modq %a_1902, %b_1907 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1916 = tile.matmul_modq %a_1902, %b_1908 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1917 = tile.matmul_modq %a_1903, %b_1905 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1918 = tile.matmul_modq %a_1903, %b_1906 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1919 = tile.matmul_modq %a_1903, %b_1907 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1920 = tile.matmul_modq %a_1903, %b_1908 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1921 = tile.matmul_modq %a_1904, %b_1905 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1922 = tile.matmul_modq %a_1904, %b_1906 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1923 = tile.matmul_modq %a_1904, %b_1907 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1924 = tile.matmul_modq %a_1904, %b_1908 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1925 = tile.add_modq %P_1909 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1926 = tile.add_modq %P_1910, %P_1913 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1927 = tile.add_modq %P_1911, %P_1914, %P_1917 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1928 = tile.add_modq %P_1912, %P_1915, %P_1918, %P_1921 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1929 = tile.add_modq %P_1916, %P_1919, %P_1922 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1930 = tile.add_modq %P_1920, %P_1923 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1931 = tile.add_modq %P_1924 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_AxP0_fbc_268434977 = tile.recombine_modq %C_1925, %C_1926, %C_1927, %C_1928, %C_1929, %C_1930, %C_1931 {radix = 128 : i64, q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_1932, %a_1933, %a_1934, %a_1935 = tile.decompose %fbcD__268434721__268434401__268433761__268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1936, %b_1937, %b_1938, %b_1939 = tile.decompose %rl_BxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1940 = tile.matmul_modq %a_1932, %b_1936 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1941 = tile.matmul_modq %a_1932, %b_1937 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1942 = tile.matmul_modq %a_1932, %b_1938 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1943 = tile.matmul_modq %a_1932, %b_1939 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1944 = tile.matmul_modq %a_1933, %b_1936 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1945 = tile.matmul_modq %a_1933, %b_1937 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1946 = tile.matmul_modq %a_1933, %b_1938 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1947 = tile.matmul_modq %a_1933, %b_1939 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1948 = tile.matmul_modq %a_1934, %b_1936 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1949 = tile.matmul_modq %a_1934, %b_1937 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1950 = tile.matmul_modq %a_1934, %b_1938 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1951 = tile.matmul_modq %a_1934, %b_1939 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1952 = tile.matmul_modq %a_1935, %b_1936 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1953 = tile.matmul_modq %a_1935, %b_1937 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1954 = tile.matmul_modq %a_1935, %b_1938 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1955 = tile.matmul_modq %a_1935, %b_1939 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1956 = tile.add_modq %P_1940 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1957 = tile.add_modq %P_1941, %P_1944 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1958 = tile.add_modq %P_1942, %P_1945, %P_1948 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1959 = tile.add_modq %P_1943, %P_1946, %P_1949, %P_1952 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1960 = tile.add_modq %P_1947, %P_1950, %P_1953 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1961 = tile.add_modq %P_1951, %P_1954 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1962 = tile.add_modq %P_1955 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_BxP0_fbc_268435361 = tile.recombine_modq %C_1956, %C_1957, %C_1958, %C_1959, %C_1960, %C_1961, %C_1962 {radix = 128 : i64, q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_BxP0 = tile.reshape %rl_BxP0_fbc_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_1963, %a_1964, %a_1965, %a_1966 = tile.decompose %fbcD__268434721__268434401__268433761__268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1967, %b_1968, %b_1969, %b_1970 = tile.decompose %rl_BxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1971 = tile.matmul_modq %a_1963, %b_1967 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1972 = tile.matmul_modq %a_1963, %b_1968 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1973 = tile.matmul_modq %a_1963, %b_1969 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1974 = tile.matmul_modq %a_1963, %b_1970 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1975 = tile.matmul_modq %a_1964, %b_1967 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1976 = tile.matmul_modq %a_1964, %b_1968 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1977 = tile.matmul_modq %a_1964, %b_1969 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1978 = tile.matmul_modq %a_1964, %b_1970 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1979 = tile.matmul_modq %a_1965, %b_1967 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1980 = tile.matmul_modq %a_1965, %b_1968 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1981 = tile.matmul_modq %a_1965, %b_1969 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1982 = tile.matmul_modq %a_1965, %b_1970 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1983 = tile.matmul_modq %a_1966, %b_1967 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1984 = tile.matmul_modq %a_1966, %b_1968 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1985 = tile.matmul_modq %a_1966, %b_1969 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1986 = tile.matmul_modq %a_1966, %b_1970 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1987 = tile.add_modq %P_1971 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1988 = tile.add_modq %P_1972, %P_1975 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1989 = tile.add_modq %P_1973, %P_1976, %P_1979 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1990 = tile.add_modq %P_1974, %P_1977, %P_1980, %P_1983 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1991 = tile.add_modq %P_1978, %P_1981, %P_1984 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1992 = tile.add_modq %P_1982, %P_1985 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1993 = tile.add_modq %P_1986 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_BxP0_fbc_268435009 = tile.recombine_modq %C_1987, %C_1988, %C_1989, %C_1990, %C_1991, %C_1992, %C_1993 {radix = 128 : i64, q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_BxP1 = tile.reshape %rl_BxP0_fbc_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_1994, %a_1995, %a_1996, %a_1997 = tile.decompose %fbcD__268434721__268434401__268433761__268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1998, %b_1999, %b_2000, %b_2001 = tile.decompose %rl_BxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_2002 = tile.matmul_modq %a_1994, %b_1998 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2003 = tile.matmul_modq %a_1994, %b_1999 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2004 = tile.matmul_modq %a_1994, %b_2000 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2005 = tile.matmul_modq %a_1994, %b_2001 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2006 = tile.matmul_modq %a_1995, %b_1998 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2007 = tile.matmul_modq %a_1995, %b_1999 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2008 = tile.matmul_modq %a_1995, %b_2000 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2009 = tile.matmul_modq %a_1995, %b_2001 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2010 = tile.matmul_modq %a_1996, %b_1998 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2011 = tile.matmul_modq %a_1996, %b_1999 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2012 = tile.matmul_modq %a_1996, %b_2000 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2013 = tile.matmul_modq %a_1996, %b_2001 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2014 = tile.matmul_modq %a_1997, %b_1998 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2015 = tile.matmul_modq %a_1997, %b_1999 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2016 = tile.matmul_modq %a_1997, %b_2000 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2017 = tile.matmul_modq %a_1997, %b_2001 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_2018 = tile.add_modq %P_2002 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2019 = tile.add_modq %P_2003, %P_2006 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2020 = tile.add_modq %P_2004, %P_2007, %P_2010 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2021 = tile.add_modq %P_2005, %P_2008, %P_2011, %P_2014 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2022 = tile.add_modq %P_2009, %P_2012, %P_2015 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2023 = tile.add_modq %P_2013, %P_2016 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2024 = tile.add_modq %P_2017 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_BxP0_fbc_268434977 = tile.recombine_modq %C_2018, %C_2019, %C_2020, %C_2021, %C_2022, %C_2023, %C_2024 {radix = 128 : i64, q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
