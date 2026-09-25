module {
  func.func @tensor_l3(%c0_0: tensor<16xi32>, %c1_0: tensor<16xi32>, %d0_0: tensor<16xi32>, %d1_0: tensor<16xi32>, %c0_1: tensor<16xi32>, %c1_1: tensor<16xi32>, %d0_1: tensor<16xi32>, %d1_1: tensor<16xi32>, %c0_2: tensor<16xi32>, %c1_2: tensor<16xi32>, %d0_2: tensor<16xi32>, %d1_2: tensor<16xi32>, %c0_3: tensor<16xi32>, %c1_3: tensor<16xi32>, %d0_3: tensor<16xi32>, %d1_3: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %psi_268435361_16 = tile.const {name = "psi_268435361_16", q = 268435361 : i64} : tensor<16xi32>
    %C0_0_s1 = rns.mul %c0_0, %psi_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C0_0_s2 = tile.reshape %C0_0_s1 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %Wf_268435361_16_0 = tile.const {name = "Wf_268435361_16_0", q = 268435361 : i64} : tensor<4x4xi32>
    %a_1, %a_2, %a_3, %a_4 = tile.decompose %Wf_268435361_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_5, %b_6, %b_7, %b_8 = tile.decompose %C0_0_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_9 = tile.matmul_modq %a_1, %b_5 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_10 = tile.matmul_modq %a_1, %b_6 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_11 = tile.matmul_modq %a_1, %b_7 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_12 = tile.matmul_modq %a_1, %b_8 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_13 = tile.matmul_modq %a_2, %b_5 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_14 = tile.matmul_modq %a_2, %b_6 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_15 = tile.matmul_modq %a_2, %b_7 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_16 = tile.matmul_modq %a_2, %b_8 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_17 = tile.matmul_modq %a_3, %b_5 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_18 = tile.matmul_modq %a_3, %b_6 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_19 = tile.matmul_modq %a_3, %b_7 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_20 = tile.matmul_modq %a_3, %b_8 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_21 = tile.matmul_modq %a_4, %b_5 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_22 = tile.matmul_modq %a_4, %b_6 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_23 = tile.matmul_modq %a_4, %b_7 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_24 = tile.matmul_modq %a_4, %b_8 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_25 = tile.add_modq %P_9 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_26 = tile.add_modq %P_10, %P_13 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_27 = tile.add_modq %P_11, %P_14, %P_17 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_28 = tile.add_modq %P_12, %P_15, %P_18, %P_21 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_29 = tile.add_modq %P_16, %P_19, %P_22 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_30 = tile.add_modq %P_20, %P_23 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_31 = tile.add_modq %P_24 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_0_s3 = tile.recombine_modq %C_25, %C_26, %C_27, %C_28, %C_29, %C_30, %C_31 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268435361_16_0 = tile.const {name = "Tf_268435361_16_0", q = 268435361 : i64} : tensor<4x4xi32>
    %C0_0_s4 = tile.mul_modq %C0_0_s3, %Tf_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_0_s5 = tile.transpose %C0_0_s4 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_0_s6 = tile.reshape %C0_0_s5 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wf_268435361_16_1 = tile.const {name = "Wf_268435361_16_1", q = 268435361 : i64} : tensor<4x4xi32>
    %a_32, %a_33, %a_34, %a_35 = tile.decompose %Wf_268435361_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_36, %b_37, %b_38, %b_39 = tile.decompose %C0_0_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_40 = tile.matmul_modq %a_32, %b_36 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_41 = tile.matmul_modq %a_32, %b_37 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_42 = tile.matmul_modq %a_32, %b_38 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_43 = tile.matmul_modq %a_32, %b_39 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_44 = tile.matmul_modq %a_33, %b_36 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_45 = tile.matmul_modq %a_33, %b_37 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_46 = tile.matmul_modq %a_33, %b_38 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_47 = tile.matmul_modq %a_33, %b_39 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_48 = tile.matmul_modq %a_34, %b_36 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_49 = tile.matmul_modq %a_34, %b_37 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_50 = tile.matmul_modq %a_34, %b_38 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_51 = tile.matmul_modq %a_34, %b_39 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_52 = tile.matmul_modq %a_35, %b_36 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_53 = tile.matmul_modq %a_35, %b_37 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_54 = tile.matmul_modq %a_35, %b_38 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_55 = tile.matmul_modq %a_35, %b_39 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_56 = tile.add_modq %P_40 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_57 = tile.add_modq %P_41, %P_44 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_58 = tile.add_modq %P_42, %P_45, %P_48 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_59 = tile.add_modq %P_43, %P_46, %P_49, %P_52 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_60 = tile.add_modq %P_47, %P_50, %P_53 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_61 = tile.add_modq %P_51, %P_54 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_62 = tile.add_modq %P_55 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_0_s7 = tile.recombine_modq %C_56, %C_57, %C_58, %C_59, %C_60, %C_61, %C_62 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268435361_16_1 = tile.const {name = "Tf_268435361_16_1", q = 268435361 : i64} : tensor<4x4xi32>
    %C0_0_s8 = tile.mul_modq %C0_0_s7, %Tf_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_0_s9 = tile.transpose %C0_0_s8 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_0 = tile.reshape %C0_0_s9 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %C1_0_s1 = rns.mul %c1_0, %psi_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C1_0_s2 = tile.reshape %C1_0_s1 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_63, %a_64, %a_65, %a_66 = tile.decompose %Wf_268435361_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_67, %b_68, %b_69, %b_70 = tile.decompose %C1_0_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_71 = tile.matmul_modq %a_63, %b_67 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_72 = tile.matmul_modq %a_63, %b_68 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_73 = tile.matmul_modq %a_63, %b_69 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_74 = tile.matmul_modq %a_63, %b_70 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_75 = tile.matmul_modq %a_64, %b_67 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_76 = tile.matmul_modq %a_64, %b_68 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_77 = tile.matmul_modq %a_64, %b_69 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_78 = tile.matmul_modq %a_64, %b_70 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_79 = tile.matmul_modq %a_65, %b_67 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_80 = tile.matmul_modq %a_65, %b_68 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_81 = tile.matmul_modq %a_65, %b_69 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_82 = tile.matmul_modq %a_65, %b_70 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_83 = tile.matmul_modq %a_66, %b_67 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_84 = tile.matmul_modq %a_66, %b_68 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_85 = tile.matmul_modq %a_66, %b_69 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_86 = tile.matmul_modq %a_66, %b_70 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_87 = tile.add_modq %P_71 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_88 = tile.add_modq %P_72, %P_75 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_89 = tile.add_modq %P_73, %P_76, %P_79 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_90 = tile.add_modq %P_74, %P_77, %P_80, %P_83 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_91 = tile.add_modq %P_78, %P_81, %P_84 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_92 = tile.add_modq %P_82, %P_85 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_93 = tile.add_modq %P_86 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s3 = tile.recombine_modq %C_87, %C_88, %C_89, %C_90, %C_91, %C_92, %C_93 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s4 = tile.mul_modq %C1_0_s3, %Tf_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s5 = tile.transpose %C1_0_s4 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s6 = tile.reshape %C1_0_s5 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_94, %a_95, %a_96, %a_97 = tile.decompose %Wf_268435361_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_98, %b_99, %b_100, %b_101 = tile.decompose %C1_0_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_102 = tile.matmul_modq %a_94, %b_98 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_103 = tile.matmul_modq %a_94, %b_99 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_104 = tile.matmul_modq %a_94, %b_100 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_105 = tile.matmul_modq %a_94, %b_101 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_106 = tile.matmul_modq %a_95, %b_98 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_107 = tile.matmul_modq %a_95, %b_99 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_108 = tile.matmul_modq %a_95, %b_100 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_109 = tile.matmul_modq %a_95, %b_101 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_110 = tile.matmul_modq %a_96, %b_98 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_111 = tile.matmul_modq %a_96, %b_99 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_112 = tile.matmul_modq %a_96, %b_100 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_113 = tile.matmul_modq %a_96, %b_101 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_114 = tile.matmul_modq %a_97, %b_98 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_115 = tile.matmul_modq %a_97, %b_99 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_116 = tile.matmul_modq %a_97, %b_100 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_117 = tile.matmul_modq %a_97, %b_101 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_118 = tile.add_modq %P_102 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_119 = tile.add_modq %P_103, %P_106 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_120 = tile.add_modq %P_104, %P_107, %P_110 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_121 = tile.add_modq %P_105, %P_108, %P_111, %P_114 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_122 = tile.add_modq %P_109, %P_112, %P_115 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_123 = tile.add_modq %P_113, %P_116 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_124 = tile.add_modq %P_117 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s7 = tile.recombine_modq %C_118, %C_119, %C_120, %C_121, %C_122, %C_123, %C_124 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s8 = tile.mul_modq %C1_0_s7, %Tf_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0_s9 = tile.transpose %C1_0_s8 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_0 = tile.reshape %C1_0_s9 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D0_0_s1 = rns.mul %d0_0, %psi_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D0_0_s2 = tile.reshape %D0_0_s1 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_125, %a_126, %a_127, %a_128 = tile.decompose %Wf_268435361_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_129, %b_130, %b_131, %b_132 = tile.decompose %D0_0_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_133 = tile.matmul_modq %a_125, %b_129 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_134 = tile.matmul_modq %a_125, %b_130 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_135 = tile.matmul_modq %a_125, %b_131 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_136 = tile.matmul_modq %a_125, %b_132 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_137 = tile.matmul_modq %a_126, %b_129 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_138 = tile.matmul_modq %a_126, %b_130 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_139 = tile.matmul_modq %a_126, %b_131 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_140 = tile.matmul_modq %a_126, %b_132 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_141 = tile.matmul_modq %a_127, %b_129 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_142 = tile.matmul_modq %a_127, %b_130 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_143 = tile.matmul_modq %a_127, %b_131 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_144 = tile.matmul_modq %a_127, %b_132 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_145 = tile.matmul_modq %a_128, %b_129 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_146 = tile.matmul_modq %a_128, %b_130 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_147 = tile.matmul_modq %a_128, %b_131 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_148 = tile.matmul_modq %a_128, %b_132 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_149 = tile.add_modq %P_133 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_150 = tile.add_modq %P_134, %P_137 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_151 = tile.add_modq %P_135, %P_138, %P_141 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_152 = tile.add_modq %P_136, %P_139, %P_142, %P_145 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_153 = tile.add_modq %P_140, %P_143, %P_146 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_154 = tile.add_modq %P_144, %P_147 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_155 = tile.add_modq %P_148 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s3 = tile.recombine_modq %C_149, %C_150, %C_151, %C_152, %C_153, %C_154, %C_155 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s4 = tile.mul_modq %D0_0_s3, %Tf_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s5 = tile.transpose %D0_0_s4 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s6 = tile.reshape %D0_0_s5 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_156, %a_157, %a_158, %a_159 = tile.decompose %Wf_268435361_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_160, %b_161, %b_162, %b_163 = tile.decompose %D0_0_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_164 = tile.matmul_modq %a_156, %b_160 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_165 = tile.matmul_modq %a_156, %b_161 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_166 = tile.matmul_modq %a_156, %b_162 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_167 = tile.matmul_modq %a_156, %b_163 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_168 = tile.matmul_modq %a_157, %b_160 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_169 = tile.matmul_modq %a_157, %b_161 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_170 = tile.matmul_modq %a_157, %b_162 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_171 = tile.matmul_modq %a_157, %b_163 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_172 = tile.matmul_modq %a_158, %b_160 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_173 = tile.matmul_modq %a_158, %b_161 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_174 = tile.matmul_modq %a_158, %b_162 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_175 = tile.matmul_modq %a_158, %b_163 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_176 = tile.matmul_modq %a_159, %b_160 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_177 = tile.matmul_modq %a_159, %b_161 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_178 = tile.matmul_modq %a_159, %b_162 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_179 = tile.matmul_modq %a_159, %b_163 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_180 = tile.add_modq %P_164 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_181 = tile.add_modq %P_165, %P_168 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_182 = tile.add_modq %P_166, %P_169, %P_172 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_183 = tile.add_modq %P_167, %P_170, %P_173, %P_176 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_184 = tile.add_modq %P_171, %P_174, %P_177 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_185 = tile.add_modq %P_175, %P_178 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_186 = tile.add_modq %P_179 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s7 = tile.recombine_modq %C_180, %C_181, %C_182, %C_183, %C_184, %C_185, %C_186 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s8 = tile.mul_modq %D0_0_s7, %Tf_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0_s9 = tile.transpose %D0_0_s8 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_0 = tile.reshape %D0_0_s9 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D1_0_s1 = rns.mul %d1_0, %psi_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D1_0_s2 = tile.reshape %D1_0_s1 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_187, %a_188, %a_189, %a_190 = tile.decompose %Wf_268435361_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_191, %b_192, %b_193, %b_194 = tile.decompose %D1_0_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_195 = tile.matmul_modq %a_187, %b_191 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_196 = tile.matmul_modq %a_187, %b_192 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_197 = tile.matmul_modq %a_187, %b_193 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_198 = tile.matmul_modq %a_187, %b_194 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_199 = tile.matmul_modq %a_188, %b_191 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_200 = tile.matmul_modq %a_188, %b_192 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_201 = tile.matmul_modq %a_188, %b_193 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_202 = tile.matmul_modq %a_188, %b_194 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_203 = tile.matmul_modq %a_189, %b_191 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_204 = tile.matmul_modq %a_189, %b_192 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_205 = tile.matmul_modq %a_189, %b_193 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_206 = tile.matmul_modq %a_189, %b_194 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_207 = tile.matmul_modq %a_190, %b_191 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_208 = tile.matmul_modq %a_190, %b_192 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_209 = tile.matmul_modq %a_190, %b_193 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_210 = tile.matmul_modq %a_190, %b_194 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_211 = tile.add_modq %P_195 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_212 = tile.add_modq %P_196, %P_199 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_213 = tile.add_modq %P_197, %P_200, %P_203 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_214 = tile.add_modq %P_198, %P_201, %P_204, %P_207 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_215 = tile.add_modq %P_202, %P_205, %P_208 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_216 = tile.add_modq %P_206, %P_209 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_217 = tile.add_modq %P_210 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s3 = tile.recombine_modq %C_211, %C_212, %C_213, %C_214, %C_215, %C_216, %C_217 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s4 = tile.mul_modq %D1_0_s3, %Tf_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s5 = tile.transpose %D1_0_s4 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s6 = tile.reshape %D1_0_s5 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_218, %a_219, %a_220, %a_221 = tile.decompose %Wf_268435361_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_222, %b_223, %b_224, %b_225 = tile.decompose %D1_0_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_226 = tile.matmul_modq %a_218, %b_222 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_227 = tile.matmul_modq %a_218, %b_223 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_228 = tile.matmul_modq %a_218, %b_224 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_229 = tile.matmul_modq %a_218, %b_225 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_230 = tile.matmul_modq %a_219, %b_222 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_231 = tile.matmul_modq %a_219, %b_223 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_232 = tile.matmul_modq %a_219, %b_224 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_233 = tile.matmul_modq %a_219, %b_225 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_234 = tile.matmul_modq %a_220, %b_222 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_235 = tile.matmul_modq %a_220, %b_223 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_236 = tile.matmul_modq %a_220, %b_224 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_237 = tile.matmul_modq %a_220, %b_225 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_238 = tile.matmul_modq %a_221, %b_222 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_239 = tile.matmul_modq %a_221, %b_223 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_240 = tile.matmul_modq %a_221, %b_224 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_241 = tile.matmul_modq %a_221, %b_225 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_242 = tile.add_modq %P_226 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_243 = tile.add_modq %P_227, %P_230 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_244 = tile.add_modq %P_228, %P_231, %P_234 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_245 = tile.add_modq %P_229, %P_232, %P_235, %P_238 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_246 = tile.add_modq %P_233, %P_236, %P_239 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_247 = tile.add_modq %P_237, %P_240 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_248 = tile.add_modq %P_241 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_0_s7 = tile.recombine_modq %C_242, %C_243, %C_244, %C_245, %C_246, %C_247, %C_248 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
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
    %a_249, %a_250, %a_251, %a_252 = tile.decompose %Wi_268435361_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_253, %b_254, %b_255, %b_256 = tile.decompose %e0_0_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_257 = tile.matmul_modq %a_249, %b_253 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_258 = tile.matmul_modq %a_249, %b_254 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_259 = tile.matmul_modq %a_249, %b_255 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_260 = tile.matmul_modq %a_249, %b_256 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_261 = tile.matmul_modq %a_250, %b_253 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_262 = tile.matmul_modq %a_250, %b_254 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_263 = tile.matmul_modq %a_250, %b_255 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_264 = tile.matmul_modq %a_250, %b_256 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_265 = tile.matmul_modq %a_251, %b_253 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_266 = tile.matmul_modq %a_251, %b_254 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_267 = tile.matmul_modq %a_251, %b_255 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_268 = tile.matmul_modq %a_251, %b_256 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_269 = tile.matmul_modq %a_252, %b_253 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_270 = tile.matmul_modq %a_252, %b_254 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_271 = tile.matmul_modq %a_252, %b_255 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_272 = tile.matmul_modq %a_252, %b_256 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_273 = tile.add_modq %P_257 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_274 = tile.add_modq %P_258, %P_261 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_275 = tile.add_modq %P_259, %P_262, %P_265 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_276 = tile.add_modq %P_260, %P_263, %P_266, %P_269 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_277 = tile.add_modq %P_264, %P_267, %P_270 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_278 = tile.add_modq %P_268, %P_271 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_279 = tile.add_modq %P_272 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_0_s4 = tile.recombine_modq %C_273, %C_274, %C_275, %C_276, %C_277, %C_278, %C_279 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_0_s5 = tile.reshape %e0_0_s4 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_0_s6 = tile.transpose %e0_0_s5 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268435361_16_0 = tile.const {name = "Ti_268435361_16_0", q = 268435361 : i64} : tensor<4x4xi32>
    %e0_0_s7 = tile.mul_modq %e0_0_s6, %Ti_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268435361_16_0 = tile.const {name = "Wi_268435361_16_0", q = 268435361 : i64} : tensor<4x4xi32>
    %a_280, %a_281, %a_282, %a_283 = tile.decompose %Wi_268435361_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_284, %b_285, %b_286, %b_287 = tile.decompose %e0_0_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_288 = tile.matmul_modq %a_280, %b_284 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_289 = tile.matmul_modq %a_280, %b_285 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_290 = tile.matmul_modq %a_280, %b_286 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_291 = tile.matmul_modq %a_280, %b_287 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_292 = tile.matmul_modq %a_281, %b_284 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_293 = tile.matmul_modq %a_281, %b_285 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_294 = tile.matmul_modq %a_281, %b_286 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_295 = tile.matmul_modq %a_281, %b_287 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_296 = tile.matmul_modq %a_282, %b_284 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_297 = tile.matmul_modq %a_282, %b_285 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_298 = tile.matmul_modq %a_282, %b_286 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_299 = tile.matmul_modq %a_282, %b_287 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_300 = tile.matmul_modq %a_283, %b_284 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_301 = tile.matmul_modq %a_283, %b_285 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_302 = tile.matmul_modq %a_283, %b_286 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_303 = tile.matmul_modq %a_283, %b_287 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_304 = tile.add_modq %P_288 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_305 = tile.add_modq %P_289, %P_292 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_306 = tile.add_modq %P_290, %P_293, %P_296 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_307 = tile.add_modq %P_291, %P_294, %P_297, %P_300 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_308 = tile.add_modq %P_295, %P_298, %P_301 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_309 = tile.add_modq %P_299, %P_302 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_310 = tile.add_modq %P_303 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_0_s8 = tile.recombine_modq %C_304, %C_305, %C_306, %C_307, %C_308, %C_309, %C_310 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_0_s9 = tile.reshape %e0_0_s8 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %psinv_268435361_16 = tile.const {name = "psinv_268435361_16", q = 268435361 : i64} : tensor<16xi32>
    %e0_0 = rns.mul %e0_0_s9, %psinv_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_0_s1 = tile.reshape %E1_0 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e1_0_s2 = tile.transpose %e1_0_s1 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s3 = tile.mul_modq %e1_0_s2, %Ti_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_311, %a_312, %a_313, %a_314 = tile.decompose %Wi_268435361_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_315, %b_316, %b_317, %b_318 = tile.decompose %e1_0_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_319 = tile.matmul_modq %a_311, %b_315 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_320 = tile.matmul_modq %a_311, %b_316 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_321 = tile.matmul_modq %a_311, %b_317 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_322 = tile.matmul_modq %a_311, %b_318 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_323 = tile.matmul_modq %a_312, %b_315 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_324 = tile.matmul_modq %a_312, %b_316 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_325 = tile.matmul_modq %a_312, %b_317 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_326 = tile.matmul_modq %a_312, %b_318 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_327 = tile.matmul_modq %a_313, %b_315 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_328 = tile.matmul_modq %a_313, %b_316 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_329 = tile.matmul_modq %a_313, %b_317 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_330 = tile.matmul_modq %a_313, %b_318 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_331 = tile.matmul_modq %a_314, %b_315 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_332 = tile.matmul_modq %a_314, %b_316 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_333 = tile.matmul_modq %a_314, %b_317 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_334 = tile.matmul_modq %a_314, %b_318 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_335 = tile.add_modq %P_319 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_336 = tile.add_modq %P_320, %P_323 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_337 = tile.add_modq %P_321, %P_324, %P_327 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_338 = tile.add_modq %P_322, %P_325, %P_328, %P_331 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_339 = tile.add_modq %P_326, %P_329, %P_332 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_340 = tile.add_modq %P_330, %P_333 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_341 = tile.add_modq %P_334 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s4 = tile.recombine_modq %C_335, %C_336, %C_337, %C_338, %C_339, %C_340, %C_341 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s5 = tile.reshape %e1_0_s4 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s6 = tile.transpose %e1_0_s5 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s7 = tile.mul_modq %e1_0_s6, %Ti_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_342, %a_343, %a_344, %a_345 = tile.decompose %Wi_268435361_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_346, %b_347, %b_348, %b_349 = tile.decompose %e1_0_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_350 = tile.matmul_modq %a_342, %b_346 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_351 = tile.matmul_modq %a_342, %b_347 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_352 = tile.matmul_modq %a_342, %b_348 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_353 = tile.matmul_modq %a_342, %b_349 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_354 = tile.matmul_modq %a_343, %b_346 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_355 = tile.matmul_modq %a_343, %b_347 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_356 = tile.matmul_modq %a_343, %b_348 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_357 = tile.matmul_modq %a_343, %b_349 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_358 = tile.matmul_modq %a_344, %b_346 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_359 = tile.matmul_modq %a_344, %b_347 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_360 = tile.matmul_modq %a_344, %b_348 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_361 = tile.matmul_modq %a_344, %b_349 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_362 = tile.matmul_modq %a_345, %b_346 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_363 = tile.matmul_modq %a_345, %b_347 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_364 = tile.matmul_modq %a_345, %b_348 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_365 = tile.matmul_modq %a_345, %b_349 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_366 = tile.add_modq %P_350 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_367 = tile.add_modq %P_351, %P_354 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_368 = tile.add_modq %P_352, %P_355, %P_358 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_369 = tile.add_modq %P_353, %P_356, %P_359, %P_362 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_370 = tile.add_modq %P_357, %P_360, %P_363 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_371 = tile.add_modq %P_361, %P_364 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_372 = tile.add_modq %P_365 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s8 = tile.recombine_modq %C_366, %C_367, %C_368, %C_369, %C_370, %C_371, %C_372 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_0_s9 = tile.reshape %e1_0_s8 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e1_0 = rns.mul %e1_0_s9, %psinv_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_0_s1 = tile.reshape %E2_0 {shape = [4, 4], q = 268435361 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e2_0_s2 = tile.transpose %e2_0_s1 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s3 = tile.mul_modq %e2_0_s2, %Ti_268435361_16_1 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_373, %a_374, %a_375, %a_376 = tile.decompose %Wi_268435361_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_377, %b_378, %b_379, %b_380 = tile.decompose %e2_0_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_381 = tile.matmul_modq %a_373, %b_377 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_382 = tile.matmul_modq %a_373, %b_378 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_383 = tile.matmul_modq %a_373, %b_379 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_384 = tile.matmul_modq %a_373, %b_380 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_385 = tile.matmul_modq %a_374, %b_377 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_386 = tile.matmul_modq %a_374, %b_378 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_387 = tile.matmul_modq %a_374, %b_379 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_388 = tile.matmul_modq %a_374, %b_380 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_389 = tile.matmul_modq %a_375, %b_377 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_390 = tile.matmul_modq %a_375, %b_378 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_391 = tile.matmul_modq %a_375, %b_379 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_392 = tile.matmul_modq %a_375, %b_380 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_393 = tile.matmul_modq %a_376, %b_377 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_394 = tile.matmul_modq %a_376, %b_378 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_395 = tile.matmul_modq %a_376, %b_379 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_396 = tile.matmul_modq %a_376, %b_380 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_397 = tile.add_modq %P_381 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_398 = tile.add_modq %P_382, %P_385 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_399 = tile.add_modq %P_383, %P_386, %P_389 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_400 = tile.add_modq %P_384, %P_387, %P_390, %P_393 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_401 = tile.add_modq %P_388, %P_391, %P_394 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_402 = tile.add_modq %P_392, %P_395 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_403 = tile.add_modq %P_396 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s4 = tile.recombine_modq %C_397, %C_398, %C_399, %C_400, %C_401, %C_402, %C_403 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s5 = tile.reshape %e2_0_s4 {shape = [4, 4], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s6 = tile.transpose %e2_0_s5 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s7 = tile.mul_modq %e2_0_s6, %Ti_268435361_16_0 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_404, %a_405, %a_406, %a_407 = tile.decompose %Wi_268435361_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_408, %b_409, %b_410, %b_411 = tile.decompose %e2_0_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_412 = tile.matmul_modq %a_404, %b_408 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_413 = tile.matmul_modq %a_404, %b_409 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_414 = tile.matmul_modq %a_404, %b_410 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_415 = tile.matmul_modq %a_404, %b_411 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_416 = tile.matmul_modq %a_405, %b_408 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_417 = tile.matmul_modq %a_405, %b_409 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_418 = tile.matmul_modq %a_405, %b_410 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_419 = tile.matmul_modq %a_405, %b_411 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_420 = tile.matmul_modq %a_406, %b_408 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_421 = tile.matmul_modq %a_406, %b_409 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_422 = tile.matmul_modq %a_406, %b_410 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_423 = tile.matmul_modq %a_406, %b_411 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_424 = tile.matmul_modq %a_407, %b_408 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_425 = tile.matmul_modq %a_407, %b_409 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_426 = tile.matmul_modq %a_407, %b_410 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_427 = tile.matmul_modq %a_407, %b_411 {q = 268435361 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_428 = tile.add_modq %P_412 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_429 = tile.add_modq %P_413, %P_416 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_430 = tile.add_modq %P_414, %P_417, %P_420 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_431 = tile.add_modq %P_415, %P_418, %P_421, %P_424 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_432 = tile.add_modq %P_419, %P_422, %P_425 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_433 = tile.add_modq %P_423, %P_426 {q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_434 = tile.add_modq %P_427 {q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s8 = tile.recombine_modq %C_428, %C_429, %C_430, %C_431, %C_432, %C_433, %C_434 {radix = 128 : i64, q = 268435361 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_0_s9 = tile.reshape %e2_0_s8 {shape = [16], q = 268435361 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e2_0 = rns.mul %e2_0_s9, %psinv_268435361_16 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %psi_268435009_16 = tile.const {name = "psi_268435009_16", q = 268435009 : i64} : tensor<16xi32>
    %C0_1_s1 = rns.mul %c0_1, %psi_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C0_1_s2 = tile.reshape %C0_1_s1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %Wf_268435009_16_0 = tile.const {name = "Wf_268435009_16_0", q = 268435009 : i64} : tensor<4x4xi32>
    %a_435, %a_436, %a_437, %a_438 = tile.decompose %Wf_268435009_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_439, %b_440, %b_441, %b_442 = tile.decompose %C0_1_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_443 = tile.matmul_modq %a_435, %b_439 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_444 = tile.matmul_modq %a_435, %b_440 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_445 = tile.matmul_modq %a_435, %b_441 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_446 = tile.matmul_modq %a_435, %b_442 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_447 = tile.matmul_modq %a_436, %b_439 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_448 = tile.matmul_modq %a_436, %b_440 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_449 = tile.matmul_modq %a_436, %b_441 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_450 = tile.matmul_modq %a_436, %b_442 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_451 = tile.matmul_modq %a_437, %b_439 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_452 = tile.matmul_modq %a_437, %b_440 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_453 = tile.matmul_modq %a_437, %b_441 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_454 = tile.matmul_modq %a_437, %b_442 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_455 = tile.matmul_modq %a_438, %b_439 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_456 = tile.matmul_modq %a_438, %b_440 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_457 = tile.matmul_modq %a_438, %b_441 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_458 = tile.matmul_modq %a_438, %b_442 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_459 = tile.add_modq %P_443 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_460 = tile.add_modq %P_444, %P_447 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_461 = tile.add_modq %P_445, %P_448, %P_451 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_462 = tile.add_modq %P_446, %P_449, %P_452, %P_455 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_463 = tile.add_modq %P_450, %P_453, %P_456 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_464 = tile.add_modq %P_454, %P_457 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_465 = tile.add_modq %P_458 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_1_s3 = tile.recombine_modq %C_459, %C_460, %C_461, %C_462, %C_463, %C_464, %C_465 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268435009_16_0 = tile.const {name = "Tf_268435009_16_0", q = 268435009 : i64} : tensor<4x4xi32>
    %C0_1_s4 = tile.mul_modq %C0_1_s3, %Tf_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_1_s5 = tile.transpose %C0_1_s4 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_1_s6 = tile.reshape %C0_1_s5 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wf_268435009_16_1 = tile.const {name = "Wf_268435009_16_1", q = 268435009 : i64} : tensor<4x4xi32>
    %a_466, %a_467, %a_468, %a_469 = tile.decompose %Wf_268435009_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_470, %b_471, %b_472, %b_473 = tile.decompose %C0_1_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_474 = tile.matmul_modq %a_466, %b_470 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_475 = tile.matmul_modq %a_466, %b_471 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_476 = tile.matmul_modq %a_466, %b_472 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_477 = tile.matmul_modq %a_466, %b_473 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_478 = tile.matmul_modq %a_467, %b_470 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_479 = tile.matmul_modq %a_467, %b_471 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_480 = tile.matmul_modq %a_467, %b_472 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_481 = tile.matmul_modq %a_467, %b_473 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_482 = tile.matmul_modq %a_468, %b_470 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_483 = tile.matmul_modq %a_468, %b_471 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_484 = tile.matmul_modq %a_468, %b_472 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_485 = tile.matmul_modq %a_468, %b_473 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_486 = tile.matmul_modq %a_469, %b_470 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_487 = tile.matmul_modq %a_469, %b_471 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_488 = tile.matmul_modq %a_469, %b_472 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_489 = tile.matmul_modq %a_469, %b_473 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_490 = tile.add_modq %P_474 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_491 = tile.add_modq %P_475, %P_478 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_492 = tile.add_modq %P_476, %P_479, %P_482 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_493 = tile.add_modq %P_477, %P_480, %P_483, %P_486 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_494 = tile.add_modq %P_481, %P_484, %P_487 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_495 = tile.add_modq %P_485, %P_488 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_496 = tile.add_modq %P_489 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_1_s7 = tile.recombine_modq %C_490, %C_491, %C_492, %C_493, %C_494, %C_495, %C_496 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268435009_16_1 = tile.const {name = "Tf_268435009_16_1", q = 268435009 : i64} : tensor<4x4xi32>
    %C0_1_s8 = tile.mul_modq %C0_1_s7, %Tf_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_1_s9 = tile.transpose %C0_1_s8 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_1 = tile.reshape %C0_1_s9 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %C1_1_s1 = rns.mul %c1_1, %psi_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C1_1_s2 = tile.reshape %C1_1_s1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_497, %a_498, %a_499, %a_500 = tile.decompose %Wf_268435009_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_501, %b_502, %b_503, %b_504 = tile.decompose %C1_1_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_505 = tile.matmul_modq %a_497, %b_501 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_506 = tile.matmul_modq %a_497, %b_502 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_507 = tile.matmul_modq %a_497, %b_503 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_508 = tile.matmul_modq %a_497, %b_504 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_509 = tile.matmul_modq %a_498, %b_501 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_510 = tile.matmul_modq %a_498, %b_502 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_511 = tile.matmul_modq %a_498, %b_503 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_512 = tile.matmul_modq %a_498, %b_504 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_513 = tile.matmul_modq %a_499, %b_501 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_514 = tile.matmul_modq %a_499, %b_502 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_515 = tile.matmul_modq %a_499, %b_503 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_516 = tile.matmul_modq %a_499, %b_504 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_517 = tile.matmul_modq %a_500, %b_501 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_518 = tile.matmul_modq %a_500, %b_502 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_519 = tile.matmul_modq %a_500, %b_503 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_520 = tile.matmul_modq %a_500, %b_504 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_521 = tile.add_modq %P_505 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_522 = tile.add_modq %P_506, %P_509 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_523 = tile.add_modq %P_507, %P_510, %P_513 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_524 = tile.add_modq %P_508, %P_511, %P_514, %P_517 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_525 = tile.add_modq %P_512, %P_515, %P_518 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_526 = tile.add_modq %P_516, %P_519 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_527 = tile.add_modq %P_520 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s3 = tile.recombine_modq %C_521, %C_522, %C_523, %C_524, %C_525, %C_526, %C_527 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s4 = tile.mul_modq %C1_1_s3, %Tf_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s5 = tile.transpose %C1_1_s4 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s6 = tile.reshape %C1_1_s5 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_528, %a_529, %a_530, %a_531 = tile.decompose %Wf_268435009_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_532, %b_533, %b_534, %b_535 = tile.decompose %C1_1_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_536 = tile.matmul_modq %a_528, %b_532 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_537 = tile.matmul_modq %a_528, %b_533 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_538 = tile.matmul_modq %a_528, %b_534 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_539 = tile.matmul_modq %a_528, %b_535 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_540 = tile.matmul_modq %a_529, %b_532 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_541 = tile.matmul_modq %a_529, %b_533 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_542 = tile.matmul_modq %a_529, %b_534 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_543 = tile.matmul_modq %a_529, %b_535 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_544 = tile.matmul_modq %a_530, %b_532 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_545 = tile.matmul_modq %a_530, %b_533 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_546 = tile.matmul_modq %a_530, %b_534 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_547 = tile.matmul_modq %a_530, %b_535 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_548 = tile.matmul_modq %a_531, %b_532 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_549 = tile.matmul_modq %a_531, %b_533 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_550 = tile.matmul_modq %a_531, %b_534 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_551 = tile.matmul_modq %a_531, %b_535 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_552 = tile.add_modq %P_536 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_553 = tile.add_modq %P_537, %P_540 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_554 = tile.add_modq %P_538, %P_541, %P_544 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_555 = tile.add_modq %P_539, %P_542, %P_545, %P_548 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_556 = tile.add_modq %P_543, %P_546, %P_549 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_557 = tile.add_modq %P_547, %P_550 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_558 = tile.add_modq %P_551 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s7 = tile.recombine_modq %C_552, %C_553, %C_554, %C_555, %C_556, %C_557, %C_558 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s8 = tile.mul_modq %C1_1_s7, %Tf_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1_s9 = tile.transpose %C1_1_s8 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_1 = tile.reshape %C1_1_s9 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D0_1_s1 = rns.mul %d0_1, %psi_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D0_1_s2 = tile.reshape %D0_1_s1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_559, %a_560, %a_561, %a_562 = tile.decompose %Wf_268435009_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_563, %b_564, %b_565, %b_566 = tile.decompose %D0_1_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_567 = tile.matmul_modq %a_559, %b_563 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_568 = tile.matmul_modq %a_559, %b_564 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_569 = tile.matmul_modq %a_559, %b_565 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_570 = tile.matmul_modq %a_559, %b_566 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_571 = tile.matmul_modq %a_560, %b_563 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_572 = tile.matmul_modq %a_560, %b_564 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_573 = tile.matmul_modq %a_560, %b_565 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_574 = tile.matmul_modq %a_560, %b_566 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_575 = tile.matmul_modq %a_561, %b_563 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_576 = tile.matmul_modq %a_561, %b_564 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_577 = tile.matmul_modq %a_561, %b_565 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_578 = tile.matmul_modq %a_561, %b_566 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_579 = tile.matmul_modq %a_562, %b_563 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_580 = tile.matmul_modq %a_562, %b_564 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_581 = tile.matmul_modq %a_562, %b_565 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_582 = tile.matmul_modq %a_562, %b_566 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_583 = tile.add_modq %P_567 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_584 = tile.add_modq %P_568, %P_571 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_585 = tile.add_modq %P_569, %P_572, %P_575 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_586 = tile.add_modq %P_570, %P_573, %P_576, %P_579 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_587 = tile.add_modq %P_574, %P_577, %P_580 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_588 = tile.add_modq %P_578, %P_581 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_589 = tile.add_modq %P_582 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s3 = tile.recombine_modq %C_583, %C_584, %C_585, %C_586, %C_587, %C_588, %C_589 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s4 = tile.mul_modq %D0_1_s3, %Tf_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s5 = tile.transpose %D0_1_s4 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s6 = tile.reshape %D0_1_s5 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_590, %a_591, %a_592, %a_593 = tile.decompose %Wf_268435009_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_594, %b_595, %b_596, %b_597 = tile.decompose %D0_1_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_598 = tile.matmul_modq %a_590, %b_594 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_599 = tile.matmul_modq %a_590, %b_595 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_600 = tile.matmul_modq %a_590, %b_596 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_601 = tile.matmul_modq %a_590, %b_597 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_602 = tile.matmul_modq %a_591, %b_594 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_603 = tile.matmul_modq %a_591, %b_595 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_604 = tile.matmul_modq %a_591, %b_596 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_605 = tile.matmul_modq %a_591, %b_597 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_606 = tile.matmul_modq %a_592, %b_594 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_607 = tile.matmul_modq %a_592, %b_595 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_608 = tile.matmul_modq %a_592, %b_596 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_609 = tile.matmul_modq %a_592, %b_597 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_610 = tile.matmul_modq %a_593, %b_594 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_611 = tile.matmul_modq %a_593, %b_595 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_612 = tile.matmul_modq %a_593, %b_596 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_613 = tile.matmul_modq %a_593, %b_597 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_614 = tile.add_modq %P_598 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_615 = tile.add_modq %P_599, %P_602 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_616 = tile.add_modq %P_600, %P_603, %P_606 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_617 = tile.add_modq %P_601, %P_604, %P_607, %P_610 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_618 = tile.add_modq %P_605, %P_608, %P_611 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_619 = tile.add_modq %P_609, %P_612 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_620 = tile.add_modq %P_613 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s7 = tile.recombine_modq %C_614, %C_615, %C_616, %C_617, %C_618, %C_619, %C_620 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s8 = tile.mul_modq %D0_1_s7, %Tf_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1_s9 = tile.transpose %D0_1_s8 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_1 = tile.reshape %D0_1_s9 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D1_1_s1 = rns.mul %d1_1, %psi_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D1_1_s2 = tile.reshape %D1_1_s1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_621, %a_622, %a_623, %a_624 = tile.decompose %Wf_268435009_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_625, %b_626, %b_627, %b_628 = tile.decompose %D1_1_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_629 = tile.matmul_modq %a_621, %b_625 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_630 = tile.matmul_modq %a_621, %b_626 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_631 = tile.matmul_modq %a_621, %b_627 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_632 = tile.matmul_modq %a_621, %b_628 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_633 = tile.matmul_modq %a_622, %b_625 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_634 = tile.matmul_modq %a_622, %b_626 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_635 = tile.matmul_modq %a_622, %b_627 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_636 = tile.matmul_modq %a_622, %b_628 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_637 = tile.matmul_modq %a_623, %b_625 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_638 = tile.matmul_modq %a_623, %b_626 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_639 = tile.matmul_modq %a_623, %b_627 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_640 = tile.matmul_modq %a_623, %b_628 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_641 = tile.matmul_modq %a_624, %b_625 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_642 = tile.matmul_modq %a_624, %b_626 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_643 = tile.matmul_modq %a_624, %b_627 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_644 = tile.matmul_modq %a_624, %b_628 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_645 = tile.add_modq %P_629 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_646 = tile.add_modq %P_630, %P_633 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_647 = tile.add_modq %P_631, %P_634, %P_637 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_648 = tile.add_modq %P_632, %P_635, %P_638, %P_641 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_649 = tile.add_modq %P_636, %P_639, %P_642 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_650 = tile.add_modq %P_640, %P_643 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_651 = tile.add_modq %P_644 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s3 = tile.recombine_modq %C_645, %C_646, %C_647, %C_648, %C_649, %C_650, %C_651 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s4 = tile.mul_modq %D1_1_s3, %Tf_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s5 = tile.transpose %D1_1_s4 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s6 = tile.reshape %D1_1_s5 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_652, %a_653, %a_654, %a_655 = tile.decompose %Wf_268435009_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_656, %b_657, %b_658, %b_659 = tile.decompose %D1_1_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_660 = tile.matmul_modq %a_652, %b_656 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_661 = tile.matmul_modq %a_652, %b_657 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_662 = tile.matmul_modq %a_652, %b_658 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_663 = tile.matmul_modq %a_652, %b_659 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_664 = tile.matmul_modq %a_653, %b_656 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_665 = tile.matmul_modq %a_653, %b_657 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_666 = tile.matmul_modq %a_653, %b_658 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_667 = tile.matmul_modq %a_653, %b_659 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_668 = tile.matmul_modq %a_654, %b_656 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_669 = tile.matmul_modq %a_654, %b_657 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_670 = tile.matmul_modq %a_654, %b_658 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_671 = tile.matmul_modq %a_654, %b_659 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_672 = tile.matmul_modq %a_655, %b_656 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_673 = tile.matmul_modq %a_655, %b_657 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_674 = tile.matmul_modq %a_655, %b_658 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_675 = tile.matmul_modq %a_655, %b_659 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_676 = tile.add_modq %P_660 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_677 = tile.add_modq %P_661, %P_664 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_678 = tile.add_modq %P_662, %P_665, %P_668 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_679 = tile.add_modq %P_663, %P_666, %P_669, %P_672 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_680 = tile.add_modq %P_667, %P_670, %P_673 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_681 = tile.add_modq %P_671, %P_674 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_682 = tile.add_modq %P_675 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_1_s7 = tile.recombine_modq %C_676, %C_677, %C_678, %C_679, %C_680, %C_681, %C_682 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
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
    %a_683, %a_684, %a_685, %a_686 = tile.decompose %Wi_268435009_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_687, %b_688, %b_689, %b_690 = tile.decompose %e0_1_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_691 = tile.matmul_modq %a_683, %b_687 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_692 = tile.matmul_modq %a_683, %b_688 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_693 = tile.matmul_modq %a_683, %b_689 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_694 = tile.matmul_modq %a_683, %b_690 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_695 = tile.matmul_modq %a_684, %b_687 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_696 = tile.matmul_modq %a_684, %b_688 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_697 = tile.matmul_modq %a_684, %b_689 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_698 = tile.matmul_modq %a_684, %b_690 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_699 = tile.matmul_modq %a_685, %b_687 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_700 = tile.matmul_modq %a_685, %b_688 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_701 = tile.matmul_modq %a_685, %b_689 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_702 = tile.matmul_modq %a_685, %b_690 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_703 = tile.matmul_modq %a_686, %b_687 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_704 = tile.matmul_modq %a_686, %b_688 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_705 = tile.matmul_modq %a_686, %b_689 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_706 = tile.matmul_modq %a_686, %b_690 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_707 = tile.add_modq %P_691 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_708 = tile.add_modq %P_692, %P_695 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_709 = tile.add_modq %P_693, %P_696, %P_699 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_710 = tile.add_modq %P_694, %P_697, %P_700, %P_703 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_711 = tile.add_modq %P_698, %P_701, %P_704 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_712 = tile.add_modq %P_702, %P_705 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_713 = tile.add_modq %P_706 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_1_s4 = tile.recombine_modq %C_707, %C_708, %C_709, %C_710, %C_711, %C_712, %C_713 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_1_s5 = tile.reshape %e0_1_s4 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_1_s6 = tile.transpose %e0_1_s5 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268435009_16_0 = tile.const {name = "Ti_268435009_16_0", q = 268435009 : i64} : tensor<4x4xi32>
    %e0_1_s7 = tile.mul_modq %e0_1_s6, %Ti_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268435009_16_0 = tile.const {name = "Wi_268435009_16_0", q = 268435009 : i64} : tensor<4x4xi32>
    %a_714, %a_715, %a_716, %a_717 = tile.decompose %Wi_268435009_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_718, %b_719, %b_720, %b_721 = tile.decompose %e0_1_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_722 = tile.matmul_modq %a_714, %b_718 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_723 = tile.matmul_modq %a_714, %b_719 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_724 = tile.matmul_modq %a_714, %b_720 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_725 = tile.matmul_modq %a_714, %b_721 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_726 = tile.matmul_modq %a_715, %b_718 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_727 = tile.matmul_modq %a_715, %b_719 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_728 = tile.matmul_modq %a_715, %b_720 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_729 = tile.matmul_modq %a_715, %b_721 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_730 = tile.matmul_modq %a_716, %b_718 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_731 = tile.matmul_modq %a_716, %b_719 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_732 = tile.matmul_modq %a_716, %b_720 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_733 = tile.matmul_modq %a_716, %b_721 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_734 = tile.matmul_modq %a_717, %b_718 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_735 = tile.matmul_modq %a_717, %b_719 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_736 = tile.matmul_modq %a_717, %b_720 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_737 = tile.matmul_modq %a_717, %b_721 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_738 = tile.add_modq %P_722 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_739 = tile.add_modq %P_723, %P_726 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_740 = tile.add_modq %P_724, %P_727, %P_730 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_741 = tile.add_modq %P_725, %P_728, %P_731, %P_734 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_742 = tile.add_modq %P_729, %P_732, %P_735 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_743 = tile.add_modq %P_733, %P_736 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_744 = tile.add_modq %P_737 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_1_s8 = tile.recombine_modq %C_738, %C_739, %C_740, %C_741, %C_742, %C_743, %C_744 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_1_s9 = tile.reshape %e0_1_s8 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %psinv_268435009_16 = tile.const {name = "psinv_268435009_16", q = 268435009 : i64} : tensor<16xi32>
    %e0_1 = rns.mul %e0_1_s9, %psinv_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_1_s1 = tile.reshape %E1_1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e1_1_s2 = tile.transpose %e1_1_s1 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s3 = tile.mul_modq %e1_1_s2, %Ti_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_745, %a_746, %a_747, %a_748 = tile.decompose %Wi_268435009_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_749, %b_750, %b_751, %b_752 = tile.decompose %e1_1_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_753 = tile.matmul_modq %a_745, %b_749 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_754 = tile.matmul_modq %a_745, %b_750 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_755 = tile.matmul_modq %a_745, %b_751 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_756 = tile.matmul_modq %a_745, %b_752 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_757 = tile.matmul_modq %a_746, %b_749 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_758 = tile.matmul_modq %a_746, %b_750 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_759 = tile.matmul_modq %a_746, %b_751 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_760 = tile.matmul_modq %a_746, %b_752 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_761 = tile.matmul_modq %a_747, %b_749 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_762 = tile.matmul_modq %a_747, %b_750 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_763 = tile.matmul_modq %a_747, %b_751 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_764 = tile.matmul_modq %a_747, %b_752 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_765 = tile.matmul_modq %a_748, %b_749 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_766 = tile.matmul_modq %a_748, %b_750 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_767 = tile.matmul_modq %a_748, %b_751 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_768 = tile.matmul_modq %a_748, %b_752 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_769 = tile.add_modq %P_753 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_770 = tile.add_modq %P_754, %P_757 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_771 = tile.add_modq %P_755, %P_758, %P_761 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_772 = tile.add_modq %P_756, %P_759, %P_762, %P_765 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_773 = tile.add_modq %P_760, %P_763, %P_766 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_774 = tile.add_modq %P_764, %P_767 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_775 = tile.add_modq %P_768 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s4 = tile.recombine_modq %C_769, %C_770, %C_771, %C_772, %C_773, %C_774, %C_775 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s5 = tile.reshape %e1_1_s4 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s6 = tile.transpose %e1_1_s5 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s7 = tile.mul_modq %e1_1_s6, %Ti_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_776, %a_777, %a_778, %a_779 = tile.decompose %Wi_268435009_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_780, %b_781, %b_782, %b_783 = tile.decompose %e1_1_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_784 = tile.matmul_modq %a_776, %b_780 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_785 = tile.matmul_modq %a_776, %b_781 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_786 = tile.matmul_modq %a_776, %b_782 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_787 = tile.matmul_modq %a_776, %b_783 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_788 = tile.matmul_modq %a_777, %b_780 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_789 = tile.matmul_modq %a_777, %b_781 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_790 = tile.matmul_modq %a_777, %b_782 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_791 = tile.matmul_modq %a_777, %b_783 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_792 = tile.matmul_modq %a_778, %b_780 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_793 = tile.matmul_modq %a_778, %b_781 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_794 = tile.matmul_modq %a_778, %b_782 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_795 = tile.matmul_modq %a_778, %b_783 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_796 = tile.matmul_modq %a_779, %b_780 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_797 = tile.matmul_modq %a_779, %b_781 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_798 = tile.matmul_modq %a_779, %b_782 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_799 = tile.matmul_modq %a_779, %b_783 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_800 = tile.add_modq %P_784 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_801 = tile.add_modq %P_785, %P_788 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_802 = tile.add_modq %P_786, %P_789, %P_792 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_803 = tile.add_modq %P_787, %P_790, %P_793, %P_796 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_804 = tile.add_modq %P_791, %P_794, %P_797 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_805 = tile.add_modq %P_795, %P_798 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_806 = tile.add_modq %P_799 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s8 = tile.recombine_modq %C_800, %C_801, %C_802, %C_803, %C_804, %C_805, %C_806 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_1_s9 = tile.reshape %e1_1_s8 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e1_1 = rns.mul %e1_1_s9, %psinv_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_1_s1 = tile.reshape %E2_1 {shape = [4, 4], q = 268435009 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e2_1_s2 = tile.transpose %e2_1_s1 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s3 = tile.mul_modq %e2_1_s2, %Ti_268435009_16_1 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_807, %a_808, %a_809, %a_810 = tile.decompose %Wi_268435009_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_811, %b_812, %b_813, %b_814 = tile.decompose %e2_1_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_815 = tile.matmul_modq %a_807, %b_811 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_816 = tile.matmul_modq %a_807, %b_812 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_817 = tile.matmul_modq %a_807, %b_813 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_818 = tile.matmul_modq %a_807, %b_814 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_819 = tile.matmul_modq %a_808, %b_811 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_820 = tile.matmul_modq %a_808, %b_812 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_821 = tile.matmul_modq %a_808, %b_813 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_822 = tile.matmul_modq %a_808, %b_814 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_823 = tile.matmul_modq %a_809, %b_811 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_824 = tile.matmul_modq %a_809, %b_812 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_825 = tile.matmul_modq %a_809, %b_813 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_826 = tile.matmul_modq %a_809, %b_814 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_827 = tile.matmul_modq %a_810, %b_811 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_828 = tile.matmul_modq %a_810, %b_812 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_829 = tile.matmul_modq %a_810, %b_813 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_830 = tile.matmul_modq %a_810, %b_814 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_831 = tile.add_modq %P_815 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_832 = tile.add_modq %P_816, %P_819 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_833 = tile.add_modq %P_817, %P_820, %P_823 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_834 = tile.add_modq %P_818, %P_821, %P_824, %P_827 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_835 = tile.add_modq %P_822, %P_825, %P_828 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_836 = tile.add_modq %P_826, %P_829 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_837 = tile.add_modq %P_830 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s4 = tile.recombine_modq %C_831, %C_832, %C_833, %C_834, %C_835, %C_836, %C_837 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s5 = tile.reshape %e2_1_s4 {shape = [4, 4], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s6 = tile.transpose %e2_1_s5 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s7 = tile.mul_modq %e2_1_s6, %Ti_268435009_16_0 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_838, %a_839, %a_840, %a_841 = tile.decompose %Wi_268435009_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_842, %b_843, %b_844, %b_845 = tile.decompose %e2_1_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_846 = tile.matmul_modq %a_838, %b_842 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_847 = tile.matmul_modq %a_838, %b_843 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_848 = tile.matmul_modq %a_838, %b_844 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_849 = tile.matmul_modq %a_838, %b_845 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_850 = tile.matmul_modq %a_839, %b_842 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_851 = tile.matmul_modq %a_839, %b_843 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_852 = tile.matmul_modq %a_839, %b_844 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_853 = tile.matmul_modq %a_839, %b_845 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_854 = tile.matmul_modq %a_840, %b_842 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_855 = tile.matmul_modq %a_840, %b_843 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_856 = tile.matmul_modq %a_840, %b_844 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_857 = tile.matmul_modq %a_840, %b_845 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_858 = tile.matmul_modq %a_841, %b_842 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_859 = tile.matmul_modq %a_841, %b_843 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_860 = tile.matmul_modq %a_841, %b_844 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_861 = tile.matmul_modq %a_841, %b_845 {q = 268435009 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_862 = tile.add_modq %P_846 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_863 = tile.add_modq %P_847, %P_850 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_864 = tile.add_modq %P_848, %P_851, %P_854 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_865 = tile.add_modq %P_849, %P_852, %P_855, %P_858 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_866 = tile.add_modq %P_853, %P_856, %P_859 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_867 = tile.add_modq %P_857, %P_860 {q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_868 = tile.add_modq %P_861 {q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s8 = tile.recombine_modq %C_862, %C_863, %C_864, %C_865, %C_866, %C_867, %C_868 {radix = 128 : i64, q = 268435009 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_1_s9 = tile.reshape %e2_1_s8 {shape = [16], q = 268435009 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e2_1 = rns.mul %e2_1_s9, %psinv_268435009_16 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %psi_268434977_16 = tile.const {name = "psi_268434977_16", q = 268434977 : i64} : tensor<16xi32>
    %C0_2_s1 = rns.mul %c0_2, %psi_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C0_2_s2 = tile.reshape %C0_2_s1 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %Wf_268434977_16_0 = tile.const {name = "Wf_268434977_16_0", q = 268434977 : i64} : tensor<4x4xi32>
    %a_869, %a_870, %a_871, %a_872 = tile.decompose %Wf_268434977_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_873, %b_874, %b_875, %b_876 = tile.decompose %C0_2_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_877 = tile.matmul_modq %a_869, %b_873 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_878 = tile.matmul_modq %a_869, %b_874 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_879 = tile.matmul_modq %a_869, %b_875 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_880 = tile.matmul_modq %a_869, %b_876 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_881 = tile.matmul_modq %a_870, %b_873 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_882 = tile.matmul_modq %a_870, %b_874 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_883 = tile.matmul_modq %a_870, %b_875 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_884 = tile.matmul_modq %a_870, %b_876 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_885 = tile.matmul_modq %a_871, %b_873 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_886 = tile.matmul_modq %a_871, %b_874 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_887 = tile.matmul_modq %a_871, %b_875 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_888 = tile.matmul_modq %a_871, %b_876 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_889 = tile.matmul_modq %a_872, %b_873 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_890 = tile.matmul_modq %a_872, %b_874 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_891 = tile.matmul_modq %a_872, %b_875 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_892 = tile.matmul_modq %a_872, %b_876 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_893 = tile.add_modq %P_877 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_894 = tile.add_modq %P_878, %P_881 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_895 = tile.add_modq %P_879, %P_882, %P_885 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_896 = tile.add_modq %P_880, %P_883, %P_886, %P_889 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_897 = tile.add_modq %P_884, %P_887, %P_890 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_898 = tile.add_modq %P_888, %P_891 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_899 = tile.add_modq %P_892 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_2_s3 = tile.recombine_modq %C_893, %C_894, %C_895, %C_896, %C_897, %C_898, %C_899 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268434977_16_0 = tile.const {name = "Tf_268434977_16_0", q = 268434977 : i64} : tensor<4x4xi32>
    %C0_2_s4 = tile.mul_modq %C0_2_s3, %Tf_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_2_s5 = tile.transpose %C0_2_s4 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_2_s6 = tile.reshape %C0_2_s5 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wf_268434977_16_1 = tile.const {name = "Wf_268434977_16_1", q = 268434977 : i64} : tensor<4x4xi32>
    %a_900, %a_901, %a_902, %a_903 = tile.decompose %Wf_268434977_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_904, %b_905, %b_906, %b_907 = tile.decompose %C0_2_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_908 = tile.matmul_modq %a_900, %b_904 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_909 = tile.matmul_modq %a_900, %b_905 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_910 = tile.matmul_modq %a_900, %b_906 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_911 = tile.matmul_modq %a_900, %b_907 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_912 = tile.matmul_modq %a_901, %b_904 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_913 = tile.matmul_modq %a_901, %b_905 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_914 = tile.matmul_modq %a_901, %b_906 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_915 = tile.matmul_modq %a_901, %b_907 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_916 = tile.matmul_modq %a_902, %b_904 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_917 = tile.matmul_modq %a_902, %b_905 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_918 = tile.matmul_modq %a_902, %b_906 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_919 = tile.matmul_modq %a_902, %b_907 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_920 = tile.matmul_modq %a_903, %b_904 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_921 = tile.matmul_modq %a_903, %b_905 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_922 = tile.matmul_modq %a_903, %b_906 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_923 = tile.matmul_modq %a_903, %b_907 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_924 = tile.add_modq %P_908 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_925 = tile.add_modq %P_909, %P_912 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_926 = tile.add_modq %P_910, %P_913, %P_916 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_927 = tile.add_modq %P_911, %P_914, %P_917, %P_920 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_928 = tile.add_modq %P_915, %P_918, %P_921 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_929 = tile.add_modq %P_919, %P_922 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_930 = tile.add_modq %P_923 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_2_s7 = tile.recombine_modq %C_924, %C_925, %C_926, %C_927, %C_928, %C_929, %C_930 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268434977_16_1 = tile.const {name = "Tf_268434977_16_1", q = 268434977 : i64} : tensor<4x4xi32>
    %C0_2_s8 = tile.mul_modq %C0_2_s7, %Tf_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_2_s9 = tile.transpose %C0_2_s8 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_2 = tile.reshape %C0_2_s9 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %C1_2_s1 = rns.mul %c1_2, %psi_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C1_2_s2 = tile.reshape %C1_2_s1 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_931, %a_932, %a_933, %a_934 = tile.decompose %Wf_268434977_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_935, %b_936, %b_937, %b_938 = tile.decompose %C1_2_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_939 = tile.matmul_modq %a_931, %b_935 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_940 = tile.matmul_modq %a_931, %b_936 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_941 = tile.matmul_modq %a_931, %b_937 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_942 = tile.matmul_modq %a_931, %b_938 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_943 = tile.matmul_modq %a_932, %b_935 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_944 = tile.matmul_modq %a_932, %b_936 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_945 = tile.matmul_modq %a_932, %b_937 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_946 = tile.matmul_modq %a_932, %b_938 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_947 = tile.matmul_modq %a_933, %b_935 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_948 = tile.matmul_modq %a_933, %b_936 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_949 = tile.matmul_modq %a_933, %b_937 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_950 = tile.matmul_modq %a_933, %b_938 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_951 = tile.matmul_modq %a_934, %b_935 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_952 = tile.matmul_modq %a_934, %b_936 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_953 = tile.matmul_modq %a_934, %b_937 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_954 = tile.matmul_modq %a_934, %b_938 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_955 = tile.add_modq %P_939 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_956 = tile.add_modq %P_940, %P_943 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_957 = tile.add_modq %P_941, %P_944, %P_947 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_958 = tile.add_modq %P_942, %P_945, %P_948, %P_951 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_959 = tile.add_modq %P_946, %P_949, %P_952 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_960 = tile.add_modq %P_950, %P_953 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_961 = tile.add_modq %P_954 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s3 = tile.recombine_modq %C_955, %C_956, %C_957, %C_958, %C_959, %C_960, %C_961 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s4 = tile.mul_modq %C1_2_s3, %Tf_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s5 = tile.transpose %C1_2_s4 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s6 = tile.reshape %C1_2_s5 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_962, %a_963, %a_964, %a_965 = tile.decompose %Wf_268434977_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_966, %b_967, %b_968, %b_969 = tile.decompose %C1_2_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_970 = tile.matmul_modq %a_962, %b_966 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_971 = tile.matmul_modq %a_962, %b_967 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_972 = tile.matmul_modq %a_962, %b_968 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_973 = tile.matmul_modq %a_962, %b_969 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_974 = tile.matmul_modq %a_963, %b_966 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_975 = tile.matmul_modq %a_963, %b_967 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_976 = tile.matmul_modq %a_963, %b_968 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_977 = tile.matmul_modq %a_963, %b_969 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_978 = tile.matmul_modq %a_964, %b_966 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_979 = tile.matmul_modq %a_964, %b_967 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_980 = tile.matmul_modq %a_964, %b_968 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_981 = tile.matmul_modq %a_964, %b_969 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_982 = tile.matmul_modq %a_965, %b_966 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_983 = tile.matmul_modq %a_965, %b_967 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_984 = tile.matmul_modq %a_965, %b_968 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_985 = tile.matmul_modq %a_965, %b_969 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_986 = tile.add_modq %P_970 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_987 = tile.add_modq %P_971, %P_974 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_988 = tile.add_modq %P_972, %P_975, %P_978 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_989 = tile.add_modq %P_973, %P_976, %P_979, %P_982 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_990 = tile.add_modq %P_977, %P_980, %P_983 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_991 = tile.add_modq %P_981, %P_984 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_992 = tile.add_modq %P_985 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s7 = tile.recombine_modq %C_986, %C_987, %C_988, %C_989, %C_990, %C_991, %C_992 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s8 = tile.mul_modq %C1_2_s7, %Tf_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2_s9 = tile.transpose %C1_2_s8 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_2 = tile.reshape %C1_2_s9 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D0_2_s1 = rns.mul %d0_2, %psi_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D0_2_s2 = tile.reshape %D0_2_s1 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_993, %a_994, %a_995, %a_996 = tile.decompose %Wf_268434977_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_997, %b_998, %b_999, %b_1000 = tile.decompose %D0_2_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1001 = tile.matmul_modq %a_993, %b_997 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1002 = tile.matmul_modq %a_993, %b_998 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1003 = tile.matmul_modq %a_993, %b_999 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1004 = tile.matmul_modq %a_993, %b_1000 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1005 = tile.matmul_modq %a_994, %b_997 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1006 = tile.matmul_modq %a_994, %b_998 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1007 = tile.matmul_modq %a_994, %b_999 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1008 = tile.matmul_modq %a_994, %b_1000 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1009 = tile.matmul_modq %a_995, %b_997 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1010 = tile.matmul_modq %a_995, %b_998 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1011 = tile.matmul_modq %a_995, %b_999 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1012 = tile.matmul_modq %a_995, %b_1000 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1013 = tile.matmul_modq %a_996, %b_997 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1014 = tile.matmul_modq %a_996, %b_998 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1015 = tile.matmul_modq %a_996, %b_999 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1016 = tile.matmul_modq %a_996, %b_1000 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1017 = tile.add_modq %P_1001 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1018 = tile.add_modq %P_1002, %P_1005 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1019 = tile.add_modq %P_1003, %P_1006, %P_1009 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1020 = tile.add_modq %P_1004, %P_1007, %P_1010, %P_1013 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1021 = tile.add_modq %P_1008, %P_1011, %P_1014 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1022 = tile.add_modq %P_1012, %P_1015 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1023 = tile.add_modq %P_1016 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s3 = tile.recombine_modq %C_1017, %C_1018, %C_1019, %C_1020, %C_1021, %C_1022, %C_1023 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s4 = tile.mul_modq %D0_2_s3, %Tf_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s5 = tile.transpose %D0_2_s4 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s6 = tile.reshape %D0_2_s5 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1024, %a_1025, %a_1026, %a_1027 = tile.decompose %Wf_268434977_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1028, %b_1029, %b_1030, %b_1031 = tile.decompose %D0_2_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1032 = tile.matmul_modq %a_1024, %b_1028 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1033 = tile.matmul_modq %a_1024, %b_1029 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1034 = tile.matmul_modq %a_1024, %b_1030 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1035 = tile.matmul_modq %a_1024, %b_1031 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1036 = tile.matmul_modq %a_1025, %b_1028 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1037 = tile.matmul_modq %a_1025, %b_1029 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1038 = tile.matmul_modq %a_1025, %b_1030 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1039 = tile.matmul_modq %a_1025, %b_1031 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1040 = tile.matmul_modq %a_1026, %b_1028 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1041 = tile.matmul_modq %a_1026, %b_1029 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1042 = tile.matmul_modq %a_1026, %b_1030 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1043 = tile.matmul_modq %a_1026, %b_1031 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1044 = tile.matmul_modq %a_1027, %b_1028 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1045 = tile.matmul_modq %a_1027, %b_1029 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1046 = tile.matmul_modq %a_1027, %b_1030 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1047 = tile.matmul_modq %a_1027, %b_1031 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1048 = tile.add_modq %P_1032 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1049 = tile.add_modq %P_1033, %P_1036 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1050 = tile.add_modq %P_1034, %P_1037, %P_1040 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1051 = tile.add_modq %P_1035, %P_1038, %P_1041, %P_1044 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1052 = tile.add_modq %P_1039, %P_1042, %P_1045 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1053 = tile.add_modq %P_1043, %P_1046 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1054 = tile.add_modq %P_1047 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s7 = tile.recombine_modq %C_1048, %C_1049, %C_1050, %C_1051, %C_1052, %C_1053, %C_1054 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s8 = tile.mul_modq %D0_2_s7, %Tf_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2_s9 = tile.transpose %D0_2_s8 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_2 = tile.reshape %D0_2_s9 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D1_2_s1 = rns.mul %d1_2, %psi_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D1_2_s2 = tile.reshape %D1_2_s1 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_1055, %a_1056, %a_1057, %a_1058 = tile.decompose %Wf_268434977_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1059, %b_1060, %b_1061, %b_1062 = tile.decompose %D1_2_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1063 = tile.matmul_modq %a_1055, %b_1059 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1064 = tile.matmul_modq %a_1055, %b_1060 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1065 = tile.matmul_modq %a_1055, %b_1061 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1066 = tile.matmul_modq %a_1055, %b_1062 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1067 = tile.matmul_modq %a_1056, %b_1059 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1068 = tile.matmul_modq %a_1056, %b_1060 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1069 = tile.matmul_modq %a_1056, %b_1061 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1070 = tile.matmul_modq %a_1056, %b_1062 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1071 = tile.matmul_modq %a_1057, %b_1059 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1072 = tile.matmul_modq %a_1057, %b_1060 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1073 = tile.matmul_modq %a_1057, %b_1061 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1074 = tile.matmul_modq %a_1057, %b_1062 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1075 = tile.matmul_modq %a_1058, %b_1059 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1076 = tile.matmul_modq %a_1058, %b_1060 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1077 = tile.matmul_modq %a_1058, %b_1061 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1078 = tile.matmul_modq %a_1058, %b_1062 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1079 = tile.add_modq %P_1063 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1080 = tile.add_modq %P_1064, %P_1067 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1081 = tile.add_modq %P_1065, %P_1068, %P_1071 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1082 = tile.add_modq %P_1066, %P_1069, %P_1072, %P_1075 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1083 = tile.add_modq %P_1070, %P_1073, %P_1076 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1084 = tile.add_modq %P_1074, %P_1077 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1085 = tile.add_modq %P_1078 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s3 = tile.recombine_modq %C_1079, %C_1080, %C_1081, %C_1082, %C_1083, %C_1084, %C_1085 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s4 = tile.mul_modq %D1_2_s3, %Tf_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s5 = tile.transpose %D1_2_s4 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s6 = tile.reshape %D1_2_s5 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1086, %a_1087, %a_1088, %a_1089 = tile.decompose %Wf_268434977_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1090, %b_1091, %b_1092, %b_1093 = tile.decompose %D1_2_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1094 = tile.matmul_modq %a_1086, %b_1090 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1095 = tile.matmul_modq %a_1086, %b_1091 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1096 = tile.matmul_modq %a_1086, %b_1092 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1097 = tile.matmul_modq %a_1086, %b_1093 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1098 = tile.matmul_modq %a_1087, %b_1090 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1099 = tile.matmul_modq %a_1087, %b_1091 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1100 = tile.matmul_modq %a_1087, %b_1092 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1101 = tile.matmul_modq %a_1087, %b_1093 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1102 = tile.matmul_modq %a_1088, %b_1090 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1103 = tile.matmul_modq %a_1088, %b_1091 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1104 = tile.matmul_modq %a_1088, %b_1092 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1105 = tile.matmul_modq %a_1088, %b_1093 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1106 = tile.matmul_modq %a_1089, %b_1090 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1107 = tile.matmul_modq %a_1089, %b_1091 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1108 = tile.matmul_modq %a_1089, %b_1092 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1109 = tile.matmul_modq %a_1089, %b_1093 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1110 = tile.add_modq %P_1094 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1111 = tile.add_modq %P_1095, %P_1098 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1112 = tile.add_modq %P_1096, %P_1099, %P_1102 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1113 = tile.add_modq %P_1097, %P_1100, %P_1103, %P_1106 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1114 = tile.add_modq %P_1101, %P_1104, %P_1107 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1115 = tile.add_modq %P_1105, %P_1108 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1116 = tile.add_modq %P_1109 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_2_s7 = tile.recombine_modq %C_1110, %C_1111, %C_1112, %C_1113, %C_1114, %C_1115, %C_1116 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
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
    %a_1117, %a_1118, %a_1119, %a_1120 = tile.decompose %Wi_268434977_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1121, %b_1122, %b_1123, %b_1124 = tile.decompose %e0_2_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1125 = tile.matmul_modq %a_1117, %b_1121 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1126 = tile.matmul_modq %a_1117, %b_1122 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1127 = tile.matmul_modq %a_1117, %b_1123 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1128 = tile.matmul_modq %a_1117, %b_1124 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1129 = tile.matmul_modq %a_1118, %b_1121 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1130 = tile.matmul_modq %a_1118, %b_1122 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1131 = tile.matmul_modq %a_1118, %b_1123 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1132 = tile.matmul_modq %a_1118, %b_1124 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1133 = tile.matmul_modq %a_1119, %b_1121 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1134 = tile.matmul_modq %a_1119, %b_1122 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1135 = tile.matmul_modq %a_1119, %b_1123 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1136 = tile.matmul_modq %a_1119, %b_1124 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1137 = tile.matmul_modq %a_1120, %b_1121 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1138 = tile.matmul_modq %a_1120, %b_1122 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1139 = tile.matmul_modq %a_1120, %b_1123 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1140 = tile.matmul_modq %a_1120, %b_1124 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1141 = tile.add_modq %P_1125 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1142 = tile.add_modq %P_1126, %P_1129 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1143 = tile.add_modq %P_1127, %P_1130, %P_1133 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1144 = tile.add_modq %P_1128, %P_1131, %P_1134, %P_1137 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1145 = tile.add_modq %P_1132, %P_1135, %P_1138 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1146 = tile.add_modq %P_1136, %P_1139 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1147 = tile.add_modq %P_1140 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_2_s4 = tile.recombine_modq %C_1141, %C_1142, %C_1143, %C_1144, %C_1145, %C_1146, %C_1147 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_2_s5 = tile.reshape %e0_2_s4 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_2_s6 = tile.transpose %e0_2_s5 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268434977_16_0 = tile.const {name = "Ti_268434977_16_0", q = 268434977 : i64} : tensor<4x4xi32>
    %e0_2_s7 = tile.mul_modq %e0_2_s6, %Ti_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268434977_16_0 = tile.const {name = "Wi_268434977_16_0", q = 268434977 : i64} : tensor<4x4xi32>
    %a_1148, %a_1149, %a_1150, %a_1151 = tile.decompose %Wi_268434977_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1152, %b_1153, %b_1154, %b_1155 = tile.decompose %e0_2_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1156 = tile.matmul_modq %a_1148, %b_1152 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1157 = tile.matmul_modq %a_1148, %b_1153 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1158 = tile.matmul_modq %a_1148, %b_1154 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1159 = tile.matmul_modq %a_1148, %b_1155 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1160 = tile.matmul_modq %a_1149, %b_1152 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1161 = tile.matmul_modq %a_1149, %b_1153 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1162 = tile.matmul_modq %a_1149, %b_1154 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1163 = tile.matmul_modq %a_1149, %b_1155 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1164 = tile.matmul_modq %a_1150, %b_1152 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1165 = tile.matmul_modq %a_1150, %b_1153 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1166 = tile.matmul_modq %a_1150, %b_1154 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1167 = tile.matmul_modq %a_1150, %b_1155 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1168 = tile.matmul_modq %a_1151, %b_1152 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1169 = tile.matmul_modq %a_1151, %b_1153 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1170 = tile.matmul_modq %a_1151, %b_1154 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1171 = tile.matmul_modq %a_1151, %b_1155 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1172 = tile.add_modq %P_1156 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1173 = tile.add_modq %P_1157, %P_1160 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1174 = tile.add_modq %P_1158, %P_1161, %P_1164 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1175 = tile.add_modq %P_1159, %P_1162, %P_1165, %P_1168 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1176 = tile.add_modq %P_1163, %P_1166, %P_1169 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1177 = tile.add_modq %P_1167, %P_1170 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1178 = tile.add_modq %P_1171 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_2_s8 = tile.recombine_modq %C_1172, %C_1173, %C_1174, %C_1175, %C_1176, %C_1177, %C_1178 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_2_s9 = tile.reshape %e0_2_s8 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %psinv_268434977_16 = tile.const {name = "psinv_268434977_16", q = 268434977 : i64} : tensor<16xi32>
    %e0_2 = rns.mul %e0_2_s9, %psinv_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_2_s1 = tile.reshape %E1_2 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e1_2_s2 = tile.transpose %e1_2_s1 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s3 = tile.mul_modq %e1_2_s2, %Ti_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1179, %a_1180, %a_1181, %a_1182 = tile.decompose %Wi_268434977_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1183, %b_1184, %b_1185, %b_1186 = tile.decompose %e1_2_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1187 = tile.matmul_modq %a_1179, %b_1183 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1188 = tile.matmul_modq %a_1179, %b_1184 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1189 = tile.matmul_modq %a_1179, %b_1185 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1190 = tile.matmul_modq %a_1179, %b_1186 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1191 = tile.matmul_modq %a_1180, %b_1183 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1192 = tile.matmul_modq %a_1180, %b_1184 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1193 = tile.matmul_modq %a_1180, %b_1185 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1194 = tile.matmul_modq %a_1180, %b_1186 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1195 = tile.matmul_modq %a_1181, %b_1183 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1196 = tile.matmul_modq %a_1181, %b_1184 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1197 = tile.matmul_modq %a_1181, %b_1185 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1198 = tile.matmul_modq %a_1181, %b_1186 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1199 = tile.matmul_modq %a_1182, %b_1183 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1200 = tile.matmul_modq %a_1182, %b_1184 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1201 = tile.matmul_modq %a_1182, %b_1185 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1202 = tile.matmul_modq %a_1182, %b_1186 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1203 = tile.add_modq %P_1187 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1204 = tile.add_modq %P_1188, %P_1191 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1205 = tile.add_modq %P_1189, %P_1192, %P_1195 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1206 = tile.add_modq %P_1190, %P_1193, %P_1196, %P_1199 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1207 = tile.add_modq %P_1194, %P_1197, %P_1200 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1208 = tile.add_modq %P_1198, %P_1201 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1209 = tile.add_modq %P_1202 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s4 = tile.recombine_modq %C_1203, %C_1204, %C_1205, %C_1206, %C_1207, %C_1208, %C_1209 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s5 = tile.reshape %e1_2_s4 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s6 = tile.transpose %e1_2_s5 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s7 = tile.mul_modq %e1_2_s6, %Ti_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1210, %a_1211, %a_1212, %a_1213 = tile.decompose %Wi_268434977_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1214, %b_1215, %b_1216, %b_1217 = tile.decompose %e1_2_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1218 = tile.matmul_modq %a_1210, %b_1214 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1219 = tile.matmul_modq %a_1210, %b_1215 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1220 = tile.matmul_modq %a_1210, %b_1216 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1221 = tile.matmul_modq %a_1210, %b_1217 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1222 = tile.matmul_modq %a_1211, %b_1214 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1223 = tile.matmul_modq %a_1211, %b_1215 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1224 = tile.matmul_modq %a_1211, %b_1216 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1225 = tile.matmul_modq %a_1211, %b_1217 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1226 = tile.matmul_modq %a_1212, %b_1214 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1227 = tile.matmul_modq %a_1212, %b_1215 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1228 = tile.matmul_modq %a_1212, %b_1216 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1229 = tile.matmul_modq %a_1212, %b_1217 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1230 = tile.matmul_modq %a_1213, %b_1214 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1231 = tile.matmul_modq %a_1213, %b_1215 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1232 = tile.matmul_modq %a_1213, %b_1216 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1233 = tile.matmul_modq %a_1213, %b_1217 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1234 = tile.add_modq %P_1218 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1235 = tile.add_modq %P_1219, %P_1222 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1236 = tile.add_modq %P_1220, %P_1223, %P_1226 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1237 = tile.add_modq %P_1221, %P_1224, %P_1227, %P_1230 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1238 = tile.add_modq %P_1225, %P_1228, %P_1231 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1239 = tile.add_modq %P_1229, %P_1232 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1240 = tile.add_modq %P_1233 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s8 = tile.recombine_modq %C_1234, %C_1235, %C_1236, %C_1237, %C_1238, %C_1239, %C_1240 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_2_s9 = tile.reshape %e1_2_s8 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e1_2 = rns.mul %e1_2_s9, %psinv_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_2_s1 = tile.reshape %E2_2 {shape = [4, 4], q = 268434977 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e2_2_s2 = tile.transpose %e2_2_s1 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s3 = tile.mul_modq %e2_2_s2, %Ti_268434977_16_1 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1241, %a_1242, %a_1243, %a_1244 = tile.decompose %Wi_268434977_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1245, %b_1246, %b_1247, %b_1248 = tile.decompose %e2_2_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1249 = tile.matmul_modq %a_1241, %b_1245 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1250 = tile.matmul_modq %a_1241, %b_1246 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1251 = tile.matmul_modq %a_1241, %b_1247 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1252 = tile.matmul_modq %a_1241, %b_1248 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1253 = tile.matmul_modq %a_1242, %b_1245 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1254 = tile.matmul_modq %a_1242, %b_1246 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1255 = tile.matmul_modq %a_1242, %b_1247 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1256 = tile.matmul_modq %a_1242, %b_1248 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1257 = tile.matmul_modq %a_1243, %b_1245 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1258 = tile.matmul_modq %a_1243, %b_1246 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1259 = tile.matmul_modq %a_1243, %b_1247 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1260 = tile.matmul_modq %a_1243, %b_1248 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1261 = tile.matmul_modq %a_1244, %b_1245 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1262 = tile.matmul_modq %a_1244, %b_1246 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1263 = tile.matmul_modq %a_1244, %b_1247 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1264 = tile.matmul_modq %a_1244, %b_1248 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1265 = tile.add_modq %P_1249 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1266 = tile.add_modq %P_1250, %P_1253 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1267 = tile.add_modq %P_1251, %P_1254, %P_1257 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1268 = tile.add_modq %P_1252, %P_1255, %P_1258, %P_1261 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1269 = tile.add_modq %P_1256, %P_1259, %P_1262 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1270 = tile.add_modq %P_1260, %P_1263 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1271 = tile.add_modq %P_1264 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s4 = tile.recombine_modq %C_1265, %C_1266, %C_1267, %C_1268, %C_1269, %C_1270, %C_1271 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s5 = tile.reshape %e2_2_s4 {shape = [4, 4], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s6 = tile.transpose %e2_2_s5 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s7 = tile.mul_modq %e2_2_s6, %Ti_268434977_16_0 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1272, %a_1273, %a_1274, %a_1275 = tile.decompose %Wi_268434977_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1276, %b_1277, %b_1278, %b_1279 = tile.decompose %e2_2_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1280 = tile.matmul_modq %a_1272, %b_1276 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1281 = tile.matmul_modq %a_1272, %b_1277 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1282 = tile.matmul_modq %a_1272, %b_1278 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1283 = tile.matmul_modq %a_1272, %b_1279 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1284 = tile.matmul_modq %a_1273, %b_1276 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1285 = tile.matmul_modq %a_1273, %b_1277 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1286 = tile.matmul_modq %a_1273, %b_1278 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1287 = tile.matmul_modq %a_1273, %b_1279 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1288 = tile.matmul_modq %a_1274, %b_1276 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1289 = tile.matmul_modq %a_1274, %b_1277 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1290 = tile.matmul_modq %a_1274, %b_1278 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1291 = tile.matmul_modq %a_1274, %b_1279 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1292 = tile.matmul_modq %a_1275, %b_1276 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1293 = tile.matmul_modq %a_1275, %b_1277 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1294 = tile.matmul_modq %a_1275, %b_1278 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1295 = tile.matmul_modq %a_1275, %b_1279 {q = 268434977 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1296 = tile.add_modq %P_1280 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1297 = tile.add_modq %P_1281, %P_1284 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1298 = tile.add_modq %P_1282, %P_1285, %P_1288 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1299 = tile.add_modq %P_1283, %P_1286, %P_1289, %P_1292 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1300 = tile.add_modq %P_1287, %P_1290, %P_1293 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1301 = tile.add_modq %P_1291, %P_1294 {q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1302 = tile.add_modq %P_1295 {q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s8 = tile.recombine_modq %C_1296, %C_1297, %C_1298, %C_1299, %C_1300, %C_1301, %C_1302 {radix = 128 : i64, q = 268434977 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_2_s9 = tile.reshape %e2_2_s8 {shape = [16], q = 268434977 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e2_2 = rns.mul %e2_2_s9, %psinv_268434977_16 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %psi_268434721_16 = tile.const {name = "psi_268434721_16", q = 268434721 : i64} : tensor<16xi32>
    %C0_3_s1 = rns.mul %c0_3, %psi_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C0_3_s2 = tile.reshape %C0_3_s1 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %Wf_268434721_16_0 = tile.const {name = "Wf_268434721_16_0", q = 268434721 : i64} : tensor<4x4xi32>
    %a_1303, %a_1304, %a_1305, %a_1306 = tile.decompose %Wf_268434721_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1307, %b_1308, %b_1309, %b_1310 = tile.decompose %C0_3_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1311 = tile.matmul_modq %a_1303, %b_1307 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1312 = tile.matmul_modq %a_1303, %b_1308 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1313 = tile.matmul_modq %a_1303, %b_1309 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1314 = tile.matmul_modq %a_1303, %b_1310 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1315 = tile.matmul_modq %a_1304, %b_1307 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1316 = tile.matmul_modq %a_1304, %b_1308 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1317 = tile.matmul_modq %a_1304, %b_1309 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1318 = tile.matmul_modq %a_1304, %b_1310 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1319 = tile.matmul_modq %a_1305, %b_1307 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1320 = tile.matmul_modq %a_1305, %b_1308 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1321 = tile.matmul_modq %a_1305, %b_1309 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1322 = tile.matmul_modq %a_1305, %b_1310 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1323 = tile.matmul_modq %a_1306, %b_1307 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1324 = tile.matmul_modq %a_1306, %b_1308 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1325 = tile.matmul_modq %a_1306, %b_1309 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1326 = tile.matmul_modq %a_1306, %b_1310 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1327 = tile.add_modq %P_1311 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1328 = tile.add_modq %P_1312, %P_1315 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1329 = tile.add_modq %P_1313, %P_1316, %P_1319 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1330 = tile.add_modq %P_1314, %P_1317, %P_1320, %P_1323 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1331 = tile.add_modq %P_1318, %P_1321, %P_1324 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1332 = tile.add_modq %P_1322, %P_1325 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1333 = tile.add_modq %P_1326 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_3_s3 = tile.recombine_modq %C_1327, %C_1328, %C_1329, %C_1330, %C_1331, %C_1332, %C_1333 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268434721_16_0 = tile.const {name = "Tf_268434721_16_0", q = 268434721 : i64} : tensor<4x4xi32>
    %C0_3_s4 = tile.mul_modq %C0_3_s3, %Tf_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_3_s5 = tile.transpose %C0_3_s4 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_3_s6 = tile.reshape %C0_3_s5 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wf_268434721_16_1 = tile.const {name = "Wf_268434721_16_1", q = 268434721 : i64} : tensor<4x4xi32>
    %a_1334, %a_1335, %a_1336, %a_1337 = tile.decompose %Wf_268434721_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1338, %b_1339, %b_1340, %b_1341 = tile.decompose %C0_3_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1342 = tile.matmul_modq %a_1334, %b_1338 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1343 = tile.matmul_modq %a_1334, %b_1339 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1344 = tile.matmul_modq %a_1334, %b_1340 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1345 = tile.matmul_modq %a_1334, %b_1341 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1346 = tile.matmul_modq %a_1335, %b_1338 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1347 = tile.matmul_modq %a_1335, %b_1339 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1348 = tile.matmul_modq %a_1335, %b_1340 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1349 = tile.matmul_modq %a_1335, %b_1341 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1350 = tile.matmul_modq %a_1336, %b_1338 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1351 = tile.matmul_modq %a_1336, %b_1339 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1352 = tile.matmul_modq %a_1336, %b_1340 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1353 = tile.matmul_modq %a_1336, %b_1341 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1354 = tile.matmul_modq %a_1337, %b_1338 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1355 = tile.matmul_modq %a_1337, %b_1339 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1356 = tile.matmul_modq %a_1337, %b_1340 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1357 = tile.matmul_modq %a_1337, %b_1341 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1358 = tile.add_modq %P_1342 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1359 = tile.add_modq %P_1343, %P_1346 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1360 = tile.add_modq %P_1344, %P_1347, %P_1350 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1361 = tile.add_modq %P_1345, %P_1348, %P_1351, %P_1354 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1362 = tile.add_modq %P_1349, %P_1352, %P_1355 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1363 = tile.add_modq %P_1353, %P_1356 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1364 = tile.add_modq %P_1357 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_3_s7 = tile.recombine_modq %C_1358, %C_1359, %C_1360, %C_1361, %C_1362, %C_1363, %C_1364 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Tf_268434721_16_1 = tile.const {name = "Tf_268434721_16_1", q = 268434721 : i64} : tensor<4x4xi32>
    %C0_3_s8 = tile.mul_modq %C0_3_s7, %Tf_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_3_s9 = tile.transpose %C0_3_s8 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C0_3 = tile.reshape %C0_3_s9 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %C1_3_s1 = rns.mul %c1_3, %psi_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C1_3_s2 = tile.reshape %C1_3_s1 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_1365, %a_1366, %a_1367, %a_1368 = tile.decompose %Wf_268434721_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1369, %b_1370, %b_1371, %b_1372 = tile.decompose %C1_3_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1373 = tile.matmul_modq %a_1365, %b_1369 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1374 = tile.matmul_modq %a_1365, %b_1370 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1375 = tile.matmul_modq %a_1365, %b_1371 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1376 = tile.matmul_modq %a_1365, %b_1372 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1377 = tile.matmul_modq %a_1366, %b_1369 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1378 = tile.matmul_modq %a_1366, %b_1370 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1379 = tile.matmul_modq %a_1366, %b_1371 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1380 = tile.matmul_modq %a_1366, %b_1372 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1381 = tile.matmul_modq %a_1367, %b_1369 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1382 = tile.matmul_modq %a_1367, %b_1370 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1383 = tile.matmul_modq %a_1367, %b_1371 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1384 = tile.matmul_modq %a_1367, %b_1372 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1385 = tile.matmul_modq %a_1368, %b_1369 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1386 = tile.matmul_modq %a_1368, %b_1370 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1387 = tile.matmul_modq %a_1368, %b_1371 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1388 = tile.matmul_modq %a_1368, %b_1372 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1389 = tile.add_modq %P_1373 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1390 = tile.add_modq %P_1374, %P_1377 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1391 = tile.add_modq %P_1375, %P_1378, %P_1381 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1392 = tile.add_modq %P_1376, %P_1379, %P_1382, %P_1385 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1393 = tile.add_modq %P_1380, %P_1383, %P_1386 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1394 = tile.add_modq %P_1384, %P_1387 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1395 = tile.add_modq %P_1388 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s3 = tile.recombine_modq %C_1389, %C_1390, %C_1391, %C_1392, %C_1393, %C_1394, %C_1395 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s4 = tile.mul_modq %C1_3_s3, %Tf_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s5 = tile.transpose %C1_3_s4 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s6 = tile.reshape %C1_3_s5 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1396, %a_1397, %a_1398, %a_1399 = tile.decompose %Wf_268434721_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1400, %b_1401, %b_1402, %b_1403 = tile.decompose %C1_3_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1404 = tile.matmul_modq %a_1396, %b_1400 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1405 = tile.matmul_modq %a_1396, %b_1401 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1406 = tile.matmul_modq %a_1396, %b_1402 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1407 = tile.matmul_modq %a_1396, %b_1403 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1408 = tile.matmul_modq %a_1397, %b_1400 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1409 = tile.matmul_modq %a_1397, %b_1401 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1410 = tile.matmul_modq %a_1397, %b_1402 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1411 = tile.matmul_modq %a_1397, %b_1403 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1412 = tile.matmul_modq %a_1398, %b_1400 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1413 = tile.matmul_modq %a_1398, %b_1401 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1414 = tile.matmul_modq %a_1398, %b_1402 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1415 = tile.matmul_modq %a_1398, %b_1403 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1416 = tile.matmul_modq %a_1399, %b_1400 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1417 = tile.matmul_modq %a_1399, %b_1401 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1418 = tile.matmul_modq %a_1399, %b_1402 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1419 = tile.matmul_modq %a_1399, %b_1403 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1420 = tile.add_modq %P_1404 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1421 = tile.add_modq %P_1405, %P_1408 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1422 = tile.add_modq %P_1406, %P_1409, %P_1412 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1423 = tile.add_modq %P_1407, %P_1410, %P_1413, %P_1416 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1424 = tile.add_modq %P_1411, %P_1414, %P_1417 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1425 = tile.add_modq %P_1415, %P_1418 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1426 = tile.add_modq %P_1419 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s7 = tile.recombine_modq %C_1420, %C_1421, %C_1422, %C_1423, %C_1424, %C_1425, %C_1426 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s8 = tile.mul_modq %C1_3_s7, %Tf_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3_s9 = tile.transpose %C1_3_s8 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C1_3 = tile.reshape %C1_3_s9 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D0_3_s1 = rns.mul %d0_3, %psi_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D0_3_s2 = tile.reshape %D0_3_s1 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_1427, %a_1428, %a_1429, %a_1430 = tile.decompose %Wf_268434721_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1431, %b_1432, %b_1433, %b_1434 = tile.decompose %D0_3_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1435 = tile.matmul_modq %a_1427, %b_1431 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1436 = tile.matmul_modq %a_1427, %b_1432 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1437 = tile.matmul_modq %a_1427, %b_1433 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1438 = tile.matmul_modq %a_1427, %b_1434 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1439 = tile.matmul_modq %a_1428, %b_1431 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1440 = tile.matmul_modq %a_1428, %b_1432 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1441 = tile.matmul_modq %a_1428, %b_1433 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1442 = tile.matmul_modq %a_1428, %b_1434 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1443 = tile.matmul_modq %a_1429, %b_1431 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1444 = tile.matmul_modq %a_1429, %b_1432 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1445 = tile.matmul_modq %a_1429, %b_1433 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1446 = tile.matmul_modq %a_1429, %b_1434 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1447 = tile.matmul_modq %a_1430, %b_1431 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1448 = tile.matmul_modq %a_1430, %b_1432 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1449 = tile.matmul_modq %a_1430, %b_1433 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1450 = tile.matmul_modq %a_1430, %b_1434 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1451 = tile.add_modq %P_1435 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1452 = tile.add_modq %P_1436, %P_1439 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1453 = tile.add_modq %P_1437, %P_1440, %P_1443 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1454 = tile.add_modq %P_1438, %P_1441, %P_1444, %P_1447 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1455 = tile.add_modq %P_1442, %P_1445, %P_1448 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1456 = tile.add_modq %P_1446, %P_1449 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1457 = tile.add_modq %P_1450 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s3 = tile.recombine_modq %C_1451, %C_1452, %C_1453, %C_1454, %C_1455, %C_1456, %C_1457 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s4 = tile.mul_modq %D0_3_s3, %Tf_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s5 = tile.transpose %D0_3_s4 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s6 = tile.reshape %D0_3_s5 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1458, %a_1459, %a_1460, %a_1461 = tile.decompose %Wf_268434721_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1462, %b_1463, %b_1464, %b_1465 = tile.decompose %D0_3_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1466 = tile.matmul_modq %a_1458, %b_1462 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1467 = tile.matmul_modq %a_1458, %b_1463 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1468 = tile.matmul_modq %a_1458, %b_1464 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1469 = tile.matmul_modq %a_1458, %b_1465 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1470 = tile.matmul_modq %a_1459, %b_1462 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1471 = tile.matmul_modq %a_1459, %b_1463 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1472 = tile.matmul_modq %a_1459, %b_1464 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1473 = tile.matmul_modq %a_1459, %b_1465 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1474 = tile.matmul_modq %a_1460, %b_1462 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1475 = tile.matmul_modq %a_1460, %b_1463 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1476 = tile.matmul_modq %a_1460, %b_1464 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1477 = tile.matmul_modq %a_1460, %b_1465 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1478 = tile.matmul_modq %a_1461, %b_1462 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1479 = tile.matmul_modq %a_1461, %b_1463 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1480 = tile.matmul_modq %a_1461, %b_1464 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1481 = tile.matmul_modq %a_1461, %b_1465 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1482 = tile.add_modq %P_1466 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1483 = tile.add_modq %P_1467, %P_1470 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1484 = tile.add_modq %P_1468, %P_1471, %P_1474 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1485 = tile.add_modq %P_1469, %P_1472, %P_1475, %P_1478 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1486 = tile.add_modq %P_1473, %P_1476, %P_1479 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1487 = tile.add_modq %P_1477, %P_1480 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1488 = tile.add_modq %P_1481 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s7 = tile.recombine_modq %C_1482, %C_1483, %C_1484, %C_1485, %C_1486, %C_1487, %C_1488 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s8 = tile.mul_modq %D0_3_s7, %Tf_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3_s9 = tile.transpose %D0_3_s8 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D0_3 = tile.reshape %D0_3_s9 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %D1_3_s1 = rns.mul %d1_3, %psi_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %D1_3_s2 = tile.reshape %D1_3_s1 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %a_1489, %a_1490, %a_1491, %a_1492 = tile.decompose %Wf_268434721_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1493, %b_1494, %b_1495, %b_1496 = tile.decompose %D1_3_s2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1497 = tile.matmul_modq %a_1489, %b_1493 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1498 = tile.matmul_modq %a_1489, %b_1494 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1499 = tile.matmul_modq %a_1489, %b_1495 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1500 = tile.matmul_modq %a_1489, %b_1496 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1501 = tile.matmul_modq %a_1490, %b_1493 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1502 = tile.matmul_modq %a_1490, %b_1494 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1503 = tile.matmul_modq %a_1490, %b_1495 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1504 = tile.matmul_modq %a_1490, %b_1496 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1505 = tile.matmul_modq %a_1491, %b_1493 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1506 = tile.matmul_modq %a_1491, %b_1494 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1507 = tile.matmul_modq %a_1491, %b_1495 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1508 = tile.matmul_modq %a_1491, %b_1496 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1509 = tile.matmul_modq %a_1492, %b_1493 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1510 = tile.matmul_modq %a_1492, %b_1494 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1511 = tile.matmul_modq %a_1492, %b_1495 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1512 = tile.matmul_modq %a_1492, %b_1496 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1513 = tile.add_modq %P_1497 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1514 = tile.add_modq %P_1498, %P_1501 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1515 = tile.add_modq %P_1499, %P_1502, %P_1505 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1516 = tile.add_modq %P_1500, %P_1503, %P_1506, %P_1509 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1517 = tile.add_modq %P_1504, %P_1507, %P_1510 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1518 = tile.add_modq %P_1508, %P_1511 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1519 = tile.add_modq %P_1512 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s3 = tile.recombine_modq %C_1513, %C_1514, %C_1515, %C_1516, %C_1517, %C_1518, %C_1519 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s4 = tile.mul_modq %D1_3_s3, %Tf_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s5 = tile.transpose %D1_3_s4 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s6 = tile.reshape %D1_3_s5 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1520, %a_1521, %a_1522, %a_1523 = tile.decompose %Wf_268434721_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1524, %b_1525, %b_1526, %b_1527 = tile.decompose %D1_3_s6 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1528 = tile.matmul_modq %a_1520, %b_1524 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1529 = tile.matmul_modq %a_1520, %b_1525 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1530 = tile.matmul_modq %a_1520, %b_1526 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1531 = tile.matmul_modq %a_1520, %b_1527 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1532 = tile.matmul_modq %a_1521, %b_1524 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1533 = tile.matmul_modq %a_1521, %b_1525 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1534 = tile.matmul_modq %a_1521, %b_1526 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1535 = tile.matmul_modq %a_1521, %b_1527 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1536 = tile.matmul_modq %a_1522, %b_1524 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1537 = tile.matmul_modq %a_1522, %b_1525 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1538 = tile.matmul_modq %a_1522, %b_1526 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1539 = tile.matmul_modq %a_1522, %b_1527 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1540 = tile.matmul_modq %a_1523, %b_1524 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1541 = tile.matmul_modq %a_1523, %b_1525 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1542 = tile.matmul_modq %a_1523, %b_1526 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1543 = tile.matmul_modq %a_1523, %b_1527 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1544 = tile.add_modq %P_1528 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1545 = tile.add_modq %P_1529, %P_1532 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1546 = tile.add_modq %P_1530, %P_1533, %P_1536 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1547 = tile.add_modq %P_1531, %P_1534, %P_1537, %P_1540 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1548 = tile.add_modq %P_1535, %P_1538, %P_1541 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1549 = tile.add_modq %P_1539, %P_1542 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1550 = tile.add_modq %P_1543 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %D1_3_s7 = tile.recombine_modq %C_1544, %C_1545, %C_1546, %C_1547, %C_1548, %C_1549, %C_1550 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
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
    %a_1551, %a_1552, %a_1553, %a_1554 = tile.decompose %Wi_268434721_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1555, %b_1556, %b_1557, %b_1558 = tile.decompose %e0_3_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1559 = tile.matmul_modq %a_1551, %b_1555 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1560 = tile.matmul_modq %a_1551, %b_1556 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1561 = tile.matmul_modq %a_1551, %b_1557 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1562 = tile.matmul_modq %a_1551, %b_1558 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1563 = tile.matmul_modq %a_1552, %b_1555 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1564 = tile.matmul_modq %a_1552, %b_1556 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1565 = tile.matmul_modq %a_1552, %b_1557 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1566 = tile.matmul_modq %a_1552, %b_1558 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1567 = tile.matmul_modq %a_1553, %b_1555 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1568 = tile.matmul_modq %a_1553, %b_1556 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1569 = tile.matmul_modq %a_1553, %b_1557 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1570 = tile.matmul_modq %a_1553, %b_1558 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1571 = tile.matmul_modq %a_1554, %b_1555 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1572 = tile.matmul_modq %a_1554, %b_1556 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1573 = tile.matmul_modq %a_1554, %b_1557 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1574 = tile.matmul_modq %a_1554, %b_1558 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1575 = tile.add_modq %P_1559 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1576 = tile.add_modq %P_1560, %P_1563 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1577 = tile.add_modq %P_1561, %P_1564, %P_1567 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1578 = tile.add_modq %P_1562, %P_1565, %P_1568, %P_1571 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1579 = tile.add_modq %P_1566, %P_1569, %P_1572 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1580 = tile.add_modq %P_1570, %P_1573 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1581 = tile.add_modq %P_1574 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_3_s4 = tile.recombine_modq %C_1575, %C_1576, %C_1577, %C_1578, %C_1579, %C_1580, %C_1581 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_3_s5 = tile.reshape %e0_3_s4 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_3_s6 = tile.transpose %e0_3_s5 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %Ti_268434721_16_0 = tile.const {name = "Ti_268434721_16_0", q = 268434721 : i64} : tensor<4x4xi32>
    %e0_3_s7 = tile.mul_modq %e0_3_s6, %Ti_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %Wi_268434721_16_0 = tile.const {name = "Wi_268434721_16_0", q = 268434721 : i64} : tensor<4x4xi32>
    %a_1582, %a_1583, %a_1584, %a_1585 = tile.decompose %Wi_268434721_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1586, %b_1587, %b_1588, %b_1589 = tile.decompose %e0_3_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1590 = tile.matmul_modq %a_1582, %b_1586 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1591 = tile.matmul_modq %a_1582, %b_1587 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1592 = tile.matmul_modq %a_1582, %b_1588 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1593 = tile.matmul_modq %a_1582, %b_1589 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1594 = tile.matmul_modq %a_1583, %b_1586 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1595 = tile.matmul_modq %a_1583, %b_1587 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1596 = tile.matmul_modq %a_1583, %b_1588 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1597 = tile.matmul_modq %a_1583, %b_1589 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1598 = tile.matmul_modq %a_1584, %b_1586 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1599 = tile.matmul_modq %a_1584, %b_1587 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1600 = tile.matmul_modq %a_1584, %b_1588 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1601 = tile.matmul_modq %a_1584, %b_1589 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1602 = tile.matmul_modq %a_1585, %b_1586 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1603 = tile.matmul_modq %a_1585, %b_1587 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1604 = tile.matmul_modq %a_1585, %b_1588 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1605 = tile.matmul_modq %a_1585, %b_1589 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1606 = tile.add_modq %P_1590 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1607 = tile.add_modq %P_1591, %P_1594 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1608 = tile.add_modq %P_1592, %P_1595, %P_1598 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1609 = tile.add_modq %P_1593, %P_1596, %P_1599, %P_1602 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1610 = tile.add_modq %P_1597, %P_1600, %P_1603 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1611 = tile.add_modq %P_1601, %P_1604 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1612 = tile.add_modq %P_1605 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_3_s8 = tile.recombine_modq %C_1606, %C_1607, %C_1608, %C_1609, %C_1610, %C_1611, %C_1612 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e0_3_s9 = tile.reshape %e0_3_s8 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %psinv_268434721_16 = tile.const {name = "psinv_268434721_16", q = 268434721 : i64} : tensor<16xi32>
    %e0_3 = rns.mul %e0_3_s9, %psinv_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e1_3_s1 = tile.reshape %E1_3 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e1_3_s2 = tile.transpose %e1_3_s1 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s3 = tile.mul_modq %e1_3_s2, %Ti_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1613, %a_1614, %a_1615, %a_1616 = tile.decompose %Wi_268434721_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1617, %b_1618, %b_1619, %b_1620 = tile.decompose %e1_3_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1621 = tile.matmul_modq %a_1613, %b_1617 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1622 = tile.matmul_modq %a_1613, %b_1618 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1623 = tile.matmul_modq %a_1613, %b_1619 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1624 = tile.matmul_modq %a_1613, %b_1620 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1625 = tile.matmul_modq %a_1614, %b_1617 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1626 = tile.matmul_modq %a_1614, %b_1618 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1627 = tile.matmul_modq %a_1614, %b_1619 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1628 = tile.matmul_modq %a_1614, %b_1620 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1629 = tile.matmul_modq %a_1615, %b_1617 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1630 = tile.matmul_modq %a_1615, %b_1618 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1631 = tile.matmul_modq %a_1615, %b_1619 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1632 = tile.matmul_modq %a_1615, %b_1620 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1633 = tile.matmul_modq %a_1616, %b_1617 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1634 = tile.matmul_modq %a_1616, %b_1618 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1635 = tile.matmul_modq %a_1616, %b_1619 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1636 = tile.matmul_modq %a_1616, %b_1620 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1637 = tile.add_modq %P_1621 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1638 = tile.add_modq %P_1622, %P_1625 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1639 = tile.add_modq %P_1623, %P_1626, %P_1629 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1640 = tile.add_modq %P_1624, %P_1627, %P_1630, %P_1633 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1641 = tile.add_modq %P_1628, %P_1631, %P_1634 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1642 = tile.add_modq %P_1632, %P_1635 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1643 = tile.add_modq %P_1636 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s4 = tile.recombine_modq %C_1637, %C_1638, %C_1639, %C_1640, %C_1641, %C_1642, %C_1643 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s5 = tile.reshape %e1_3_s4 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s6 = tile.transpose %e1_3_s5 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s7 = tile.mul_modq %e1_3_s6, %Ti_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1644, %a_1645, %a_1646, %a_1647 = tile.decompose %Wi_268434721_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1648, %b_1649, %b_1650, %b_1651 = tile.decompose %e1_3_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1652 = tile.matmul_modq %a_1644, %b_1648 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1653 = tile.matmul_modq %a_1644, %b_1649 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1654 = tile.matmul_modq %a_1644, %b_1650 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1655 = tile.matmul_modq %a_1644, %b_1651 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1656 = tile.matmul_modq %a_1645, %b_1648 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1657 = tile.matmul_modq %a_1645, %b_1649 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1658 = tile.matmul_modq %a_1645, %b_1650 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1659 = tile.matmul_modq %a_1645, %b_1651 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1660 = tile.matmul_modq %a_1646, %b_1648 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1661 = tile.matmul_modq %a_1646, %b_1649 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1662 = tile.matmul_modq %a_1646, %b_1650 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1663 = tile.matmul_modq %a_1646, %b_1651 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1664 = tile.matmul_modq %a_1647, %b_1648 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1665 = tile.matmul_modq %a_1647, %b_1649 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1666 = tile.matmul_modq %a_1647, %b_1650 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1667 = tile.matmul_modq %a_1647, %b_1651 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1668 = tile.add_modq %P_1652 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1669 = tile.add_modq %P_1653, %P_1656 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1670 = tile.add_modq %P_1654, %P_1657, %P_1660 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1671 = tile.add_modq %P_1655, %P_1658, %P_1661, %P_1664 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1672 = tile.add_modq %P_1659, %P_1662, %P_1665 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1673 = tile.add_modq %P_1663, %P_1666 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1674 = tile.add_modq %P_1667 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s8 = tile.recombine_modq %C_1668, %C_1669, %C_1670, %C_1671, %C_1672, %C_1673, %C_1674 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1_3_s9 = tile.reshape %e1_3_s8 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e1_3 = rns.mul %e1_3_s9, %psinv_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e2_3_s1 = tile.reshape %E2_3 {shape = [4, 4], q = 268434721 : i64} : (tensor<16xi32>) -> tensor<4x4xi32>
    %e2_3_s2 = tile.transpose %e2_3_s1 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s3 = tile.mul_modq %e2_3_s2, %Ti_268434721_16_1 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1675, %a_1676, %a_1677, %a_1678 = tile.decompose %Wi_268434721_16_1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1679, %b_1680, %b_1681, %b_1682 = tile.decompose %e2_3_s3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1683 = tile.matmul_modq %a_1675, %b_1679 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1684 = tile.matmul_modq %a_1675, %b_1680 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1685 = tile.matmul_modq %a_1675, %b_1681 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1686 = tile.matmul_modq %a_1675, %b_1682 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1687 = tile.matmul_modq %a_1676, %b_1679 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1688 = tile.matmul_modq %a_1676, %b_1680 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1689 = tile.matmul_modq %a_1676, %b_1681 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1690 = tile.matmul_modq %a_1676, %b_1682 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1691 = tile.matmul_modq %a_1677, %b_1679 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1692 = tile.matmul_modq %a_1677, %b_1680 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1693 = tile.matmul_modq %a_1677, %b_1681 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1694 = tile.matmul_modq %a_1677, %b_1682 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1695 = tile.matmul_modq %a_1678, %b_1679 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1696 = tile.matmul_modq %a_1678, %b_1680 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1697 = tile.matmul_modq %a_1678, %b_1681 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1698 = tile.matmul_modq %a_1678, %b_1682 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1699 = tile.add_modq %P_1683 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1700 = tile.add_modq %P_1684, %P_1687 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1701 = tile.add_modq %P_1685, %P_1688, %P_1691 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1702 = tile.add_modq %P_1686, %P_1689, %P_1692, %P_1695 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1703 = tile.add_modq %P_1690, %P_1693, %P_1696 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1704 = tile.add_modq %P_1694, %P_1697 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1705 = tile.add_modq %P_1698 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s4 = tile.recombine_modq %C_1699, %C_1700, %C_1701, %C_1702, %C_1703, %C_1704, %C_1705 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s5 = tile.reshape %e2_3_s4 {shape = [4, 4], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s6 = tile.transpose %e2_3_s5 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s7 = tile.mul_modq %e2_3_s6, %Ti_268434721_16_0 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %a_1706, %a_1707, %a_1708, %a_1709 = tile.decompose %Wi_268434721_16_0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %b_1710, %b_1711, %b_1712, %b_1713 = tile.decompose %e2_3_s7 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x4xi32>) -> (tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>, tensor<4x4xi8>)
    %P_1714 = tile.matmul_modq %a_1706, %b_1710 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1715 = tile.matmul_modq %a_1706, %b_1711 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1716 = tile.matmul_modq %a_1706, %b_1712 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1717 = tile.matmul_modq %a_1706, %b_1713 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1718 = tile.matmul_modq %a_1707, %b_1710 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1719 = tile.matmul_modq %a_1707, %b_1711 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1720 = tile.matmul_modq %a_1707, %b_1712 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1721 = tile.matmul_modq %a_1707, %b_1713 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1722 = tile.matmul_modq %a_1708, %b_1710 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1723 = tile.matmul_modq %a_1708, %b_1711 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1724 = tile.matmul_modq %a_1708, %b_1712 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1725 = tile.matmul_modq %a_1708, %b_1713 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1726 = tile.matmul_modq %a_1709, %b_1710 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1727 = tile.matmul_modq %a_1709, %b_1711 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1728 = tile.matmul_modq %a_1709, %b_1712 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %P_1729 = tile.matmul_modq %a_1709, %b_1713 {q = 268434721 : i64} : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
    %C_1730 = tile.add_modq %P_1714 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1731 = tile.add_modq %P_1715, %P_1718 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1732 = tile.add_modq %P_1716, %P_1719, %P_1722 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1733 = tile.add_modq %P_1717, %P_1720, %P_1723, %P_1726 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1734 = tile.add_modq %P_1721, %P_1724, %P_1727 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1735 = tile.add_modq %P_1725, %P_1728 {q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %C_1736 = tile.add_modq %P_1729 {q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s8 = tile.recombine_modq %C_1730, %C_1731, %C_1732, %C_1733, %C_1734, %C_1735, %C_1736 {radix = 128 : i64, q = 268434721 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
    %e2_3_s9 = tile.reshape %e2_3_s8 {shape = [16], q = 268434721 : i64} : (tensor<4x4xi32>) -> tensor<16xi32>
    %e2_3 = rns.mul %e2_3_s9, %psinv_268434721_16 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    return %e0_0, %e0_1, %e0_2, %e0_3, %e1_0, %e1_1, %e1_2, %e1_3, %e2_0, %e2_1, %e2_2, %e2_3 : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }
}
