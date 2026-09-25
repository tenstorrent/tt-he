module {
  func.func @ct_mul_l3(%c00: tensor<16xi32>, %c01: tensor<16xi32>, %c02: tensor<16xi32>, %c03: tensor<16xi32>, %c04: tensor<16xi32>, %c05: tensor<16xi32>, %c06: tensor<16xi32>, %c10: tensor<16xi32>, %c11: tensor<16xi32>, %c12: tensor<16xi32>, %c13: tensor<16xi32>, %c14: tensor<16xi32>, %c15: tensor<16xi32>, %c16: tensor<16xi32>, %d00: tensor<16xi32>, %d01: tensor<16xi32>, %d02: tensor<16xi32>, %d03: tensor<16xi32>, %d04: tensor<16xi32>, %d05: tensor<16xi32>, %d06: tensor<16xi32>, %d10: tensor<16xi32>, %d11: tensor<16xi32>, %d12: tensor<16xi32>, %d13: tensor<16xi32>, %d14: tensor<16xi32>, %d15: tensor<16xi32>, %d16: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %W_f_268435361 = tile.const {name = "W_f_268435361", q = 268435361 : i64} : tensor<16x16xi32>
    %a_1, %a_2, %a_3, %a_4 = tile.decompose %W_f_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_5, %b_6, %b_7, %b_8 = tile.decompose %c00 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_9 = tile.matmul_modq %a_1, %b_5 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_10 = tile.matmul_modq %a_1, %b_6 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_11 = tile.matmul_modq %a_1, %b_7 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_12 = tile.matmul_modq %a_1, %b_8 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_13 = tile.matmul_modq %a_2, %b_5 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_14 = tile.matmul_modq %a_2, %b_6 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_15 = tile.matmul_modq %a_2, %b_7 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_16 = tile.matmul_modq %a_2, %b_8 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_17 = tile.matmul_modq %a_3, %b_5 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_18 = tile.matmul_modq %a_3, %b_6 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_19 = tile.matmul_modq %a_3, %b_7 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_20 = tile.matmul_modq %a_3, %b_8 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_21 = tile.matmul_modq %a_4, %b_5 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_22 = tile.matmul_modq %a_4, %b_6 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_23 = tile.matmul_modq %a_4, %b_7 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_24 = tile.matmul_modq %a_4, %b_8 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_25 = tile.add_modq %P_9 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_26 = tile.add_modq %P_10, %P_13 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_27 = tile.add_modq %P_11, %P_14, %P_17 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_28 = tile.add_modq %P_12, %P_15, %P_18, %P_21 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_29 = tile.add_modq %P_16, %P_19, %P_22 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_30 = tile.add_modq %P_20, %P_23 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_31 = tile.add_modq %P_24 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC00 = tile.recombine_modq %C_25, %C_26, %C_27, %C_28, %C_29, %C_30, %C_31 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_32, %a_33, %a_34, %a_35 = tile.decompose %W_f_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_36, %b_37, %b_38, %b_39 = tile.decompose %c10 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_40 = tile.matmul_modq %a_32, %b_36 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_41 = tile.matmul_modq %a_32, %b_37 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_42 = tile.matmul_modq %a_32, %b_38 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_43 = tile.matmul_modq %a_32, %b_39 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_44 = tile.matmul_modq %a_33, %b_36 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_45 = tile.matmul_modq %a_33, %b_37 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_46 = tile.matmul_modq %a_33, %b_38 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_47 = tile.matmul_modq %a_33, %b_39 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_48 = tile.matmul_modq %a_34, %b_36 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_49 = tile.matmul_modq %a_34, %b_37 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_50 = tile.matmul_modq %a_34, %b_38 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_51 = tile.matmul_modq %a_34, %b_39 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_52 = tile.matmul_modq %a_35, %b_36 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_53 = tile.matmul_modq %a_35, %b_37 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_54 = tile.matmul_modq %a_35, %b_38 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_55 = tile.matmul_modq %a_35, %b_39 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_56 = tile.add_modq %P_40 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_57 = tile.add_modq %P_41, %P_44 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_58 = tile.add_modq %P_42, %P_45, %P_48 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_59 = tile.add_modq %P_43, %P_46, %P_49, %P_52 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_60 = tile.add_modq %P_47, %P_50, %P_53 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_61 = tile.add_modq %P_51, %P_54 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_62 = tile.add_modq %P_55 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC10 = tile.recombine_modq %C_56, %C_57, %C_58, %C_59, %C_60, %C_61, %C_62 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_63, %a_64, %a_65, %a_66 = tile.decompose %W_f_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_67, %b_68, %b_69, %b_70 = tile.decompose %d00 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_71 = tile.matmul_modq %a_63, %b_67 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_72 = tile.matmul_modq %a_63, %b_68 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_73 = tile.matmul_modq %a_63, %b_69 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_74 = tile.matmul_modq %a_63, %b_70 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_75 = tile.matmul_modq %a_64, %b_67 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_76 = tile.matmul_modq %a_64, %b_68 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_77 = tile.matmul_modq %a_64, %b_69 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_78 = tile.matmul_modq %a_64, %b_70 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_79 = tile.matmul_modq %a_65, %b_67 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_80 = tile.matmul_modq %a_65, %b_68 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_81 = tile.matmul_modq %a_65, %b_69 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_82 = tile.matmul_modq %a_65, %b_70 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_83 = tile.matmul_modq %a_66, %b_67 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_84 = tile.matmul_modq %a_66, %b_68 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_85 = tile.matmul_modq %a_66, %b_69 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_86 = tile.matmul_modq %a_66, %b_70 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_87 = tile.add_modq %P_71 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_88 = tile.add_modq %P_72, %P_75 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_89 = tile.add_modq %P_73, %P_76, %P_79 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_90 = tile.add_modq %P_74, %P_77, %P_80, %P_83 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_91 = tile.add_modq %P_78, %P_81, %P_84 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_92 = tile.add_modq %P_82, %P_85 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_93 = tile.add_modq %P_86 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD00 = tile.recombine_modq %C_87, %C_88, %C_89, %C_90, %C_91, %C_92, %C_93 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_94, %a_95, %a_96, %a_97 = tile.decompose %W_f_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_98, %b_99, %b_100, %b_101 = tile.decompose %d10 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_102 = tile.matmul_modq %a_94, %b_98 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_103 = tile.matmul_modq %a_94, %b_99 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_104 = tile.matmul_modq %a_94, %b_100 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_105 = tile.matmul_modq %a_94, %b_101 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_106 = tile.matmul_modq %a_95, %b_98 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_107 = tile.matmul_modq %a_95, %b_99 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_108 = tile.matmul_modq %a_95, %b_100 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_109 = tile.matmul_modq %a_95, %b_101 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_110 = tile.matmul_modq %a_96, %b_98 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_111 = tile.matmul_modq %a_96, %b_99 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_112 = tile.matmul_modq %a_96, %b_100 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_113 = tile.matmul_modq %a_96, %b_101 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_114 = tile.matmul_modq %a_97, %b_98 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_115 = tile.matmul_modq %a_97, %b_99 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_116 = tile.matmul_modq %a_97, %b_100 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_117 = tile.matmul_modq %a_97, %b_101 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_118 = tile.add_modq %P_102 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_119 = tile.add_modq %P_103, %P_106 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_120 = tile.add_modq %P_104, %P_107, %P_110 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_121 = tile.add_modq %P_105, %P_108, %P_111, %P_114 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_122 = tile.add_modq %P_109, %P_112, %P_115 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_123 = tile.add_modq %P_113, %P_116 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_124 = tile.add_modq %P_117 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD10 = tile.recombine_modq %C_118, %C_119, %C_120, %C_121, %C_122, %C_123, %C_124 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE00 = tile.mul_modq %tC00, %tD00 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE20 = tile.mul_modq %tC10, %tD10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT00 = tile.mul_modq %tC00, %tD10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT10 = tile.mul_modq %tC10, %tD00 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE10 = tile.add_modq %tT00, %tT10 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268435361 = tile.const {name = "W_i_268435361", q = 268435361 : i64} : tensor<16x16xi32>
    %a_125, %a_126, %a_127, %a_128 = tile.decompose %W_i_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_129, %b_130, %b_131, %b_132 = tile.decompose %tE00 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_133 = tile.matmul_modq %a_125, %b_129 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_134 = tile.matmul_modq %a_125, %b_130 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_135 = tile.matmul_modq %a_125, %b_131 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_136 = tile.matmul_modq %a_125, %b_132 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_137 = tile.matmul_modq %a_126, %b_129 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_138 = tile.matmul_modq %a_126, %b_130 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_139 = tile.matmul_modq %a_126, %b_131 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_140 = tile.matmul_modq %a_126, %b_132 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_141 = tile.matmul_modq %a_127, %b_129 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_142 = tile.matmul_modq %a_127, %b_130 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_143 = tile.matmul_modq %a_127, %b_131 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_144 = tile.matmul_modq %a_127, %b_132 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_145 = tile.matmul_modq %a_128, %b_129 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_146 = tile.matmul_modq %a_128, %b_130 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_147 = tile.matmul_modq %a_128, %b_131 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_148 = tile.matmul_modq %a_128, %b_132 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_149 = tile.add_modq %P_133 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_150 = tile.add_modq %P_134, %P_137 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_151 = tile.add_modq %P_135, %P_138, %P_141 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_152 = tile.add_modq %P_136, %P_139, %P_142, %P_145 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_153 = tile.add_modq %P_140, %P_143, %P_146 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_154 = tile.add_modq %P_144, %P_147 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_155 = tile.add_modq %P_148 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e0_0 = tile.recombine_modq %C_149, %C_150, %C_151, %C_152, %C_153, %C_154, %C_155 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_156, %a_157, %a_158, %a_159 = tile.decompose %W_i_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_160, %b_161, %b_162, %b_163 = tile.decompose %tE10 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_164 = tile.matmul_modq %a_156, %b_160 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_165 = tile.matmul_modq %a_156, %b_161 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_166 = tile.matmul_modq %a_156, %b_162 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_167 = tile.matmul_modq %a_156, %b_163 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_168 = tile.matmul_modq %a_157, %b_160 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_169 = tile.matmul_modq %a_157, %b_161 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_170 = tile.matmul_modq %a_157, %b_162 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_171 = tile.matmul_modq %a_157, %b_163 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_172 = tile.matmul_modq %a_158, %b_160 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_173 = tile.matmul_modq %a_158, %b_161 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_174 = tile.matmul_modq %a_158, %b_162 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_175 = tile.matmul_modq %a_158, %b_163 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_176 = tile.matmul_modq %a_159, %b_160 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_177 = tile.matmul_modq %a_159, %b_161 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_178 = tile.matmul_modq %a_159, %b_162 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_179 = tile.matmul_modq %a_159, %b_163 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_180 = tile.add_modq %P_164 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_181 = tile.add_modq %P_165, %P_168 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_182 = tile.add_modq %P_166, %P_169, %P_172 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_183 = tile.add_modq %P_167, %P_170, %P_173, %P_176 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_184 = tile.add_modq %P_171, %P_174, %P_177 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_185 = tile.add_modq %P_175, %P_178 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_186 = tile.add_modq %P_179 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_0 = tile.recombine_modq %C_180, %C_181, %C_182, %C_183, %C_184, %C_185, %C_186 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_187, %a_188, %a_189, %a_190 = tile.decompose %W_i_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_191, %b_192, %b_193, %b_194 = tile.decompose %tE20 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_195 = tile.matmul_modq %a_187, %b_191 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_196 = tile.matmul_modq %a_187, %b_192 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_197 = tile.matmul_modq %a_187, %b_193 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_198 = tile.matmul_modq %a_187, %b_194 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_199 = tile.matmul_modq %a_188, %b_191 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_200 = tile.matmul_modq %a_188, %b_192 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_201 = tile.matmul_modq %a_188, %b_193 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_202 = tile.matmul_modq %a_188, %b_194 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_203 = tile.matmul_modq %a_189, %b_191 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_204 = tile.matmul_modq %a_189, %b_192 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_205 = tile.matmul_modq %a_189, %b_193 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_206 = tile.matmul_modq %a_189, %b_194 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_207 = tile.matmul_modq %a_190, %b_191 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_208 = tile.matmul_modq %a_190, %b_192 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_209 = tile.matmul_modq %a_190, %b_193 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_210 = tile.matmul_modq %a_190, %b_194 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_211 = tile.add_modq %P_195 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_212 = tile.add_modq %P_196, %P_199 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_213 = tile.add_modq %P_197, %P_200, %P_203 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_214 = tile.add_modq %P_198, %P_201, %P_204, %P_207 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_215 = tile.add_modq %P_202, %P_205, %P_208 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_216 = tile.add_modq %P_206, %P_209 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_217 = tile.add_modq %P_210 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_0 = tile.recombine_modq %C_211, %C_212, %C_213, %C_214, %C_215, %C_216, %C_217 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268435009 = tile.const {name = "W_f_268435009", q = 268435009 : i64} : tensor<16x16xi32>
    %a_218, %a_219, %a_220, %a_221 = tile.decompose %W_f_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_222, %b_223, %b_224, %b_225 = tile.decompose %c01 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_226 = tile.matmul_modq %a_218, %b_222 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_227 = tile.matmul_modq %a_218, %b_223 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_228 = tile.matmul_modq %a_218, %b_224 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_229 = tile.matmul_modq %a_218, %b_225 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_230 = tile.matmul_modq %a_219, %b_222 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_231 = tile.matmul_modq %a_219, %b_223 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_232 = tile.matmul_modq %a_219, %b_224 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_233 = tile.matmul_modq %a_219, %b_225 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_234 = tile.matmul_modq %a_220, %b_222 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_235 = tile.matmul_modq %a_220, %b_223 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_236 = tile.matmul_modq %a_220, %b_224 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_237 = tile.matmul_modq %a_220, %b_225 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_238 = tile.matmul_modq %a_221, %b_222 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_239 = tile.matmul_modq %a_221, %b_223 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_240 = tile.matmul_modq %a_221, %b_224 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_241 = tile.matmul_modq %a_221, %b_225 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_242 = tile.add_modq %P_226 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_243 = tile.add_modq %P_227, %P_230 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_244 = tile.add_modq %P_228, %P_231, %P_234 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_245 = tile.add_modq %P_229, %P_232, %P_235, %P_238 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_246 = tile.add_modq %P_233, %P_236, %P_239 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_247 = tile.add_modq %P_237, %P_240 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_248 = tile.add_modq %P_241 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC01 = tile.recombine_modq %C_242, %C_243, %C_244, %C_245, %C_246, %C_247, %C_248 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_249, %a_250, %a_251, %a_252 = tile.decompose %W_f_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_253, %b_254, %b_255, %b_256 = tile.decompose %c11 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_257 = tile.matmul_modq %a_249, %b_253 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_258 = tile.matmul_modq %a_249, %b_254 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_259 = tile.matmul_modq %a_249, %b_255 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_260 = tile.matmul_modq %a_249, %b_256 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_261 = tile.matmul_modq %a_250, %b_253 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_262 = tile.matmul_modq %a_250, %b_254 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_263 = tile.matmul_modq %a_250, %b_255 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_264 = tile.matmul_modq %a_250, %b_256 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_265 = tile.matmul_modq %a_251, %b_253 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_266 = tile.matmul_modq %a_251, %b_254 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_267 = tile.matmul_modq %a_251, %b_255 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_268 = tile.matmul_modq %a_251, %b_256 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_269 = tile.matmul_modq %a_252, %b_253 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_270 = tile.matmul_modq %a_252, %b_254 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_271 = tile.matmul_modq %a_252, %b_255 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_272 = tile.matmul_modq %a_252, %b_256 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_273 = tile.add_modq %P_257 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_274 = tile.add_modq %P_258, %P_261 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_275 = tile.add_modq %P_259, %P_262, %P_265 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_276 = tile.add_modq %P_260, %P_263, %P_266, %P_269 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_277 = tile.add_modq %P_264, %P_267, %P_270 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_278 = tile.add_modq %P_268, %P_271 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_279 = tile.add_modq %P_272 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC11 = tile.recombine_modq %C_273, %C_274, %C_275, %C_276, %C_277, %C_278, %C_279 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_280, %a_281, %a_282, %a_283 = tile.decompose %W_f_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_284, %b_285, %b_286, %b_287 = tile.decompose %d01 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_288 = tile.matmul_modq %a_280, %b_284 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_289 = tile.matmul_modq %a_280, %b_285 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_290 = tile.matmul_modq %a_280, %b_286 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_291 = tile.matmul_modq %a_280, %b_287 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_292 = tile.matmul_modq %a_281, %b_284 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_293 = tile.matmul_modq %a_281, %b_285 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_294 = tile.matmul_modq %a_281, %b_286 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_295 = tile.matmul_modq %a_281, %b_287 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_296 = tile.matmul_modq %a_282, %b_284 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_297 = tile.matmul_modq %a_282, %b_285 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_298 = tile.matmul_modq %a_282, %b_286 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_299 = tile.matmul_modq %a_282, %b_287 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_300 = tile.matmul_modq %a_283, %b_284 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_301 = tile.matmul_modq %a_283, %b_285 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_302 = tile.matmul_modq %a_283, %b_286 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_303 = tile.matmul_modq %a_283, %b_287 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_304 = tile.add_modq %P_288 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_305 = tile.add_modq %P_289, %P_292 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_306 = tile.add_modq %P_290, %P_293, %P_296 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_307 = tile.add_modq %P_291, %P_294, %P_297, %P_300 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_308 = tile.add_modq %P_295, %P_298, %P_301 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_309 = tile.add_modq %P_299, %P_302 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_310 = tile.add_modq %P_303 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD01 = tile.recombine_modq %C_304, %C_305, %C_306, %C_307, %C_308, %C_309, %C_310 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_311, %a_312, %a_313, %a_314 = tile.decompose %W_f_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_315, %b_316, %b_317, %b_318 = tile.decompose %d11 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_319 = tile.matmul_modq %a_311, %b_315 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_320 = tile.matmul_modq %a_311, %b_316 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_321 = tile.matmul_modq %a_311, %b_317 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_322 = tile.matmul_modq %a_311, %b_318 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_323 = tile.matmul_modq %a_312, %b_315 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_324 = tile.matmul_modq %a_312, %b_316 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_325 = tile.matmul_modq %a_312, %b_317 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_326 = tile.matmul_modq %a_312, %b_318 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_327 = tile.matmul_modq %a_313, %b_315 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_328 = tile.matmul_modq %a_313, %b_316 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_329 = tile.matmul_modq %a_313, %b_317 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_330 = tile.matmul_modq %a_313, %b_318 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_331 = tile.matmul_modq %a_314, %b_315 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_332 = tile.matmul_modq %a_314, %b_316 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_333 = tile.matmul_modq %a_314, %b_317 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_334 = tile.matmul_modq %a_314, %b_318 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_335 = tile.add_modq %P_319 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_336 = tile.add_modq %P_320, %P_323 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_337 = tile.add_modq %P_321, %P_324, %P_327 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_338 = tile.add_modq %P_322, %P_325, %P_328, %P_331 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_339 = tile.add_modq %P_326, %P_329, %P_332 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_340 = tile.add_modq %P_330, %P_333 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_341 = tile.add_modq %P_334 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD11 = tile.recombine_modq %C_335, %C_336, %C_337, %C_338, %C_339, %C_340, %C_341 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE01 = tile.mul_modq %tC01, %tD01 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE21 = tile.mul_modq %tC11, %tD11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT01 = tile.mul_modq %tC01, %tD11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT11 = tile.mul_modq %tC11, %tD01 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE11 = tile.add_modq %tT01, %tT11 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268435009 = tile.const {name = "W_i_268435009", q = 268435009 : i64} : tensor<16x16xi32>
    %a_342, %a_343, %a_344, %a_345 = tile.decompose %W_i_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_346, %b_347, %b_348, %b_349 = tile.decompose %tE01 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_350 = tile.matmul_modq %a_342, %b_346 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_351 = tile.matmul_modq %a_342, %b_347 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_352 = tile.matmul_modq %a_342, %b_348 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_353 = tile.matmul_modq %a_342, %b_349 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_354 = tile.matmul_modq %a_343, %b_346 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_355 = tile.matmul_modq %a_343, %b_347 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_356 = tile.matmul_modq %a_343, %b_348 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_357 = tile.matmul_modq %a_343, %b_349 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_358 = tile.matmul_modq %a_344, %b_346 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_359 = tile.matmul_modq %a_344, %b_347 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_360 = tile.matmul_modq %a_344, %b_348 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_361 = tile.matmul_modq %a_344, %b_349 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_362 = tile.matmul_modq %a_345, %b_346 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_363 = tile.matmul_modq %a_345, %b_347 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_364 = tile.matmul_modq %a_345, %b_348 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_365 = tile.matmul_modq %a_345, %b_349 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_366 = tile.add_modq %P_350 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_367 = tile.add_modq %P_351, %P_354 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_368 = tile.add_modq %P_352, %P_355, %P_358 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_369 = tile.add_modq %P_353, %P_356, %P_359, %P_362 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_370 = tile.add_modq %P_357, %P_360, %P_363 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_371 = tile.add_modq %P_361, %P_364 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_372 = tile.add_modq %P_365 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e0_1 = tile.recombine_modq %C_366, %C_367, %C_368, %C_369, %C_370, %C_371, %C_372 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_373, %a_374, %a_375, %a_376 = tile.decompose %W_i_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_377, %b_378, %b_379, %b_380 = tile.decompose %tE11 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_381 = tile.matmul_modq %a_373, %b_377 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_382 = tile.matmul_modq %a_373, %b_378 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_383 = tile.matmul_modq %a_373, %b_379 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_384 = tile.matmul_modq %a_373, %b_380 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_385 = tile.matmul_modq %a_374, %b_377 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_386 = tile.matmul_modq %a_374, %b_378 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_387 = tile.matmul_modq %a_374, %b_379 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_388 = tile.matmul_modq %a_374, %b_380 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_389 = tile.matmul_modq %a_375, %b_377 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_390 = tile.matmul_modq %a_375, %b_378 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_391 = tile.matmul_modq %a_375, %b_379 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_392 = tile.matmul_modq %a_375, %b_380 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_393 = tile.matmul_modq %a_376, %b_377 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_394 = tile.matmul_modq %a_376, %b_378 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_395 = tile.matmul_modq %a_376, %b_379 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_396 = tile.matmul_modq %a_376, %b_380 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_397 = tile.add_modq %P_381 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_398 = tile.add_modq %P_382, %P_385 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_399 = tile.add_modq %P_383, %P_386, %P_389 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_400 = tile.add_modq %P_384, %P_387, %P_390, %P_393 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_401 = tile.add_modq %P_388, %P_391, %P_394 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_402 = tile.add_modq %P_392, %P_395 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_403 = tile.add_modq %P_396 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_1 = tile.recombine_modq %C_397, %C_398, %C_399, %C_400, %C_401, %C_402, %C_403 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_404, %a_405, %a_406, %a_407 = tile.decompose %W_i_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_408, %b_409, %b_410, %b_411 = tile.decompose %tE21 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_412 = tile.matmul_modq %a_404, %b_408 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_413 = tile.matmul_modq %a_404, %b_409 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_414 = tile.matmul_modq %a_404, %b_410 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_415 = tile.matmul_modq %a_404, %b_411 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_416 = tile.matmul_modq %a_405, %b_408 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_417 = tile.matmul_modq %a_405, %b_409 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_418 = tile.matmul_modq %a_405, %b_410 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_419 = tile.matmul_modq %a_405, %b_411 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_420 = tile.matmul_modq %a_406, %b_408 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_421 = tile.matmul_modq %a_406, %b_409 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_422 = tile.matmul_modq %a_406, %b_410 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_423 = tile.matmul_modq %a_406, %b_411 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_424 = tile.matmul_modq %a_407, %b_408 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_425 = tile.matmul_modq %a_407, %b_409 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_426 = tile.matmul_modq %a_407, %b_410 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_427 = tile.matmul_modq %a_407, %b_411 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_428 = tile.add_modq %P_412 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_429 = tile.add_modq %P_413, %P_416 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_430 = tile.add_modq %P_414, %P_417, %P_420 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_431 = tile.add_modq %P_415, %P_418, %P_421, %P_424 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_432 = tile.add_modq %P_419, %P_422, %P_425 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_433 = tile.add_modq %P_423, %P_426 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_434 = tile.add_modq %P_427 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_1 = tile.recombine_modq %C_428, %C_429, %C_430, %C_431, %C_432, %C_433, %C_434 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268434977 = tile.const {name = "W_f_268434977", q = 268434977 : i64} : tensor<16x16xi32>
    %a_435, %a_436, %a_437, %a_438 = tile.decompose %W_f_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_439, %b_440, %b_441, %b_442 = tile.decompose %c02 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_443 = tile.matmul_modq %a_435, %b_439 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_444 = tile.matmul_modq %a_435, %b_440 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_445 = tile.matmul_modq %a_435, %b_441 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_446 = tile.matmul_modq %a_435, %b_442 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_447 = tile.matmul_modq %a_436, %b_439 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_448 = tile.matmul_modq %a_436, %b_440 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_449 = tile.matmul_modq %a_436, %b_441 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_450 = tile.matmul_modq %a_436, %b_442 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_451 = tile.matmul_modq %a_437, %b_439 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_452 = tile.matmul_modq %a_437, %b_440 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_453 = tile.matmul_modq %a_437, %b_441 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_454 = tile.matmul_modq %a_437, %b_442 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_455 = tile.matmul_modq %a_438, %b_439 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_456 = tile.matmul_modq %a_438, %b_440 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_457 = tile.matmul_modq %a_438, %b_441 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_458 = tile.matmul_modq %a_438, %b_442 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_459 = tile.add_modq %P_443 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_460 = tile.add_modq %P_444, %P_447 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_461 = tile.add_modq %P_445, %P_448, %P_451 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_462 = tile.add_modq %P_446, %P_449, %P_452, %P_455 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_463 = tile.add_modq %P_450, %P_453, %P_456 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_464 = tile.add_modq %P_454, %P_457 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_465 = tile.add_modq %P_458 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC02 = tile.recombine_modq %C_459, %C_460, %C_461, %C_462, %C_463, %C_464, %C_465 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_466, %a_467, %a_468, %a_469 = tile.decompose %W_f_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_470, %b_471, %b_472, %b_473 = tile.decompose %c12 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_474 = tile.matmul_modq %a_466, %b_470 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_475 = tile.matmul_modq %a_466, %b_471 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_476 = tile.matmul_modq %a_466, %b_472 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_477 = tile.matmul_modq %a_466, %b_473 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_478 = tile.matmul_modq %a_467, %b_470 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_479 = tile.matmul_modq %a_467, %b_471 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_480 = tile.matmul_modq %a_467, %b_472 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_481 = tile.matmul_modq %a_467, %b_473 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_482 = tile.matmul_modq %a_468, %b_470 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_483 = tile.matmul_modq %a_468, %b_471 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_484 = tile.matmul_modq %a_468, %b_472 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_485 = tile.matmul_modq %a_468, %b_473 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_486 = tile.matmul_modq %a_469, %b_470 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_487 = tile.matmul_modq %a_469, %b_471 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_488 = tile.matmul_modq %a_469, %b_472 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_489 = tile.matmul_modq %a_469, %b_473 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_490 = tile.add_modq %P_474 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_491 = tile.add_modq %P_475, %P_478 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_492 = tile.add_modq %P_476, %P_479, %P_482 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_493 = tile.add_modq %P_477, %P_480, %P_483, %P_486 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_494 = tile.add_modq %P_481, %P_484, %P_487 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_495 = tile.add_modq %P_485, %P_488 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_496 = tile.add_modq %P_489 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC12 = tile.recombine_modq %C_490, %C_491, %C_492, %C_493, %C_494, %C_495, %C_496 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_497, %a_498, %a_499, %a_500 = tile.decompose %W_f_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_501, %b_502, %b_503, %b_504 = tile.decompose %d02 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_505 = tile.matmul_modq %a_497, %b_501 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_506 = tile.matmul_modq %a_497, %b_502 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_507 = tile.matmul_modq %a_497, %b_503 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_508 = tile.matmul_modq %a_497, %b_504 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_509 = tile.matmul_modq %a_498, %b_501 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_510 = tile.matmul_modq %a_498, %b_502 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_511 = tile.matmul_modq %a_498, %b_503 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_512 = tile.matmul_modq %a_498, %b_504 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_513 = tile.matmul_modq %a_499, %b_501 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_514 = tile.matmul_modq %a_499, %b_502 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_515 = tile.matmul_modq %a_499, %b_503 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_516 = tile.matmul_modq %a_499, %b_504 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_517 = tile.matmul_modq %a_500, %b_501 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_518 = tile.matmul_modq %a_500, %b_502 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_519 = tile.matmul_modq %a_500, %b_503 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_520 = tile.matmul_modq %a_500, %b_504 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_521 = tile.add_modq %P_505 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_522 = tile.add_modq %P_506, %P_509 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_523 = tile.add_modq %P_507, %P_510, %P_513 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_524 = tile.add_modq %P_508, %P_511, %P_514, %P_517 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_525 = tile.add_modq %P_512, %P_515, %P_518 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_526 = tile.add_modq %P_516, %P_519 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_527 = tile.add_modq %P_520 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD02 = tile.recombine_modq %C_521, %C_522, %C_523, %C_524, %C_525, %C_526, %C_527 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_528, %a_529, %a_530, %a_531 = tile.decompose %W_f_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_532, %b_533, %b_534, %b_535 = tile.decompose %d12 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_536 = tile.matmul_modq %a_528, %b_532 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_537 = tile.matmul_modq %a_528, %b_533 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_538 = tile.matmul_modq %a_528, %b_534 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_539 = tile.matmul_modq %a_528, %b_535 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_540 = tile.matmul_modq %a_529, %b_532 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_541 = tile.matmul_modq %a_529, %b_533 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_542 = tile.matmul_modq %a_529, %b_534 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_543 = tile.matmul_modq %a_529, %b_535 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_544 = tile.matmul_modq %a_530, %b_532 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_545 = tile.matmul_modq %a_530, %b_533 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_546 = tile.matmul_modq %a_530, %b_534 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_547 = tile.matmul_modq %a_530, %b_535 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_548 = tile.matmul_modq %a_531, %b_532 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_549 = tile.matmul_modq %a_531, %b_533 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_550 = tile.matmul_modq %a_531, %b_534 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_551 = tile.matmul_modq %a_531, %b_535 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_552 = tile.add_modq %P_536 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_553 = tile.add_modq %P_537, %P_540 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_554 = tile.add_modq %P_538, %P_541, %P_544 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_555 = tile.add_modq %P_539, %P_542, %P_545, %P_548 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_556 = tile.add_modq %P_543, %P_546, %P_549 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_557 = tile.add_modq %P_547, %P_550 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_558 = tile.add_modq %P_551 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD12 = tile.recombine_modq %C_552, %C_553, %C_554, %C_555, %C_556, %C_557, %C_558 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE02 = tile.mul_modq %tC02, %tD02 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE22 = tile.mul_modq %tC12, %tD12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT02 = tile.mul_modq %tC02, %tD12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT12 = tile.mul_modq %tC12, %tD02 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE12 = tile.add_modq %tT02, %tT12 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268434977 = tile.const {name = "W_i_268434977", q = 268434977 : i64} : tensor<16x16xi32>
    %a_559, %a_560, %a_561, %a_562 = tile.decompose %W_i_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_563, %b_564, %b_565, %b_566 = tile.decompose %tE02 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_567 = tile.matmul_modq %a_559, %b_563 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_568 = tile.matmul_modq %a_559, %b_564 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_569 = tile.matmul_modq %a_559, %b_565 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_570 = tile.matmul_modq %a_559, %b_566 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_571 = tile.matmul_modq %a_560, %b_563 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_572 = tile.matmul_modq %a_560, %b_564 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_573 = tile.matmul_modq %a_560, %b_565 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_574 = tile.matmul_modq %a_560, %b_566 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_575 = tile.matmul_modq %a_561, %b_563 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_576 = tile.matmul_modq %a_561, %b_564 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_577 = tile.matmul_modq %a_561, %b_565 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_578 = tile.matmul_modq %a_561, %b_566 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_579 = tile.matmul_modq %a_562, %b_563 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_580 = tile.matmul_modq %a_562, %b_564 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_581 = tile.matmul_modq %a_562, %b_565 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_582 = tile.matmul_modq %a_562, %b_566 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_583 = tile.add_modq %P_567 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_584 = tile.add_modq %P_568, %P_571 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_585 = tile.add_modq %P_569, %P_572, %P_575 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_586 = tile.add_modq %P_570, %P_573, %P_576, %P_579 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_587 = tile.add_modq %P_574, %P_577, %P_580 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_588 = tile.add_modq %P_578, %P_581 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_589 = tile.add_modq %P_582 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e0_2 = tile.recombine_modq %C_583, %C_584, %C_585, %C_586, %C_587, %C_588, %C_589 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_590, %a_591, %a_592, %a_593 = tile.decompose %W_i_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_594, %b_595, %b_596, %b_597 = tile.decompose %tE12 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_598 = tile.matmul_modq %a_590, %b_594 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_599 = tile.matmul_modq %a_590, %b_595 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_600 = tile.matmul_modq %a_590, %b_596 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_601 = tile.matmul_modq %a_590, %b_597 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_602 = tile.matmul_modq %a_591, %b_594 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_603 = tile.matmul_modq %a_591, %b_595 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_604 = tile.matmul_modq %a_591, %b_596 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_605 = tile.matmul_modq %a_591, %b_597 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_606 = tile.matmul_modq %a_592, %b_594 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_607 = tile.matmul_modq %a_592, %b_595 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_608 = tile.matmul_modq %a_592, %b_596 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_609 = tile.matmul_modq %a_592, %b_597 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_610 = tile.matmul_modq %a_593, %b_594 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_611 = tile.matmul_modq %a_593, %b_595 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_612 = tile.matmul_modq %a_593, %b_596 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_613 = tile.matmul_modq %a_593, %b_597 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_614 = tile.add_modq %P_598 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_615 = tile.add_modq %P_599, %P_602 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_616 = tile.add_modq %P_600, %P_603, %P_606 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_617 = tile.add_modq %P_601, %P_604, %P_607, %P_610 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_618 = tile.add_modq %P_605, %P_608, %P_611 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_619 = tile.add_modq %P_609, %P_612 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_620 = tile.add_modq %P_613 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_2 = tile.recombine_modq %C_614, %C_615, %C_616, %C_617, %C_618, %C_619, %C_620 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_621, %a_622, %a_623, %a_624 = tile.decompose %W_i_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_625, %b_626, %b_627, %b_628 = tile.decompose %tE22 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_629 = tile.matmul_modq %a_621, %b_625 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_630 = tile.matmul_modq %a_621, %b_626 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_631 = tile.matmul_modq %a_621, %b_627 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_632 = tile.matmul_modq %a_621, %b_628 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_633 = tile.matmul_modq %a_622, %b_625 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_634 = tile.matmul_modq %a_622, %b_626 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_635 = tile.matmul_modq %a_622, %b_627 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_636 = tile.matmul_modq %a_622, %b_628 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_637 = tile.matmul_modq %a_623, %b_625 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_638 = tile.matmul_modq %a_623, %b_626 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_639 = tile.matmul_modq %a_623, %b_627 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_640 = tile.matmul_modq %a_623, %b_628 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_641 = tile.matmul_modq %a_624, %b_625 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_642 = tile.matmul_modq %a_624, %b_626 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_643 = tile.matmul_modq %a_624, %b_627 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_644 = tile.matmul_modq %a_624, %b_628 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_645 = tile.add_modq %P_629 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_646 = tile.add_modq %P_630, %P_633 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_647 = tile.add_modq %P_631, %P_634, %P_637 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_648 = tile.add_modq %P_632, %P_635, %P_638, %P_641 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_649 = tile.add_modq %P_636, %P_639, %P_642 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_650 = tile.add_modq %P_640, %P_643 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_651 = tile.add_modq %P_644 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_2 = tile.recombine_modq %C_645, %C_646, %C_647, %C_648, %C_649, %C_650, %C_651 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268433569 = tile.const {name = "W_f_268433569", q = 268433569 : i64} : tensor<16x16xi32>
    %a_652, %a_653, %a_654, %a_655 = tile.decompose %W_f_268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_656, %b_657, %b_658, %b_659 = tile.decompose %c03 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_660 = tile.matmul_modq %a_652, %b_656 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_661 = tile.matmul_modq %a_652, %b_657 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_662 = tile.matmul_modq %a_652, %b_658 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_663 = tile.matmul_modq %a_652, %b_659 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_664 = tile.matmul_modq %a_653, %b_656 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_665 = tile.matmul_modq %a_653, %b_657 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_666 = tile.matmul_modq %a_653, %b_658 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_667 = tile.matmul_modq %a_653, %b_659 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_668 = tile.matmul_modq %a_654, %b_656 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_669 = tile.matmul_modq %a_654, %b_657 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_670 = tile.matmul_modq %a_654, %b_658 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_671 = tile.matmul_modq %a_654, %b_659 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_672 = tile.matmul_modq %a_655, %b_656 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_673 = tile.matmul_modq %a_655, %b_657 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_674 = tile.matmul_modq %a_655, %b_658 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_675 = tile.matmul_modq %a_655, %b_659 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_676 = tile.add_modq %P_660 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_677 = tile.add_modq %P_661, %P_664 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_678 = tile.add_modq %P_662, %P_665, %P_668 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_679 = tile.add_modq %P_663, %P_666, %P_669, %P_672 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_680 = tile.add_modq %P_667, %P_670, %P_673 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_681 = tile.add_modq %P_671, %P_674 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_682 = tile.add_modq %P_675 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC03 = tile.recombine_modq %C_676, %C_677, %C_678, %C_679, %C_680, %C_681, %C_682 {radix = 128 : i64, q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_683, %a_684, %a_685, %a_686 = tile.decompose %W_f_268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_687, %b_688, %b_689, %b_690 = tile.decompose %c13 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_691 = tile.matmul_modq %a_683, %b_687 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_692 = tile.matmul_modq %a_683, %b_688 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_693 = tile.matmul_modq %a_683, %b_689 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_694 = tile.matmul_modq %a_683, %b_690 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_695 = tile.matmul_modq %a_684, %b_687 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_696 = tile.matmul_modq %a_684, %b_688 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_697 = tile.matmul_modq %a_684, %b_689 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_698 = tile.matmul_modq %a_684, %b_690 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_699 = tile.matmul_modq %a_685, %b_687 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_700 = tile.matmul_modq %a_685, %b_688 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_701 = tile.matmul_modq %a_685, %b_689 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_702 = tile.matmul_modq %a_685, %b_690 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_703 = tile.matmul_modq %a_686, %b_687 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_704 = tile.matmul_modq %a_686, %b_688 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_705 = tile.matmul_modq %a_686, %b_689 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_706 = tile.matmul_modq %a_686, %b_690 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_707 = tile.add_modq %P_691 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_708 = tile.add_modq %P_692, %P_695 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_709 = tile.add_modq %P_693, %P_696, %P_699 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_710 = tile.add_modq %P_694, %P_697, %P_700, %P_703 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_711 = tile.add_modq %P_698, %P_701, %P_704 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_712 = tile.add_modq %P_702, %P_705 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_713 = tile.add_modq %P_706 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC13 = tile.recombine_modq %C_707, %C_708, %C_709, %C_710, %C_711, %C_712, %C_713 {radix = 128 : i64, q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_714, %a_715, %a_716, %a_717 = tile.decompose %W_f_268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_718, %b_719, %b_720, %b_721 = tile.decompose %d03 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_722 = tile.matmul_modq %a_714, %b_718 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_723 = tile.matmul_modq %a_714, %b_719 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_724 = tile.matmul_modq %a_714, %b_720 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_725 = tile.matmul_modq %a_714, %b_721 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_726 = tile.matmul_modq %a_715, %b_718 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_727 = tile.matmul_modq %a_715, %b_719 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_728 = tile.matmul_modq %a_715, %b_720 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_729 = tile.matmul_modq %a_715, %b_721 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_730 = tile.matmul_modq %a_716, %b_718 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_731 = tile.matmul_modq %a_716, %b_719 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_732 = tile.matmul_modq %a_716, %b_720 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_733 = tile.matmul_modq %a_716, %b_721 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_734 = tile.matmul_modq %a_717, %b_718 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_735 = tile.matmul_modq %a_717, %b_719 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_736 = tile.matmul_modq %a_717, %b_720 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_737 = tile.matmul_modq %a_717, %b_721 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_738 = tile.add_modq %P_722 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_739 = tile.add_modq %P_723, %P_726 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_740 = tile.add_modq %P_724, %P_727, %P_730 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_741 = tile.add_modq %P_725, %P_728, %P_731, %P_734 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_742 = tile.add_modq %P_729, %P_732, %P_735 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_743 = tile.add_modq %P_733, %P_736 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_744 = tile.add_modq %P_737 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD03 = tile.recombine_modq %C_738, %C_739, %C_740, %C_741, %C_742, %C_743, %C_744 {radix = 128 : i64, q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_745, %a_746, %a_747, %a_748 = tile.decompose %W_f_268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_749, %b_750, %b_751, %b_752 = tile.decompose %d13 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_753 = tile.matmul_modq %a_745, %b_749 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_754 = tile.matmul_modq %a_745, %b_750 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_755 = tile.matmul_modq %a_745, %b_751 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_756 = tile.matmul_modq %a_745, %b_752 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_757 = tile.matmul_modq %a_746, %b_749 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_758 = tile.matmul_modq %a_746, %b_750 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_759 = tile.matmul_modq %a_746, %b_751 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_760 = tile.matmul_modq %a_746, %b_752 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_761 = tile.matmul_modq %a_747, %b_749 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_762 = tile.matmul_modq %a_747, %b_750 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_763 = tile.matmul_modq %a_747, %b_751 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_764 = tile.matmul_modq %a_747, %b_752 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_765 = tile.matmul_modq %a_748, %b_749 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_766 = tile.matmul_modq %a_748, %b_750 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_767 = tile.matmul_modq %a_748, %b_751 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_768 = tile.matmul_modq %a_748, %b_752 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_769 = tile.add_modq %P_753 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_770 = tile.add_modq %P_754, %P_757 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_771 = tile.add_modq %P_755, %P_758, %P_761 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_772 = tile.add_modq %P_756, %P_759, %P_762, %P_765 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_773 = tile.add_modq %P_760, %P_763, %P_766 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_774 = tile.add_modq %P_764, %P_767 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_775 = tile.add_modq %P_768 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD13 = tile.recombine_modq %C_769, %C_770, %C_771, %C_772, %C_773, %C_774, %C_775 {radix = 128 : i64, q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE03 = tile.mul_modq %tC03, %tD03 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE23 = tile.mul_modq %tC13, %tD13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT03 = tile.mul_modq %tC03, %tD13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT13 = tile.mul_modq %tC13, %tD03 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE13 = tile.add_modq %tT03, %tT13 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433569 = tile.const {name = "W_i_268433569", q = 268433569 : i64} : tensor<16x16xi32>
    %a_776, %a_777, %a_778, %a_779 = tile.decompose %W_i_268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_780, %b_781, %b_782, %b_783 = tile.decompose %tE03 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_784 = tile.matmul_modq %a_776, %b_780 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_785 = tile.matmul_modq %a_776, %b_781 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_786 = tile.matmul_modq %a_776, %b_782 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_787 = tile.matmul_modq %a_776, %b_783 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_788 = tile.matmul_modq %a_777, %b_780 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_789 = tile.matmul_modq %a_777, %b_781 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_790 = tile.matmul_modq %a_777, %b_782 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_791 = tile.matmul_modq %a_777, %b_783 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_792 = tile.matmul_modq %a_778, %b_780 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_793 = tile.matmul_modq %a_778, %b_781 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_794 = tile.matmul_modq %a_778, %b_782 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_795 = tile.matmul_modq %a_778, %b_783 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_796 = tile.matmul_modq %a_779, %b_780 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_797 = tile.matmul_modq %a_779, %b_781 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_798 = tile.matmul_modq %a_779, %b_782 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_799 = tile.matmul_modq %a_779, %b_783 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_800 = tile.add_modq %P_784 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_801 = tile.add_modq %P_785, %P_788 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_802 = tile.add_modq %P_786, %P_789, %P_792 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_803 = tile.add_modq %P_787, %P_790, %P_793, %P_796 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_804 = tile.add_modq %P_791, %P_794, %P_797 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_805 = tile.add_modq %P_795, %P_798 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_806 = tile.add_modq %P_799 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e0_3 = tile.recombine_modq %C_800, %C_801, %C_802, %C_803, %C_804, %C_805, %C_806 {radix = 128 : i64, q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_807, %a_808, %a_809, %a_810 = tile.decompose %W_i_268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_811, %b_812, %b_813, %b_814 = tile.decompose %tE13 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_815 = tile.matmul_modq %a_807, %b_811 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_816 = tile.matmul_modq %a_807, %b_812 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_817 = tile.matmul_modq %a_807, %b_813 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_818 = tile.matmul_modq %a_807, %b_814 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_819 = tile.matmul_modq %a_808, %b_811 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_820 = tile.matmul_modq %a_808, %b_812 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_821 = tile.matmul_modq %a_808, %b_813 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_822 = tile.matmul_modq %a_808, %b_814 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_823 = tile.matmul_modq %a_809, %b_811 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_824 = tile.matmul_modq %a_809, %b_812 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_825 = tile.matmul_modq %a_809, %b_813 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_826 = tile.matmul_modq %a_809, %b_814 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_827 = tile.matmul_modq %a_810, %b_811 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_828 = tile.matmul_modq %a_810, %b_812 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_829 = tile.matmul_modq %a_810, %b_813 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_830 = tile.matmul_modq %a_810, %b_814 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_831 = tile.add_modq %P_815 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_832 = tile.add_modq %P_816, %P_819 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_833 = tile.add_modq %P_817, %P_820, %P_823 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_834 = tile.add_modq %P_818, %P_821, %P_824, %P_827 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_835 = tile.add_modq %P_822, %P_825, %P_828 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_836 = tile.add_modq %P_826, %P_829 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_837 = tile.add_modq %P_830 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_3 = tile.recombine_modq %C_831, %C_832, %C_833, %C_834, %C_835, %C_836, %C_837 {radix = 128 : i64, q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_838, %a_839, %a_840, %a_841 = tile.decompose %W_i_268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_842, %b_843, %b_844, %b_845 = tile.decompose %tE23 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_846 = tile.matmul_modq %a_838, %b_842 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_847 = tile.matmul_modq %a_838, %b_843 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_848 = tile.matmul_modq %a_838, %b_844 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_849 = tile.matmul_modq %a_838, %b_845 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_850 = tile.matmul_modq %a_839, %b_842 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_851 = tile.matmul_modq %a_839, %b_843 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_852 = tile.matmul_modq %a_839, %b_844 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_853 = tile.matmul_modq %a_839, %b_845 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_854 = tile.matmul_modq %a_840, %b_842 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_855 = tile.matmul_modq %a_840, %b_843 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_856 = tile.matmul_modq %a_840, %b_844 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_857 = tile.matmul_modq %a_840, %b_845 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_858 = tile.matmul_modq %a_841, %b_842 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_859 = tile.matmul_modq %a_841, %b_843 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_860 = tile.matmul_modq %a_841, %b_844 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_861 = tile.matmul_modq %a_841, %b_845 {q = 268433569 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_862 = tile.add_modq %P_846 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_863 = tile.add_modq %P_847, %P_850 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_864 = tile.add_modq %P_848, %P_851, %P_854 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_865 = tile.add_modq %P_849, %P_852, %P_855, %P_858 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_866 = tile.add_modq %P_853, %P_856, %P_859 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_867 = tile.add_modq %P_857, %P_860 {q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_868 = tile.add_modq %P_861 {q = 268433569 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_3 = tile.recombine_modq %C_862, %C_863, %C_864, %C_865, %C_866, %C_867, %C_868 {radix = 128 : i64, q = 268433569 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268433441 = tile.const {name = "W_f_268433441", q = 268433441 : i64} : tensor<16x16xi32>
    %a_869, %a_870, %a_871, %a_872 = tile.decompose %W_f_268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_873, %b_874, %b_875, %b_876 = tile.decompose %c04 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_877 = tile.matmul_modq %a_869, %b_873 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_878 = tile.matmul_modq %a_869, %b_874 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_879 = tile.matmul_modq %a_869, %b_875 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_880 = tile.matmul_modq %a_869, %b_876 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_881 = tile.matmul_modq %a_870, %b_873 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_882 = tile.matmul_modq %a_870, %b_874 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_883 = tile.matmul_modq %a_870, %b_875 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_884 = tile.matmul_modq %a_870, %b_876 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_885 = tile.matmul_modq %a_871, %b_873 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_886 = tile.matmul_modq %a_871, %b_874 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_887 = tile.matmul_modq %a_871, %b_875 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_888 = tile.matmul_modq %a_871, %b_876 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_889 = tile.matmul_modq %a_872, %b_873 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_890 = tile.matmul_modq %a_872, %b_874 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_891 = tile.matmul_modq %a_872, %b_875 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_892 = tile.matmul_modq %a_872, %b_876 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_893 = tile.add_modq %P_877 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_894 = tile.add_modq %P_878, %P_881 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_895 = tile.add_modq %P_879, %P_882, %P_885 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_896 = tile.add_modq %P_880, %P_883, %P_886, %P_889 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_897 = tile.add_modq %P_884, %P_887, %P_890 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_898 = tile.add_modq %P_888, %P_891 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_899 = tile.add_modq %P_892 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC04 = tile.recombine_modq %C_893, %C_894, %C_895, %C_896, %C_897, %C_898, %C_899 {radix = 128 : i64, q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_900, %a_901, %a_902, %a_903 = tile.decompose %W_f_268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_904, %b_905, %b_906, %b_907 = tile.decompose %c14 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_908 = tile.matmul_modq %a_900, %b_904 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_909 = tile.matmul_modq %a_900, %b_905 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_910 = tile.matmul_modq %a_900, %b_906 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_911 = tile.matmul_modq %a_900, %b_907 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_912 = tile.matmul_modq %a_901, %b_904 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_913 = tile.matmul_modq %a_901, %b_905 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_914 = tile.matmul_modq %a_901, %b_906 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_915 = tile.matmul_modq %a_901, %b_907 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_916 = tile.matmul_modq %a_902, %b_904 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_917 = tile.matmul_modq %a_902, %b_905 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_918 = tile.matmul_modq %a_902, %b_906 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_919 = tile.matmul_modq %a_902, %b_907 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_920 = tile.matmul_modq %a_903, %b_904 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_921 = tile.matmul_modq %a_903, %b_905 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_922 = tile.matmul_modq %a_903, %b_906 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_923 = tile.matmul_modq %a_903, %b_907 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_924 = tile.add_modq %P_908 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_925 = tile.add_modq %P_909, %P_912 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_926 = tile.add_modq %P_910, %P_913, %P_916 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_927 = tile.add_modq %P_911, %P_914, %P_917, %P_920 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_928 = tile.add_modq %P_915, %P_918, %P_921 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_929 = tile.add_modq %P_919, %P_922 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_930 = tile.add_modq %P_923 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC14 = tile.recombine_modq %C_924, %C_925, %C_926, %C_927, %C_928, %C_929, %C_930 {radix = 128 : i64, q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_931, %a_932, %a_933, %a_934 = tile.decompose %W_f_268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_935, %b_936, %b_937, %b_938 = tile.decompose %d04 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_939 = tile.matmul_modq %a_931, %b_935 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_940 = tile.matmul_modq %a_931, %b_936 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_941 = tile.matmul_modq %a_931, %b_937 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_942 = tile.matmul_modq %a_931, %b_938 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_943 = tile.matmul_modq %a_932, %b_935 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_944 = tile.matmul_modq %a_932, %b_936 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_945 = tile.matmul_modq %a_932, %b_937 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_946 = tile.matmul_modq %a_932, %b_938 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_947 = tile.matmul_modq %a_933, %b_935 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_948 = tile.matmul_modq %a_933, %b_936 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_949 = tile.matmul_modq %a_933, %b_937 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_950 = tile.matmul_modq %a_933, %b_938 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_951 = tile.matmul_modq %a_934, %b_935 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_952 = tile.matmul_modq %a_934, %b_936 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_953 = tile.matmul_modq %a_934, %b_937 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_954 = tile.matmul_modq %a_934, %b_938 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_955 = tile.add_modq %P_939 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_956 = tile.add_modq %P_940, %P_943 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_957 = tile.add_modq %P_941, %P_944, %P_947 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_958 = tile.add_modq %P_942, %P_945, %P_948, %P_951 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_959 = tile.add_modq %P_946, %P_949, %P_952 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_960 = tile.add_modq %P_950, %P_953 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_961 = tile.add_modq %P_954 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD04 = tile.recombine_modq %C_955, %C_956, %C_957, %C_958, %C_959, %C_960, %C_961 {radix = 128 : i64, q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_962, %a_963, %a_964, %a_965 = tile.decompose %W_f_268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_966, %b_967, %b_968, %b_969 = tile.decompose %d14 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_970 = tile.matmul_modq %a_962, %b_966 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_971 = tile.matmul_modq %a_962, %b_967 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_972 = tile.matmul_modq %a_962, %b_968 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_973 = tile.matmul_modq %a_962, %b_969 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_974 = tile.matmul_modq %a_963, %b_966 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_975 = tile.matmul_modq %a_963, %b_967 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_976 = tile.matmul_modq %a_963, %b_968 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_977 = tile.matmul_modq %a_963, %b_969 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_978 = tile.matmul_modq %a_964, %b_966 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_979 = tile.matmul_modq %a_964, %b_967 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_980 = tile.matmul_modq %a_964, %b_968 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_981 = tile.matmul_modq %a_964, %b_969 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_982 = tile.matmul_modq %a_965, %b_966 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_983 = tile.matmul_modq %a_965, %b_967 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_984 = tile.matmul_modq %a_965, %b_968 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_985 = tile.matmul_modq %a_965, %b_969 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_986 = tile.add_modq %P_970 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_987 = tile.add_modq %P_971, %P_974 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_988 = tile.add_modq %P_972, %P_975, %P_978 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_989 = tile.add_modq %P_973, %P_976, %P_979, %P_982 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_990 = tile.add_modq %P_977, %P_980, %P_983 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_991 = tile.add_modq %P_981, %P_984 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_992 = tile.add_modq %P_985 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD14 = tile.recombine_modq %C_986, %C_987, %C_988, %C_989, %C_990, %C_991, %C_992 {radix = 128 : i64, q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE04 = tile.mul_modq %tC04, %tD04 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE24 = tile.mul_modq %tC14, %tD14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT04 = tile.mul_modq %tC04, %tD14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT14 = tile.mul_modq %tC14, %tD04 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE14 = tile.add_modq %tT04, %tT14 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433441 = tile.const {name = "W_i_268433441", q = 268433441 : i64} : tensor<16x16xi32>
    %a_993, %a_994, %a_995, %a_996 = tile.decompose %W_i_268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_997, %b_998, %b_999, %b_1000 = tile.decompose %tE04 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1001 = tile.matmul_modq %a_993, %b_997 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1002 = tile.matmul_modq %a_993, %b_998 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1003 = tile.matmul_modq %a_993, %b_999 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1004 = tile.matmul_modq %a_993, %b_1000 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1005 = tile.matmul_modq %a_994, %b_997 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1006 = tile.matmul_modq %a_994, %b_998 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1007 = tile.matmul_modq %a_994, %b_999 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1008 = tile.matmul_modq %a_994, %b_1000 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1009 = tile.matmul_modq %a_995, %b_997 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1010 = tile.matmul_modq %a_995, %b_998 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1011 = tile.matmul_modq %a_995, %b_999 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1012 = tile.matmul_modq %a_995, %b_1000 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1013 = tile.matmul_modq %a_996, %b_997 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1014 = tile.matmul_modq %a_996, %b_998 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1015 = tile.matmul_modq %a_996, %b_999 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1016 = tile.matmul_modq %a_996, %b_1000 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1017 = tile.add_modq %P_1001 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1018 = tile.add_modq %P_1002, %P_1005 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1019 = tile.add_modq %P_1003, %P_1006, %P_1009 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1020 = tile.add_modq %P_1004, %P_1007, %P_1010, %P_1013 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1021 = tile.add_modq %P_1008, %P_1011, %P_1014 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1022 = tile.add_modq %P_1012, %P_1015 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1023 = tile.add_modq %P_1016 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e0_4 = tile.recombine_modq %C_1017, %C_1018, %C_1019, %C_1020, %C_1021, %C_1022, %C_1023 {radix = 128 : i64, q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1024, %a_1025, %a_1026, %a_1027 = tile.decompose %W_i_268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1028, %b_1029, %b_1030, %b_1031 = tile.decompose %tE14 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1032 = tile.matmul_modq %a_1024, %b_1028 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1033 = tile.matmul_modq %a_1024, %b_1029 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1034 = tile.matmul_modq %a_1024, %b_1030 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1035 = tile.matmul_modq %a_1024, %b_1031 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1036 = tile.matmul_modq %a_1025, %b_1028 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1037 = tile.matmul_modq %a_1025, %b_1029 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1038 = tile.matmul_modq %a_1025, %b_1030 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1039 = tile.matmul_modq %a_1025, %b_1031 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1040 = tile.matmul_modq %a_1026, %b_1028 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1041 = tile.matmul_modq %a_1026, %b_1029 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1042 = tile.matmul_modq %a_1026, %b_1030 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1043 = tile.matmul_modq %a_1026, %b_1031 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1044 = tile.matmul_modq %a_1027, %b_1028 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1045 = tile.matmul_modq %a_1027, %b_1029 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1046 = tile.matmul_modq %a_1027, %b_1030 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1047 = tile.matmul_modq %a_1027, %b_1031 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1048 = tile.add_modq %P_1032 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1049 = tile.add_modq %P_1033, %P_1036 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1050 = tile.add_modq %P_1034, %P_1037, %P_1040 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1051 = tile.add_modq %P_1035, %P_1038, %P_1041, %P_1044 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1052 = tile.add_modq %P_1039, %P_1042, %P_1045 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1053 = tile.add_modq %P_1043, %P_1046 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1054 = tile.add_modq %P_1047 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_4 = tile.recombine_modq %C_1048, %C_1049, %C_1050, %C_1051, %C_1052, %C_1053, %C_1054 {radix = 128 : i64, q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1055, %a_1056, %a_1057, %a_1058 = tile.decompose %W_i_268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1059, %b_1060, %b_1061, %b_1062 = tile.decompose %tE24 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1063 = tile.matmul_modq %a_1055, %b_1059 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1064 = tile.matmul_modq %a_1055, %b_1060 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1065 = tile.matmul_modq %a_1055, %b_1061 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1066 = tile.matmul_modq %a_1055, %b_1062 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1067 = tile.matmul_modq %a_1056, %b_1059 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1068 = tile.matmul_modq %a_1056, %b_1060 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1069 = tile.matmul_modq %a_1056, %b_1061 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1070 = tile.matmul_modq %a_1056, %b_1062 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1071 = tile.matmul_modq %a_1057, %b_1059 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1072 = tile.matmul_modq %a_1057, %b_1060 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1073 = tile.matmul_modq %a_1057, %b_1061 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1074 = tile.matmul_modq %a_1057, %b_1062 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1075 = tile.matmul_modq %a_1058, %b_1059 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1076 = tile.matmul_modq %a_1058, %b_1060 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1077 = tile.matmul_modq %a_1058, %b_1061 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1078 = tile.matmul_modq %a_1058, %b_1062 {q = 268433441 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1079 = tile.add_modq %P_1063 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1080 = tile.add_modq %P_1064, %P_1067 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1081 = tile.add_modq %P_1065, %P_1068, %P_1071 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1082 = tile.add_modq %P_1066, %P_1069, %P_1072, %P_1075 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1083 = tile.add_modq %P_1070, %P_1073, %P_1076 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1084 = tile.add_modq %P_1074, %P_1077 {q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1085 = tile.add_modq %P_1078 {q = 268433441 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_4 = tile.recombine_modq %C_1079, %C_1080, %C_1081, %C_1082, %C_1083, %C_1084, %C_1085 {radix = 128 : i64, q = 268433441 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268433377 = tile.const {name = "W_f_268433377", q = 268433377 : i64} : tensor<16x16xi32>
    %a_1086, %a_1087, %a_1088, %a_1089 = tile.decompose %W_f_268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1090, %b_1091, %b_1092, %b_1093 = tile.decompose %c05 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1094 = tile.matmul_modq %a_1086, %b_1090 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1095 = tile.matmul_modq %a_1086, %b_1091 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1096 = tile.matmul_modq %a_1086, %b_1092 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1097 = tile.matmul_modq %a_1086, %b_1093 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1098 = tile.matmul_modq %a_1087, %b_1090 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1099 = tile.matmul_modq %a_1087, %b_1091 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1100 = tile.matmul_modq %a_1087, %b_1092 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1101 = tile.matmul_modq %a_1087, %b_1093 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1102 = tile.matmul_modq %a_1088, %b_1090 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1103 = tile.matmul_modq %a_1088, %b_1091 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1104 = tile.matmul_modq %a_1088, %b_1092 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1105 = tile.matmul_modq %a_1088, %b_1093 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1106 = tile.matmul_modq %a_1089, %b_1090 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1107 = tile.matmul_modq %a_1089, %b_1091 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1108 = tile.matmul_modq %a_1089, %b_1092 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1109 = tile.matmul_modq %a_1089, %b_1093 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1110 = tile.add_modq %P_1094 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1111 = tile.add_modq %P_1095, %P_1098 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1112 = tile.add_modq %P_1096, %P_1099, %P_1102 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1113 = tile.add_modq %P_1097, %P_1100, %P_1103, %P_1106 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1114 = tile.add_modq %P_1101, %P_1104, %P_1107 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1115 = tile.add_modq %P_1105, %P_1108 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1116 = tile.add_modq %P_1109 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC05 = tile.recombine_modq %C_1110, %C_1111, %C_1112, %C_1113, %C_1114, %C_1115, %C_1116 {radix = 128 : i64, q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1117, %a_1118, %a_1119, %a_1120 = tile.decompose %W_f_268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1121, %b_1122, %b_1123, %b_1124 = tile.decompose %c15 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1125 = tile.matmul_modq %a_1117, %b_1121 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1126 = tile.matmul_modq %a_1117, %b_1122 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1127 = tile.matmul_modq %a_1117, %b_1123 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1128 = tile.matmul_modq %a_1117, %b_1124 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1129 = tile.matmul_modq %a_1118, %b_1121 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1130 = tile.matmul_modq %a_1118, %b_1122 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1131 = tile.matmul_modq %a_1118, %b_1123 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1132 = tile.matmul_modq %a_1118, %b_1124 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1133 = tile.matmul_modq %a_1119, %b_1121 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1134 = tile.matmul_modq %a_1119, %b_1122 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1135 = tile.matmul_modq %a_1119, %b_1123 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1136 = tile.matmul_modq %a_1119, %b_1124 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1137 = tile.matmul_modq %a_1120, %b_1121 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1138 = tile.matmul_modq %a_1120, %b_1122 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1139 = tile.matmul_modq %a_1120, %b_1123 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1140 = tile.matmul_modq %a_1120, %b_1124 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1141 = tile.add_modq %P_1125 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1142 = tile.add_modq %P_1126, %P_1129 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1143 = tile.add_modq %P_1127, %P_1130, %P_1133 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1144 = tile.add_modq %P_1128, %P_1131, %P_1134, %P_1137 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1145 = tile.add_modq %P_1132, %P_1135, %P_1138 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1146 = tile.add_modq %P_1136, %P_1139 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1147 = tile.add_modq %P_1140 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC15 = tile.recombine_modq %C_1141, %C_1142, %C_1143, %C_1144, %C_1145, %C_1146, %C_1147 {radix = 128 : i64, q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1148, %a_1149, %a_1150, %a_1151 = tile.decompose %W_f_268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1152, %b_1153, %b_1154, %b_1155 = tile.decompose %d05 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1156 = tile.matmul_modq %a_1148, %b_1152 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1157 = tile.matmul_modq %a_1148, %b_1153 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1158 = tile.matmul_modq %a_1148, %b_1154 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1159 = tile.matmul_modq %a_1148, %b_1155 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1160 = tile.matmul_modq %a_1149, %b_1152 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1161 = tile.matmul_modq %a_1149, %b_1153 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1162 = tile.matmul_modq %a_1149, %b_1154 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1163 = tile.matmul_modq %a_1149, %b_1155 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1164 = tile.matmul_modq %a_1150, %b_1152 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1165 = tile.matmul_modq %a_1150, %b_1153 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1166 = tile.matmul_modq %a_1150, %b_1154 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1167 = tile.matmul_modq %a_1150, %b_1155 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1168 = tile.matmul_modq %a_1151, %b_1152 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1169 = tile.matmul_modq %a_1151, %b_1153 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1170 = tile.matmul_modq %a_1151, %b_1154 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1171 = tile.matmul_modq %a_1151, %b_1155 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1172 = tile.add_modq %P_1156 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1173 = tile.add_modq %P_1157, %P_1160 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1174 = tile.add_modq %P_1158, %P_1161, %P_1164 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1175 = tile.add_modq %P_1159, %P_1162, %P_1165, %P_1168 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1176 = tile.add_modq %P_1163, %P_1166, %P_1169 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1177 = tile.add_modq %P_1167, %P_1170 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1178 = tile.add_modq %P_1171 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD05 = tile.recombine_modq %C_1172, %C_1173, %C_1174, %C_1175, %C_1176, %C_1177, %C_1178 {radix = 128 : i64, q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1179, %a_1180, %a_1181, %a_1182 = tile.decompose %W_f_268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1183, %b_1184, %b_1185, %b_1186 = tile.decompose %d15 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1187 = tile.matmul_modq %a_1179, %b_1183 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1188 = tile.matmul_modq %a_1179, %b_1184 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1189 = tile.matmul_modq %a_1179, %b_1185 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1190 = tile.matmul_modq %a_1179, %b_1186 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1191 = tile.matmul_modq %a_1180, %b_1183 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1192 = tile.matmul_modq %a_1180, %b_1184 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1193 = tile.matmul_modq %a_1180, %b_1185 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1194 = tile.matmul_modq %a_1180, %b_1186 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1195 = tile.matmul_modq %a_1181, %b_1183 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1196 = tile.matmul_modq %a_1181, %b_1184 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1197 = tile.matmul_modq %a_1181, %b_1185 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1198 = tile.matmul_modq %a_1181, %b_1186 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1199 = tile.matmul_modq %a_1182, %b_1183 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1200 = tile.matmul_modq %a_1182, %b_1184 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1201 = tile.matmul_modq %a_1182, %b_1185 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1202 = tile.matmul_modq %a_1182, %b_1186 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1203 = tile.add_modq %P_1187 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1204 = tile.add_modq %P_1188, %P_1191 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1205 = tile.add_modq %P_1189, %P_1192, %P_1195 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1206 = tile.add_modq %P_1190, %P_1193, %P_1196, %P_1199 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1207 = tile.add_modq %P_1194, %P_1197, %P_1200 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1208 = tile.add_modq %P_1198, %P_1201 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1209 = tile.add_modq %P_1202 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD15 = tile.recombine_modq %C_1203, %C_1204, %C_1205, %C_1206, %C_1207, %C_1208, %C_1209 {radix = 128 : i64, q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE05 = tile.mul_modq %tC05, %tD05 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE25 = tile.mul_modq %tC15, %tD15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT05 = tile.mul_modq %tC05, %tD15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT15 = tile.mul_modq %tC15, %tD05 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE15 = tile.add_modq %tT05, %tT15 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433377 = tile.const {name = "W_i_268433377", q = 268433377 : i64} : tensor<16x16xi32>
    %a_1210, %a_1211, %a_1212, %a_1213 = tile.decompose %W_i_268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1214, %b_1215, %b_1216, %b_1217 = tile.decompose %tE05 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1218 = tile.matmul_modq %a_1210, %b_1214 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1219 = tile.matmul_modq %a_1210, %b_1215 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1220 = tile.matmul_modq %a_1210, %b_1216 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1221 = tile.matmul_modq %a_1210, %b_1217 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1222 = tile.matmul_modq %a_1211, %b_1214 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1223 = tile.matmul_modq %a_1211, %b_1215 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1224 = tile.matmul_modq %a_1211, %b_1216 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1225 = tile.matmul_modq %a_1211, %b_1217 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1226 = tile.matmul_modq %a_1212, %b_1214 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1227 = tile.matmul_modq %a_1212, %b_1215 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1228 = tile.matmul_modq %a_1212, %b_1216 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1229 = tile.matmul_modq %a_1212, %b_1217 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1230 = tile.matmul_modq %a_1213, %b_1214 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1231 = tile.matmul_modq %a_1213, %b_1215 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1232 = tile.matmul_modq %a_1213, %b_1216 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1233 = tile.matmul_modq %a_1213, %b_1217 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1234 = tile.add_modq %P_1218 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1235 = tile.add_modq %P_1219, %P_1222 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1236 = tile.add_modq %P_1220, %P_1223, %P_1226 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1237 = tile.add_modq %P_1221, %P_1224, %P_1227, %P_1230 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1238 = tile.add_modq %P_1225, %P_1228, %P_1231 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1239 = tile.add_modq %P_1229, %P_1232 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1240 = tile.add_modq %P_1233 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e0_5 = tile.recombine_modq %C_1234, %C_1235, %C_1236, %C_1237, %C_1238, %C_1239, %C_1240 {radix = 128 : i64, q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1241, %a_1242, %a_1243, %a_1244 = tile.decompose %W_i_268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1245, %b_1246, %b_1247, %b_1248 = tile.decompose %tE15 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1249 = tile.matmul_modq %a_1241, %b_1245 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1250 = tile.matmul_modq %a_1241, %b_1246 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1251 = tile.matmul_modq %a_1241, %b_1247 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1252 = tile.matmul_modq %a_1241, %b_1248 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1253 = tile.matmul_modq %a_1242, %b_1245 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1254 = tile.matmul_modq %a_1242, %b_1246 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1255 = tile.matmul_modq %a_1242, %b_1247 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1256 = tile.matmul_modq %a_1242, %b_1248 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1257 = tile.matmul_modq %a_1243, %b_1245 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1258 = tile.matmul_modq %a_1243, %b_1246 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1259 = tile.matmul_modq %a_1243, %b_1247 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1260 = tile.matmul_modq %a_1243, %b_1248 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1261 = tile.matmul_modq %a_1244, %b_1245 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1262 = tile.matmul_modq %a_1244, %b_1246 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1263 = tile.matmul_modq %a_1244, %b_1247 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1264 = tile.matmul_modq %a_1244, %b_1248 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1265 = tile.add_modq %P_1249 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1266 = tile.add_modq %P_1250, %P_1253 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1267 = tile.add_modq %P_1251, %P_1254, %P_1257 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1268 = tile.add_modq %P_1252, %P_1255, %P_1258, %P_1261 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1269 = tile.add_modq %P_1256, %P_1259, %P_1262 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1270 = tile.add_modq %P_1260, %P_1263 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1271 = tile.add_modq %P_1264 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_5 = tile.recombine_modq %C_1265, %C_1266, %C_1267, %C_1268, %C_1269, %C_1270, %C_1271 {radix = 128 : i64, q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1272, %a_1273, %a_1274, %a_1275 = tile.decompose %W_i_268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1276, %b_1277, %b_1278, %b_1279 = tile.decompose %tE25 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1280 = tile.matmul_modq %a_1272, %b_1276 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1281 = tile.matmul_modq %a_1272, %b_1277 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1282 = tile.matmul_modq %a_1272, %b_1278 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1283 = tile.matmul_modq %a_1272, %b_1279 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1284 = tile.matmul_modq %a_1273, %b_1276 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1285 = tile.matmul_modq %a_1273, %b_1277 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1286 = tile.matmul_modq %a_1273, %b_1278 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1287 = tile.matmul_modq %a_1273, %b_1279 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1288 = tile.matmul_modq %a_1274, %b_1276 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1289 = tile.matmul_modq %a_1274, %b_1277 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1290 = tile.matmul_modq %a_1274, %b_1278 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1291 = tile.matmul_modq %a_1274, %b_1279 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1292 = tile.matmul_modq %a_1275, %b_1276 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1293 = tile.matmul_modq %a_1275, %b_1277 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1294 = tile.matmul_modq %a_1275, %b_1278 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1295 = tile.matmul_modq %a_1275, %b_1279 {q = 268433377 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1296 = tile.add_modq %P_1280 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1297 = tile.add_modq %P_1281, %P_1284 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1298 = tile.add_modq %P_1282, %P_1285, %P_1288 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1299 = tile.add_modq %P_1283, %P_1286, %P_1289, %P_1292 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1300 = tile.add_modq %P_1287, %P_1290, %P_1293 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1301 = tile.add_modq %P_1291, %P_1294 {q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1302 = tile.add_modq %P_1295 {q = 268433377 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_5 = tile.recombine_modq %C_1296, %C_1297, %C_1298, %C_1299, %C_1300, %C_1301, %C_1302 {radix = 128 : i64, q = 268433377 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268433057 = tile.const {name = "W_f_268433057", q = 268433057 : i64} : tensor<16x16xi32>
    %a_1303, %a_1304, %a_1305, %a_1306 = tile.decompose %W_f_268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1307, %b_1308, %b_1309, %b_1310 = tile.decompose %c06 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1311 = tile.matmul_modq %a_1303, %b_1307 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1312 = tile.matmul_modq %a_1303, %b_1308 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1313 = tile.matmul_modq %a_1303, %b_1309 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1314 = tile.matmul_modq %a_1303, %b_1310 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1315 = tile.matmul_modq %a_1304, %b_1307 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1316 = tile.matmul_modq %a_1304, %b_1308 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1317 = tile.matmul_modq %a_1304, %b_1309 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1318 = tile.matmul_modq %a_1304, %b_1310 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1319 = tile.matmul_modq %a_1305, %b_1307 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1320 = tile.matmul_modq %a_1305, %b_1308 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1321 = tile.matmul_modq %a_1305, %b_1309 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1322 = tile.matmul_modq %a_1305, %b_1310 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1323 = tile.matmul_modq %a_1306, %b_1307 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1324 = tile.matmul_modq %a_1306, %b_1308 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1325 = tile.matmul_modq %a_1306, %b_1309 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1326 = tile.matmul_modq %a_1306, %b_1310 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1327 = tile.add_modq %P_1311 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1328 = tile.add_modq %P_1312, %P_1315 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1329 = tile.add_modq %P_1313, %P_1316, %P_1319 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1330 = tile.add_modq %P_1314, %P_1317, %P_1320, %P_1323 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1331 = tile.add_modq %P_1318, %P_1321, %P_1324 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1332 = tile.add_modq %P_1322, %P_1325 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1333 = tile.add_modq %P_1326 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC06 = tile.recombine_modq %C_1327, %C_1328, %C_1329, %C_1330, %C_1331, %C_1332, %C_1333 {radix = 128 : i64, q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1334, %a_1335, %a_1336, %a_1337 = tile.decompose %W_f_268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1338, %b_1339, %b_1340, %b_1341 = tile.decompose %c16 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1342 = tile.matmul_modq %a_1334, %b_1338 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1343 = tile.matmul_modq %a_1334, %b_1339 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1344 = tile.matmul_modq %a_1334, %b_1340 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1345 = tile.matmul_modq %a_1334, %b_1341 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1346 = tile.matmul_modq %a_1335, %b_1338 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1347 = tile.matmul_modq %a_1335, %b_1339 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1348 = tile.matmul_modq %a_1335, %b_1340 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1349 = tile.matmul_modq %a_1335, %b_1341 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1350 = tile.matmul_modq %a_1336, %b_1338 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1351 = tile.matmul_modq %a_1336, %b_1339 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1352 = tile.matmul_modq %a_1336, %b_1340 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1353 = tile.matmul_modq %a_1336, %b_1341 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1354 = tile.matmul_modq %a_1337, %b_1338 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1355 = tile.matmul_modq %a_1337, %b_1339 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1356 = tile.matmul_modq %a_1337, %b_1340 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1357 = tile.matmul_modq %a_1337, %b_1341 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1358 = tile.add_modq %P_1342 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1359 = tile.add_modq %P_1343, %P_1346 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1360 = tile.add_modq %P_1344, %P_1347, %P_1350 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1361 = tile.add_modq %P_1345, %P_1348, %P_1351, %P_1354 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1362 = tile.add_modq %P_1349, %P_1352, %P_1355 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1363 = tile.add_modq %P_1353, %P_1356 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1364 = tile.add_modq %P_1357 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tC16 = tile.recombine_modq %C_1358, %C_1359, %C_1360, %C_1361, %C_1362, %C_1363, %C_1364 {radix = 128 : i64, q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1365, %a_1366, %a_1367, %a_1368 = tile.decompose %W_f_268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1369, %b_1370, %b_1371, %b_1372 = tile.decompose %d06 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1373 = tile.matmul_modq %a_1365, %b_1369 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1374 = tile.matmul_modq %a_1365, %b_1370 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1375 = tile.matmul_modq %a_1365, %b_1371 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1376 = tile.matmul_modq %a_1365, %b_1372 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1377 = tile.matmul_modq %a_1366, %b_1369 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1378 = tile.matmul_modq %a_1366, %b_1370 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1379 = tile.matmul_modq %a_1366, %b_1371 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1380 = tile.matmul_modq %a_1366, %b_1372 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1381 = tile.matmul_modq %a_1367, %b_1369 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1382 = tile.matmul_modq %a_1367, %b_1370 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1383 = tile.matmul_modq %a_1367, %b_1371 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1384 = tile.matmul_modq %a_1367, %b_1372 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1385 = tile.matmul_modq %a_1368, %b_1369 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1386 = tile.matmul_modq %a_1368, %b_1370 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1387 = tile.matmul_modq %a_1368, %b_1371 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1388 = tile.matmul_modq %a_1368, %b_1372 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1389 = tile.add_modq %P_1373 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1390 = tile.add_modq %P_1374, %P_1377 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1391 = tile.add_modq %P_1375, %P_1378, %P_1381 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1392 = tile.add_modq %P_1376, %P_1379, %P_1382, %P_1385 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1393 = tile.add_modq %P_1380, %P_1383, %P_1386 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1394 = tile.add_modq %P_1384, %P_1387 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1395 = tile.add_modq %P_1388 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD06 = tile.recombine_modq %C_1389, %C_1390, %C_1391, %C_1392, %C_1393, %C_1394, %C_1395 {radix = 128 : i64, q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1396, %a_1397, %a_1398, %a_1399 = tile.decompose %W_f_268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1400, %b_1401, %b_1402, %b_1403 = tile.decompose %d16 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1404 = tile.matmul_modq %a_1396, %b_1400 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1405 = tile.matmul_modq %a_1396, %b_1401 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1406 = tile.matmul_modq %a_1396, %b_1402 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1407 = tile.matmul_modq %a_1396, %b_1403 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1408 = tile.matmul_modq %a_1397, %b_1400 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1409 = tile.matmul_modq %a_1397, %b_1401 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1410 = tile.matmul_modq %a_1397, %b_1402 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1411 = tile.matmul_modq %a_1397, %b_1403 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1412 = tile.matmul_modq %a_1398, %b_1400 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1413 = tile.matmul_modq %a_1398, %b_1401 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1414 = tile.matmul_modq %a_1398, %b_1402 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1415 = tile.matmul_modq %a_1398, %b_1403 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1416 = tile.matmul_modq %a_1399, %b_1400 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1417 = tile.matmul_modq %a_1399, %b_1401 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1418 = tile.matmul_modq %a_1399, %b_1402 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1419 = tile.matmul_modq %a_1399, %b_1403 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1420 = tile.add_modq %P_1404 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1421 = tile.add_modq %P_1405, %P_1408 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1422 = tile.add_modq %P_1406, %P_1409, %P_1412 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1423 = tile.add_modq %P_1407, %P_1410, %P_1413, %P_1416 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1424 = tile.add_modq %P_1411, %P_1414, %P_1417 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1425 = tile.add_modq %P_1415, %P_1418 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1426 = tile.add_modq %P_1419 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %tD16 = tile.recombine_modq %C_1420, %C_1421, %C_1422, %C_1423, %C_1424, %C_1425, %C_1426 {radix = 128 : i64, q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE06 = tile.mul_modq %tC06, %tD06 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE26 = tile.mul_modq %tC16, %tD16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT06 = tile.mul_modq %tC06, %tD16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tT16 = tile.mul_modq %tC16, %tD06 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %tE16 = tile.add_modq %tT06, %tT16 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433057 = tile.const {name = "W_i_268433057", q = 268433057 : i64} : tensor<16x16xi32>
    %a_1427, %a_1428, %a_1429, %a_1430 = tile.decompose %W_i_268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1431, %b_1432, %b_1433, %b_1434 = tile.decompose %tE06 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1435 = tile.matmul_modq %a_1427, %b_1431 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1436 = tile.matmul_modq %a_1427, %b_1432 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1437 = tile.matmul_modq %a_1427, %b_1433 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1438 = tile.matmul_modq %a_1427, %b_1434 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1439 = tile.matmul_modq %a_1428, %b_1431 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1440 = tile.matmul_modq %a_1428, %b_1432 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1441 = tile.matmul_modq %a_1428, %b_1433 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1442 = tile.matmul_modq %a_1428, %b_1434 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1443 = tile.matmul_modq %a_1429, %b_1431 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1444 = tile.matmul_modq %a_1429, %b_1432 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1445 = tile.matmul_modq %a_1429, %b_1433 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1446 = tile.matmul_modq %a_1429, %b_1434 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1447 = tile.matmul_modq %a_1430, %b_1431 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1448 = tile.matmul_modq %a_1430, %b_1432 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1449 = tile.matmul_modq %a_1430, %b_1433 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1450 = tile.matmul_modq %a_1430, %b_1434 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1451 = tile.add_modq %P_1435 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1452 = tile.add_modq %P_1436, %P_1439 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1453 = tile.add_modq %P_1437, %P_1440, %P_1443 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1454 = tile.add_modq %P_1438, %P_1441, %P_1444, %P_1447 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1455 = tile.add_modq %P_1442, %P_1445, %P_1448 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1456 = tile.add_modq %P_1446, %P_1449 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1457 = tile.add_modq %P_1450 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e0_6 = tile.recombine_modq %C_1451, %C_1452, %C_1453, %C_1454, %C_1455, %C_1456, %C_1457 {radix = 128 : i64, q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1458, %a_1459, %a_1460, %a_1461 = tile.decompose %W_i_268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1462, %b_1463, %b_1464, %b_1465 = tile.decompose %tE16 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1466 = tile.matmul_modq %a_1458, %b_1462 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1467 = tile.matmul_modq %a_1458, %b_1463 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1468 = tile.matmul_modq %a_1458, %b_1464 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1469 = tile.matmul_modq %a_1458, %b_1465 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1470 = tile.matmul_modq %a_1459, %b_1462 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1471 = tile.matmul_modq %a_1459, %b_1463 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1472 = tile.matmul_modq %a_1459, %b_1464 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1473 = tile.matmul_modq %a_1459, %b_1465 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1474 = tile.matmul_modq %a_1460, %b_1462 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1475 = tile.matmul_modq %a_1460, %b_1463 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1476 = tile.matmul_modq %a_1460, %b_1464 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1477 = tile.matmul_modq %a_1460, %b_1465 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1478 = tile.matmul_modq %a_1461, %b_1462 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1479 = tile.matmul_modq %a_1461, %b_1463 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1480 = tile.matmul_modq %a_1461, %b_1464 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1481 = tile.matmul_modq %a_1461, %b_1465 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1482 = tile.add_modq %P_1466 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1483 = tile.add_modq %P_1467, %P_1470 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1484 = tile.add_modq %P_1468, %P_1471, %P_1474 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1485 = tile.add_modq %P_1469, %P_1472, %P_1475, %P_1478 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1486 = tile.add_modq %P_1473, %P_1476, %P_1479 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1487 = tile.add_modq %P_1477, %P_1480 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1488 = tile.add_modq %P_1481 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_6 = tile.recombine_modq %C_1482, %C_1483, %C_1484, %C_1485, %C_1486, %C_1487, %C_1488 {radix = 128 : i64, q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_1489, %a_1490, %a_1491, %a_1492 = tile.decompose %W_i_268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_1493, %b_1494, %b_1495, %b_1496 = tile.decompose %tE26 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_1497 = tile.matmul_modq %a_1489, %b_1493 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1498 = tile.matmul_modq %a_1489, %b_1494 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1499 = tile.matmul_modq %a_1489, %b_1495 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1500 = tile.matmul_modq %a_1489, %b_1496 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1501 = tile.matmul_modq %a_1490, %b_1493 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1502 = tile.matmul_modq %a_1490, %b_1494 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1503 = tile.matmul_modq %a_1490, %b_1495 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1504 = tile.matmul_modq %a_1490, %b_1496 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1505 = tile.matmul_modq %a_1491, %b_1493 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1506 = tile.matmul_modq %a_1491, %b_1494 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1507 = tile.matmul_modq %a_1491, %b_1495 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1508 = tile.matmul_modq %a_1491, %b_1496 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1509 = tile.matmul_modq %a_1492, %b_1493 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1510 = tile.matmul_modq %a_1492, %b_1494 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1511 = tile.matmul_modq %a_1492, %b_1495 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_1512 = tile.matmul_modq %a_1492, %b_1496 {q = 268433057 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_1513 = tile.add_modq %P_1497 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_1514 = tile.add_modq %P_1498, %P_1501 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1515 = tile.add_modq %P_1499, %P_1502, %P_1505 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1516 = tile.add_modq %P_1500, %P_1503, %P_1506, %P_1509 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1517 = tile.add_modq %P_1504, %P_1507, %P_1510 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1518 = tile.add_modq %P_1508, %P_1511 {q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_1519 = tile.add_modq %P_1512 {q = 268433057 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_6 = tile.recombine_modq %C_1513, %C_1514, %C_1515, %C_1516, %C_1517, %C_1518, %C_1519 {radix = 128 : i64, q = 268433057 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
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
    %a_1520, %a_1521, %a_1522, %a_1523 = tile.decompose %fbcD__268435361__268435009__268434977__268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1524, %b_1525, %b_1526, %b_1527 = tile.decompose %rs0_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1528 = tile.matmul_modq %a_1520, %b_1524 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1529 = tile.matmul_modq %a_1520, %b_1525 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1530 = tile.matmul_modq %a_1520, %b_1526 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1531 = tile.matmul_modq %a_1520, %b_1527 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1532 = tile.matmul_modq %a_1521, %b_1524 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1533 = tile.matmul_modq %a_1521, %b_1525 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1534 = tile.matmul_modq %a_1521, %b_1526 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1535 = tile.matmul_modq %a_1521, %b_1527 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1536 = tile.matmul_modq %a_1522, %b_1524 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1537 = tile.matmul_modq %a_1522, %b_1525 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1538 = tile.matmul_modq %a_1522, %b_1526 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1539 = tile.matmul_modq %a_1522, %b_1527 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1540 = tile.matmul_modq %a_1523, %b_1524 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1541 = tile.matmul_modq %a_1523, %b_1525 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1542 = tile.matmul_modq %a_1523, %b_1526 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1543 = tile.matmul_modq %a_1523, %b_1527 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1544 = tile.add_modq %P_1528 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1545 = tile.add_modq %P_1529, %P_1532 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1546 = tile.add_modq %P_1530, %P_1533, %P_1536 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1547 = tile.add_modq %P_1531, %P_1534, %P_1537, %P_1540 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1548 = tile.add_modq %P_1535, %P_1538, %P_1541 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1549 = tile.add_modq %P_1539, %P_1542 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1550 = tile.add_modq %P_1543 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_fbc_268433569 = tile.recombine_modq %C_1544, %C_1545, %C_1546, %C_1547, %C_1548, %C_1549, %C_1550 {radix = 128 : i64, q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0 = tile.reshape %rs0_fbc0_fbc_268433569 {shape = [16], q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433441 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433441"} : tensor<1x3xi32>
    %a_1551, %a_1552, %a_1553, %a_1554 = tile.decompose %fbcD__268435361__268435009__268434977__268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1555, %b_1556, %b_1557, %b_1558 = tile.decompose %rs0_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1559 = tile.matmul_modq %a_1551, %b_1555 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1560 = tile.matmul_modq %a_1551, %b_1556 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1561 = tile.matmul_modq %a_1551, %b_1557 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1562 = tile.matmul_modq %a_1551, %b_1558 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1563 = tile.matmul_modq %a_1552, %b_1555 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1564 = tile.matmul_modq %a_1552, %b_1556 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1565 = tile.matmul_modq %a_1552, %b_1557 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1566 = tile.matmul_modq %a_1552, %b_1558 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1567 = tile.matmul_modq %a_1553, %b_1555 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1568 = tile.matmul_modq %a_1553, %b_1556 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1569 = tile.matmul_modq %a_1553, %b_1557 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1570 = tile.matmul_modq %a_1553, %b_1558 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1571 = tile.matmul_modq %a_1554, %b_1555 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1572 = tile.matmul_modq %a_1554, %b_1556 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1573 = tile.matmul_modq %a_1554, %b_1557 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1574 = tile.matmul_modq %a_1554, %b_1558 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1575 = tile.add_modq %P_1559 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1576 = tile.add_modq %P_1560, %P_1563 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1577 = tile.add_modq %P_1561, %P_1564, %P_1567 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1578 = tile.add_modq %P_1562, %P_1565, %P_1568, %P_1571 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1579 = tile.add_modq %P_1566, %P_1569, %P_1572 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1580 = tile.add_modq %P_1570, %P_1573 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1581 = tile.add_modq %P_1574 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_fbc_268433441 = tile.recombine_modq %C_1575, %C_1576, %C_1577, %C_1578, %C_1579, %C_1580, %C_1581 {radix = 128 : i64, q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc1 = tile.reshape %rs0_fbc0_fbc_268433441 {shape = [16], q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433377 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433377"} : tensor<1x3xi32>
    %a_1582, %a_1583, %a_1584, %a_1585 = tile.decompose %fbcD__268435361__268435009__268434977__268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1586, %b_1587, %b_1588, %b_1589 = tile.decompose %rs0_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1590 = tile.matmul_modq %a_1582, %b_1586 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1591 = tile.matmul_modq %a_1582, %b_1587 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1592 = tile.matmul_modq %a_1582, %b_1588 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1593 = tile.matmul_modq %a_1582, %b_1589 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1594 = tile.matmul_modq %a_1583, %b_1586 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1595 = tile.matmul_modq %a_1583, %b_1587 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1596 = tile.matmul_modq %a_1583, %b_1588 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1597 = tile.matmul_modq %a_1583, %b_1589 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1598 = tile.matmul_modq %a_1584, %b_1586 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1599 = tile.matmul_modq %a_1584, %b_1587 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1600 = tile.matmul_modq %a_1584, %b_1588 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1601 = tile.matmul_modq %a_1584, %b_1589 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1602 = tile.matmul_modq %a_1585, %b_1586 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1603 = tile.matmul_modq %a_1585, %b_1587 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1604 = tile.matmul_modq %a_1585, %b_1588 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1605 = tile.matmul_modq %a_1585, %b_1589 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1606 = tile.add_modq %P_1590 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1607 = tile.add_modq %P_1591, %P_1594 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1608 = tile.add_modq %P_1592, %P_1595, %P_1598 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1609 = tile.add_modq %P_1593, %P_1596, %P_1599, %P_1602 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1610 = tile.add_modq %P_1597, %P_1600, %P_1603 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1611 = tile.add_modq %P_1601, %P_1604 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1612 = tile.add_modq %P_1605 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_fbc_268433377 = tile.recombine_modq %C_1606, %C_1607, %C_1608, %C_1609, %C_1610, %C_1611, %C_1612 {radix = 128 : i64, q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc2 = tile.reshape %rs0_fbc0_fbc_268433377 {shape = [16], q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433057 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433057"} : tensor<1x3xi32>
    %a_1613, %a_1614, %a_1615, %a_1616 = tile.decompose %fbcD__268435361__268435009__268434977__268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1617, %b_1618, %b_1619, %b_1620 = tile.decompose %rs0_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1621 = tile.matmul_modq %a_1613, %b_1617 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1622 = tile.matmul_modq %a_1613, %b_1618 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1623 = tile.matmul_modq %a_1613, %b_1619 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1624 = tile.matmul_modq %a_1613, %b_1620 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1625 = tile.matmul_modq %a_1614, %b_1617 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1626 = tile.matmul_modq %a_1614, %b_1618 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1627 = tile.matmul_modq %a_1614, %b_1619 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1628 = tile.matmul_modq %a_1614, %b_1620 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1629 = tile.matmul_modq %a_1615, %b_1617 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1630 = tile.matmul_modq %a_1615, %b_1618 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1631 = tile.matmul_modq %a_1615, %b_1619 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1632 = tile.matmul_modq %a_1615, %b_1620 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1633 = tile.matmul_modq %a_1616, %b_1617 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1634 = tile.matmul_modq %a_1616, %b_1618 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1635 = tile.matmul_modq %a_1616, %b_1619 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1636 = tile.matmul_modq %a_1616, %b_1620 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1637 = tile.add_modq %P_1621 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1638 = tile.add_modq %P_1622, %P_1625 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1639 = tile.add_modq %P_1623, %P_1626, %P_1629 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1640 = tile.add_modq %P_1624, %P_1627, %P_1630, %P_1633 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1641 = tile.add_modq %P_1628, %P_1631, %P_1634 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1642 = tile.add_modq %P_1632, %P_1635 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1643 = tile.add_modq %P_1636 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_fbc0_fbc_268433057 = tile.recombine_modq %C_1637, %C_1638, %C_1639, %C_1640, %C_1641, %C_1642, %C_1643 {radix = 128 : i64, q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_1644, %a_1645, %a_1646, %a_1647, %a_1648, %a_1649 = tile.decompose %crnsF__268433569__268433441__268433377__268433057__38 {radix = 128 : i64, nlimbs = 6 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1650, %b_1651, %b_1652, %b_1653 = tile.decompose %rs0_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %wP_1654 = tile.matmul_wide %a_1644, %b_1650 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1655 = tile.matmul_wide %a_1644, %b_1651 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1656 = tile.matmul_wide %a_1644, %b_1652 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1657 = tile.matmul_wide %a_1644, %b_1653 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1658 = tile.matmul_wide %a_1645, %b_1650 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1659 = tile.matmul_wide %a_1645, %b_1651 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1660 = tile.matmul_wide %a_1645, %b_1652 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1661 = tile.matmul_wide %a_1645, %b_1653 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1662 = tile.matmul_wide %a_1646, %b_1650 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1663 = tile.matmul_wide %a_1646, %b_1651 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1664 = tile.matmul_wide %a_1646, %b_1652 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1665 = tile.matmul_wide %a_1646, %b_1653 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1666 = tile.matmul_wide %a_1647, %b_1650 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1667 = tile.matmul_wide %a_1647, %b_1651 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1668 = tile.matmul_wide %a_1647, %b_1652 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1669 = tile.matmul_wide %a_1647, %b_1653 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1670 = tile.matmul_wide %a_1648, %b_1650 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1671 = tile.matmul_wide %a_1648, %b_1651 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1672 = tile.matmul_wide %a_1648, %b_1652 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1673 = tile.matmul_wide %a_1648, %b_1653 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1674 = tile.matmul_wide %a_1649, %b_1650 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1675 = tile.matmul_wide %a_1649, %b_1651 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1676 = tile.matmul_wide %a_1649, %b_1652 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1677 = tile.matmul_wide %a_1649, %b_1653 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %rs0_etQ0_s = tile.recombine_wide %wP_1654, %wP_1655, %wP_1656, %wP_1657, %wP_1658, %wP_1659, %wP_1660, %wP_1661, %wP_1662, %wP_1663, %wP_1664, %wP_1665, %wP_1666, %wP_1667, %wP_1668, %wP_1669, %wP_1670, %wP_1671, %wP_1672, %wP_1673, %wP_1674, %wP_1675, %wP_1676, %wP_1677 {radix = 128 : i64, exps = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8]} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_k = tile.fixed_slice %rs0_etQ0_s {shift = 38 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_neg = tile.fixed_slice %rs0_etQ0_s {shift = 37 : i64, mask = 1 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsE__268433569__268433441__268433377__268433057__268435361 = tile.const {name = "crnsE_(268433569, 268433441, 268433377, 268433057)_268435361"} : tensor<1x4xi32>
    %a_1678, %a_1679, %a_1680, %a_1681 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1682, %b_1683, %b_1684, %b_1685 = tile.decompose %rs0_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_1686 = tile.matmul_modq %a_1678, %b_1682 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1687 = tile.matmul_modq %a_1678, %b_1683 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1688 = tile.matmul_modq %a_1678, %b_1684 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1689 = tile.matmul_modq %a_1678, %b_1685 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1690 = tile.matmul_modq %a_1679, %b_1682 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1691 = tile.matmul_modq %a_1679, %b_1683 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1692 = tile.matmul_modq %a_1679, %b_1684 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1693 = tile.matmul_modq %a_1679, %b_1685 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1694 = tile.matmul_modq %a_1680, %b_1682 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1695 = tile.matmul_modq %a_1680, %b_1683 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1696 = tile.matmul_modq %a_1680, %b_1684 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1697 = tile.matmul_modq %a_1680, %b_1685 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1698 = tile.matmul_modq %a_1681, %b_1682 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1699 = tile.matmul_modq %a_1681, %b_1683 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1700 = tile.matmul_modq %a_1681, %b_1684 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1701 = tile.matmul_modq %a_1681, %b_1685 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_1702 = tile.add_modq %P_1686 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1703 = tile.add_modq %P_1687, %P_1690 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1704 = tile.add_modq %P_1688, %P_1691, %P_1694 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1705 = tile.add_modq %P_1689, %P_1692, %P_1695, %P_1698 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1706 = tile.add_modq %P_1693, %P_1696, %P_1699 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1707 = tile.add_modq %P_1697, %P_1700 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1708 = tile.add_modq %P_1701 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_xE_268435361 = tile.recombine_modq %C_1702, %C_1703, %C_1704, %C_1705, %C_1706, %C_1707, %C_1708 {radix = 128 : i64, q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsg__268433569__268433441__268433377__268433057__268435361 = tile.const {name = "crnsg_(268433569, 268433441, 268433377, 268433057)_268435361", q = 268435361 : i64} : tensor<1x1xi32>
    %rs0_etQ0_kg_268435361 = tile.mul_modq %rs0_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_acc_268435361 = tile.add_modq %rs0_etQ0_xE_268435361, %rs0_etQ0_kg_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsMmod__268433569__268433441__268433377__268433057__268435361 = tile.const {name = "crnsMmod_(268433569, 268433441, 268433377, 268433057)_268435361", q = 268435361 : i64} : tensor<1x1xi32>
    %rs0_etQ0_negM_268435361 = tile.mul_modq %rs0_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_accs_268435361 = tile.sub_modq %rs0_etQ0_acc_268435361, %rs0_etQ0_negM_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0 = tile.reshape %rs0_etQ0_accs_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %crnsE__268433569__268433441__268433377__268433057__268435009 = tile.const {name = "crnsE_(268433569, 268433441, 268433377, 268433057)_268435009"} : tensor<1x4xi32>
    %a_1709, %a_1710, %a_1711, %a_1712 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1713, %b_1714, %b_1715, %b_1716 = tile.decompose %rs0_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_1717 = tile.matmul_modq %a_1709, %b_1713 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1718 = tile.matmul_modq %a_1709, %b_1714 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1719 = tile.matmul_modq %a_1709, %b_1715 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1720 = tile.matmul_modq %a_1709, %b_1716 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1721 = tile.matmul_modq %a_1710, %b_1713 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1722 = tile.matmul_modq %a_1710, %b_1714 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1723 = tile.matmul_modq %a_1710, %b_1715 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1724 = tile.matmul_modq %a_1710, %b_1716 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1725 = tile.matmul_modq %a_1711, %b_1713 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1726 = tile.matmul_modq %a_1711, %b_1714 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1727 = tile.matmul_modq %a_1711, %b_1715 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1728 = tile.matmul_modq %a_1711, %b_1716 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1729 = tile.matmul_modq %a_1712, %b_1713 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1730 = tile.matmul_modq %a_1712, %b_1714 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1731 = tile.matmul_modq %a_1712, %b_1715 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1732 = tile.matmul_modq %a_1712, %b_1716 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_1733 = tile.add_modq %P_1717 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1734 = tile.add_modq %P_1718, %P_1721 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1735 = tile.add_modq %P_1719, %P_1722, %P_1725 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1736 = tile.add_modq %P_1720, %P_1723, %P_1726, %P_1729 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1737 = tile.add_modq %P_1724, %P_1727, %P_1730 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1738 = tile.add_modq %P_1728, %P_1731 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1739 = tile.add_modq %P_1732 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_xE_268435009 = tile.recombine_modq %C_1733, %C_1734, %C_1735, %C_1736, %C_1737, %C_1738, %C_1739 {radix = 128 : i64, q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsg__268433569__268433441__268433377__268433057__268435009 = tile.const {name = "crnsg_(268433569, 268433441, 268433377, 268433057)_268435009", q = 268435009 : i64} : tensor<1x1xi32>
    %rs0_etQ0_kg_268435009 = tile.mul_modq %rs0_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_acc_268435009 = tile.add_modq %rs0_etQ0_xE_268435009, %rs0_etQ0_kg_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %crnsMmod__268433569__268433441__268433377__268433057__268435009 = tile.const {name = "crnsMmod_(268433569, 268433441, 268433377, 268433057)_268435009", q = 268435009 : i64} : tensor<1x1xi32>
    %rs0_etQ0_negM_268435009 = tile.mul_modq %rs0_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_accs_268435009 = tile.sub_modq %rs0_etQ0_acc_268435009, %rs0_etQ0_negM_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ1 = tile.reshape %rs0_etQ0_accs_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %crnsE__268433569__268433441__268433377__268433057__268434977 = tile.const {name = "crnsE_(268433569, 268433441, 268433377, 268433057)_268434977"} : tensor<1x4xi32>
    %a_1740, %a_1741, %a_1742, %a_1743 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1744, %b_1745, %b_1746, %b_1747 = tile.decompose %rs0_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_1748 = tile.matmul_modq %a_1740, %b_1744 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1749 = tile.matmul_modq %a_1740, %b_1745 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1750 = tile.matmul_modq %a_1740, %b_1746 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1751 = tile.matmul_modq %a_1740, %b_1747 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1752 = tile.matmul_modq %a_1741, %b_1744 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1753 = tile.matmul_modq %a_1741, %b_1745 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1754 = tile.matmul_modq %a_1741, %b_1746 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1755 = tile.matmul_modq %a_1741, %b_1747 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1756 = tile.matmul_modq %a_1742, %b_1744 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1757 = tile.matmul_modq %a_1742, %b_1745 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1758 = tile.matmul_modq %a_1742, %b_1746 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1759 = tile.matmul_modq %a_1742, %b_1747 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1760 = tile.matmul_modq %a_1743, %b_1744 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1761 = tile.matmul_modq %a_1743, %b_1745 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1762 = tile.matmul_modq %a_1743, %b_1746 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1763 = tile.matmul_modq %a_1743, %b_1747 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_1764 = tile.add_modq %P_1748 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1765 = tile.add_modq %P_1749, %P_1752 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1766 = tile.add_modq %P_1750, %P_1753, %P_1756 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1767 = tile.add_modq %P_1751, %P_1754, %P_1757, %P_1760 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1768 = tile.add_modq %P_1755, %P_1758, %P_1761 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1769 = tile.add_modq %P_1759, %P_1762 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1770 = tile.add_modq %P_1763 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs0_etQ0_xE_268434977 = tile.recombine_modq %C_1764, %C_1765, %C_1766, %C_1767, %C_1768, %C_1769, %C_1770 {radix = 128 : i64, q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_1771, %a_1772, %a_1773, %a_1774 = tile.decompose %fbcD__268435361__268435009__268434977__268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1775, %b_1776, %b_1777, %b_1778 = tile.decompose %rs1_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1779 = tile.matmul_modq %a_1771, %b_1775 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1780 = tile.matmul_modq %a_1771, %b_1776 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1781 = tile.matmul_modq %a_1771, %b_1777 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1782 = tile.matmul_modq %a_1771, %b_1778 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1783 = tile.matmul_modq %a_1772, %b_1775 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1784 = tile.matmul_modq %a_1772, %b_1776 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1785 = tile.matmul_modq %a_1772, %b_1777 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1786 = tile.matmul_modq %a_1772, %b_1778 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1787 = tile.matmul_modq %a_1773, %b_1775 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1788 = tile.matmul_modq %a_1773, %b_1776 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1789 = tile.matmul_modq %a_1773, %b_1777 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1790 = tile.matmul_modq %a_1773, %b_1778 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1791 = tile.matmul_modq %a_1774, %b_1775 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1792 = tile.matmul_modq %a_1774, %b_1776 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1793 = tile.matmul_modq %a_1774, %b_1777 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1794 = tile.matmul_modq %a_1774, %b_1778 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1795 = tile.add_modq %P_1779 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1796 = tile.add_modq %P_1780, %P_1783 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1797 = tile.add_modq %P_1781, %P_1784, %P_1787 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1798 = tile.add_modq %P_1782, %P_1785, %P_1788, %P_1791 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1799 = tile.add_modq %P_1786, %P_1789, %P_1792 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1800 = tile.add_modq %P_1790, %P_1793 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1801 = tile.add_modq %P_1794 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_fbc_268433569 = tile.recombine_modq %C_1795, %C_1796, %C_1797, %C_1798, %C_1799, %C_1800, %C_1801 {radix = 128 : i64, q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0 = tile.reshape %rs1_fbc0_fbc_268433569 {shape = [16], q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_1802, %a_1803, %a_1804, %a_1805 = tile.decompose %fbcD__268435361__268435009__268434977__268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1806, %b_1807, %b_1808, %b_1809 = tile.decompose %rs1_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1810 = tile.matmul_modq %a_1802, %b_1806 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1811 = tile.matmul_modq %a_1802, %b_1807 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1812 = tile.matmul_modq %a_1802, %b_1808 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1813 = tile.matmul_modq %a_1802, %b_1809 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1814 = tile.matmul_modq %a_1803, %b_1806 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1815 = tile.matmul_modq %a_1803, %b_1807 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1816 = tile.matmul_modq %a_1803, %b_1808 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1817 = tile.matmul_modq %a_1803, %b_1809 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1818 = tile.matmul_modq %a_1804, %b_1806 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1819 = tile.matmul_modq %a_1804, %b_1807 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1820 = tile.matmul_modq %a_1804, %b_1808 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1821 = tile.matmul_modq %a_1804, %b_1809 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1822 = tile.matmul_modq %a_1805, %b_1806 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1823 = tile.matmul_modq %a_1805, %b_1807 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1824 = tile.matmul_modq %a_1805, %b_1808 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1825 = tile.matmul_modq %a_1805, %b_1809 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1826 = tile.add_modq %P_1810 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1827 = tile.add_modq %P_1811, %P_1814 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1828 = tile.add_modq %P_1812, %P_1815, %P_1818 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1829 = tile.add_modq %P_1813, %P_1816, %P_1819, %P_1822 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1830 = tile.add_modq %P_1817, %P_1820, %P_1823 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1831 = tile.add_modq %P_1821, %P_1824 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1832 = tile.add_modq %P_1825 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_fbc_268433441 = tile.recombine_modq %C_1826, %C_1827, %C_1828, %C_1829, %C_1830, %C_1831, %C_1832 {radix = 128 : i64, q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc1 = tile.reshape %rs1_fbc0_fbc_268433441 {shape = [16], q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_1833, %a_1834, %a_1835, %a_1836 = tile.decompose %fbcD__268435361__268435009__268434977__268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1837, %b_1838, %b_1839, %b_1840 = tile.decompose %rs1_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1841 = tile.matmul_modq %a_1833, %b_1837 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1842 = tile.matmul_modq %a_1833, %b_1838 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1843 = tile.matmul_modq %a_1833, %b_1839 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1844 = tile.matmul_modq %a_1833, %b_1840 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1845 = tile.matmul_modq %a_1834, %b_1837 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1846 = tile.matmul_modq %a_1834, %b_1838 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1847 = tile.matmul_modq %a_1834, %b_1839 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1848 = tile.matmul_modq %a_1834, %b_1840 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1849 = tile.matmul_modq %a_1835, %b_1837 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1850 = tile.matmul_modq %a_1835, %b_1838 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1851 = tile.matmul_modq %a_1835, %b_1839 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1852 = tile.matmul_modq %a_1835, %b_1840 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1853 = tile.matmul_modq %a_1836, %b_1837 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1854 = tile.matmul_modq %a_1836, %b_1838 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1855 = tile.matmul_modq %a_1836, %b_1839 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1856 = tile.matmul_modq %a_1836, %b_1840 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1857 = tile.add_modq %P_1841 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1858 = tile.add_modq %P_1842, %P_1845 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1859 = tile.add_modq %P_1843, %P_1846, %P_1849 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1860 = tile.add_modq %P_1844, %P_1847, %P_1850, %P_1853 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1861 = tile.add_modq %P_1848, %P_1851, %P_1854 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1862 = tile.add_modq %P_1852, %P_1855 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1863 = tile.add_modq %P_1856 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_fbc_268433377 = tile.recombine_modq %C_1857, %C_1858, %C_1859, %C_1860, %C_1861, %C_1862, %C_1863 {radix = 128 : i64, q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc2 = tile.reshape %rs1_fbc0_fbc_268433377 {shape = [16], q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_1864, %a_1865, %a_1866, %a_1867 = tile.decompose %fbcD__268435361__268435009__268434977__268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_1868, %b_1869, %b_1870, %b_1871 = tile.decompose %rs1_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_1872 = tile.matmul_modq %a_1864, %b_1868 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1873 = tile.matmul_modq %a_1864, %b_1869 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1874 = tile.matmul_modq %a_1864, %b_1870 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1875 = tile.matmul_modq %a_1864, %b_1871 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1876 = tile.matmul_modq %a_1865, %b_1868 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1877 = tile.matmul_modq %a_1865, %b_1869 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1878 = tile.matmul_modq %a_1865, %b_1870 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1879 = tile.matmul_modq %a_1865, %b_1871 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1880 = tile.matmul_modq %a_1866, %b_1868 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1881 = tile.matmul_modq %a_1866, %b_1869 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1882 = tile.matmul_modq %a_1866, %b_1870 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1883 = tile.matmul_modq %a_1866, %b_1871 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1884 = tile.matmul_modq %a_1867, %b_1868 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1885 = tile.matmul_modq %a_1867, %b_1869 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1886 = tile.matmul_modq %a_1867, %b_1870 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_1887 = tile.matmul_modq %a_1867, %b_1871 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_1888 = tile.add_modq %P_1872 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1889 = tile.add_modq %P_1873, %P_1876 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1890 = tile.add_modq %P_1874, %P_1877, %P_1880 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1891 = tile.add_modq %P_1875, %P_1878, %P_1881, %P_1884 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1892 = tile.add_modq %P_1879, %P_1882, %P_1885 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1893 = tile.add_modq %P_1883, %P_1886 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1894 = tile.add_modq %P_1887 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_fbc0_fbc_268433057 = tile.recombine_modq %C_1888, %C_1889, %C_1890, %C_1891, %C_1892, %C_1893, %C_1894 {radix = 128 : i64, q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_1895, %a_1896, %a_1897, %a_1898, %a_1899, %a_1900 = tile.decompose %crnsF__268433569__268433441__268433377__268433057__38 {radix = 128 : i64, nlimbs = 6 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1901, %b_1902, %b_1903, %b_1904 = tile.decompose %rs1_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %wP_1905 = tile.matmul_wide %a_1895, %b_1901 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1906 = tile.matmul_wide %a_1895, %b_1902 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1907 = tile.matmul_wide %a_1895, %b_1903 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1908 = tile.matmul_wide %a_1895, %b_1904 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1909 = tile.matmul_wide %a_1896, %b_1901 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1910 = tile.matmul_wide %a_1896, %b_1902 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1911 = tile.matmul_wide %a_1896, %b_1903 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1912 = tile.matmul_wide %a_1896, %b_1904 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1913 = tile.matmul_wide %a_1897, %b_1901 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1914 = tile.matmul_wide %a_1897, %b_1902 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1915 = tile.matmul_wide %a_1897, %b_1903 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1916 = tile.matmul_wide %a_1897, %b_1904 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1917 = tile.matmul_wide %a_1898, %b_1901 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1918 = tile.matmul_wide %a_1898, %b_1902 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1919 = tile.matmul_wide %a_1898, %b_1903 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1920 = tile.matmul_wide %a_1898, %b_1904 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1921 = tile.matmul_wide %a_1899, %b_1901 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1922 = tile.matmul_wide %a_1899, %b_1902 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1923 = tile.matmul_wide %a_1899, %b_1903 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1924 = tile.matmul_wide %a_1899, %b_1904 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1925 = tile.matmul_wide %a_1900, %b_1901 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1926 = tile.matmul_wide %a_1900, %b_1902 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1927 = tile.matmul_wide %a_1900, %b_1903 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_1928 = tile.matmul_wide %a_1900, %b_1904 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %rs1_etQ0_s = tile.recombine_wide %wP_1905, %wP_1906, %wP_1907, %wP_1908, %wP_1909, %wP_1910, %wP_1911, %wP_1912, %wP_1913, %wP_1914, %wP_1915, %wP_1916, %wP_1917, %wP_1918, %wP_1919, %wP_1920, %wP_1921, %wP_1922, %wP_1923, %wP_1924, %wP_1925, %wP_1926, %wP_1927, %wP_1928 {radix = 128 : i64, exps = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8]} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_k = tile.fixed_slice %rs1_etQ0_s {shift = 38 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_neg = tile.fixed_slice %rs1_etQ0_s {shift = 37 : i64, mask = 1 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %a_1929, %a_1930, %a_1931, %a_1932 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1933, %b_1934, %b_1935, %b_1936 = tile.decompose %rs1_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_1937 = tile.matmul_modq %a_1929, %b_1933 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1938 = tile.matmul_modq %a_1929, %b_1934 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1939 = tile.matmul_modq %a_1929, %b_1935 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1940 = tile.matmul_modq %a_1929, %b_1936 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1941 = tile.matmul_modq %a_1930, %b_1933 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1942 = tile.matmul_modq %a_1930, %b_1934 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1943 = tile.matmul_modq %a_1930, %b_1935 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1944 = tile.matmul_modq %a_1930, %b_1936 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1945 = tile.matmul_modq %a_1931, %b_1933 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1946 = tile.matmul_modq %a_1931, %b_1934 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1947 = tile.matmul_modq %a_1931, %b_1935 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1948 = tile.matmul_modq %a_1931, %b_1936 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1949 = tile.matmul_modq %a_1932, %b_1933 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1950 = tile.matmul_modq %a_1932, %b_1934 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1951 = tile.matmul_modq %a_1932, %b_1935 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1952 = tile.matmul_modq %a_1932, %b_1936 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_1953 = tile.add_modq %P_1937 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1954 = tile.add_modq %P_1938, %P_1941 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1955 = tile.add_modq %P_1939, %P_1942, %P_1945 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1956 = tile.add_modq %P_1940, %P_1943, %P_1946, %P_1949 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1957 = tile.add_modq %P_1944, %P_1947, %P_1950 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1958 = tile.add_modq %P_1948, %P_1951 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1959 = tile.add_modq %P_1952 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_xE_268435361 = tile.recombine_modq %C_1953, %C_1954, %C_1955, %C_1956, %C_1957, %C_1958, %C_1959 {radix = 128 : i64, q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_kg_268435361 = tile.mul_modq %rs1_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_acc_268435361 = tile.add_modq %rs1_etQ0_xE_268435361, %rs1_etQ0_kg_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_negM_268435361 = tile.mul_modq %rs1_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_accs_268435361 = tile.sub_modq %rs1_etQ0_acc_268435361, %rs1_etQ0_negM_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0 = tile.reshape %rs1_etQ0_accs_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_1960, %a_1961, %a_1962, %a_1963 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1964, %b_1965, %b_1966, %b_1967 = tile.decompose %rs1_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_1968 = tile.matmul_modq %a_1960, %b_1964 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1969 = tile.matmul_modq %a_1960, %b_1965 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1970 = tile.matmul_modq %a_1960, %b_1966 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1971 = tile.matmul_modq %a_1960, %b_1967 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1972 = tile.matmul_modq %a_1961, %b_1964 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1973 = tile.matmul_modq %a_1961, %b_1965 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1974 = tile.matmul_modq %a_1961, %b_1966 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1975 = tile.matmul_modq %a_1961, %b_1967 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1976 = tile.matmul_modq %a_1962, %b_1964 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1977 = tile.matmul_modq %a_1962, %b_1965 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1978 = tile.matmul_modq %a_1962, %b_1966 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1979 = tile.matmul_modq %a_1962, %b_1967 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1980 = tile.matmul_modq %a_1963, %b_1964 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1981 = tile.matmul_modq %a_1963, %b_1965 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1982 = tile.matmul_modq %a_1963, %b_1966 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_1983 = tile.matmul_modq %a_1963, %b_1967 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_1984 = tile.add_modq %P_1968 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1985 = tile.add_modq %P_1969, %P_1972 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1986 = tile.add_modq %P_1970, %P_1973, %P_1976 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1987 = tile.add_modq %P_1971, %P_1974, %P_1977, %P_1980 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1988 = tile.add_modq %P_1975, %P_1978, %P_1981 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1989 = tile.add_modq %P_1979, %P_1982 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_1990 = tile.add_modq %P_1983 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_xE_268435009 = tile.recombine_modq %C_1984, %C_1985, %C_1986, %C_1987, %C_1988, %C_1989, %C_1990 {radix = 128 : i64, q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_kg_268435009 = tile.mul_modq %rs1_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_acc_268435009 = tile.add_modq %rs1_etQ0_xE_268435009, %rs1_etQ0_kg_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_negM_268435009 = tile.mul_modq %rs1_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_accs_268435009 = tile.sub_modq %rs1_etQ0_acc_268435009, %rs1_etQ0_negM_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ1 = tile.reshape %rs1_etQ0_accs_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_1991, %a_1992, %a_1993, %a_1994 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_1995, %b_1996, %b_1997, %b_1998 = tile.decompose %rs1_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_1999 = tile.matmul_modq %a_1991, %b_1995 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2000 = tile.matmul_modq %a_1991, %b_1996 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2001 = tile.matmul_modq %a_1991, %b_1997 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2002 = tile.matmul_modq %a_1991, %b_1998 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2003 = tile.matmul_modq %a_1992, %b_1995 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2004 = tile.matmul_modq %a_1992, %b_1996 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2005 = tile.matmul_modq %a_1992, %b_1997 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2006 = tile.matmul_modq %a_1992, %b_1998 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2007 = tile.matmul_modq %a_1993, %b_1995 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2008 = tile.matmul_modq %a_1993, %b_1996 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2009 = tile.matmul_modq %a_1993, %b_1997 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2010 = tile.matmul_modq %a_1993, %b_1998 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2011 = tile.matmul_modq %a_1994, %b_1995 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2012 = tile.matmul_modq %a_1994, %b_1996 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2013 = tile.matmul_modq %a_1994, %b_1997 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2014 = tile.matmul_modq %a_1994, %b_1998 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_2015 = tile.add_modq %P_1999 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2016 = tile.add_modq %P_2000, %P_2003 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2017 = tile.add_modq %P_2001, %P_2004, %P_2007 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2018 = tile.add_modq %P_2002, %P_2005, %P_2008, %P_2011 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2019 = tile.add_modq %P_2006, %P_2009, %P_2012 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2020 = tile.add_modq %P_2010, %P_2013 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2021 = tile.add_modq %P_2014 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs1_etQ0_xE_268434977 = tile.recombine_modq %C_2015, %C_2016, %C_2017, %C_2018, %C_2019, %C_2020, %C_2021 {radix = 128 : i64, q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_2022, %a_2023, %a_2024, %a_2025 = tile.decompose %fbcD__268435361__268435009__268434977__268433569 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_2026, %b_2027, %b_2028, %b_2029 = tile.decompose %rs2_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_2030 = tile.matmul_modq %a_2022, %b_2026 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2031 = tile.matmul_modq %a_2022, %b_2027 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2032 = tile.matmul_modq %a_2022, %b_2028 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2033 = tile.matmul_modq %a_2022, %b_2029 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2034 = tile.matmul_modq %a_2023, %b_2026 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2035 = tile.matmul_modq %a_2023, %b_2027 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2036 = tile.matmul_modq %a_2023, %b_2028 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2037 = tile.matmul_modq %a_2023, %b_2029 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2038 = tile.matmul_modq %a_2024, %b_2026 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2039 = tile.matmul_modq %a_2024, %b_2027 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2040 = tile.matmul_modq %a_2024, %b_2028 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2041 = tile.matmul_modq %a_2024, %b_2029 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2042 = tile.matmul_modq %a_2025, %b_2026 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2043 = tile.matmul_modq %a_2025, %b_2027 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2044 = tile.matmul_modq %a_2025, %b_2028 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2045 = tile.matmul_modq %a_2025, %b_2029 {q = 268433569 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_2046 = tile.add_modq %P_2030 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2047 = tile.add_modq %P_2031, %P_2034 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2048 = tile.add_modq %P_2032, %P_2035, %P_2038 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2049 = tile.add_modq %P_2033, %P_2036, %P_2039, %P_2042 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2050 = tile.add_modq %P_2037, %P_2040, %P_2043 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2051 = tile.add_modq %P_2041, %P_2044 {q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2052 = tile.add_modq %P_2045 {q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_fbc_268433569 = tile.recombine_modq %C_2046, %C_2047, %C_2048, %C_2049, %C_2050, %C_2051, %C_2052 {radix = 128 : i64, q = 268433569 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0 = tile.reshape %rs2_fbc0_fbc_268433569 {shape = [16], q = 268433569 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_2053, %a_2054, %a_2055, %a_2056 = tile.decompose %fbcD__268435361__268435009__268434977__268433441 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_2057, %b_2058, %b_2059, %b_2060 = tile.decompose %rs2_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_2061 = tile.matmul_modq %a_2053, %b_2057 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2062 = tile.matmul_modq %a_2053, %b_2058 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2063 = tile.matmul_modq %a_2053, %b_2059 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2064 = tile.matmul_modq %a_2053, %b_2060 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2065 = tile.matmul_modq %a_2054, %b_2057 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2066 = tile.matmul_modq %a_2054, %b_2058 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2067 = tile.matmul_modq %a_2054, %b_2059 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2068 = tile.matmul_modq %a_2054, %b_2060 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2069 = tile.matmul_modq %a_2055, %b_2057 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2070 = tile.matmul_modq %a_2055, %b_2058 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2071 = tile.matmul_modq %a_2055, %b_2059 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2072 = tile.matmul_modq %a_2055, %b_2060 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2073 = tile.matmul_modq %a_2056, %b_2057 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2074 = tile.matmul_modq %a_2056, %b_2058 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2075 = tile.matmul_modq %a_2056, %b_2059 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2076 = tile.matmul_modq %a_2056, %b_2060 {q = 268433441 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_2077 = tile.add_modq %P_2061 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2078 = tile.add_modq %P_2062, %P_2065 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2079 = tile.add_modq %P_2063, %P_2066, %P_2069 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2080 = tile.add_modq %P_2064, %P_2067, %P_2070, %P_2073 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2081 = tile.add_modq %P_2068, %P_2071, %P_2074 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2082 = tile.add_modq %P_2072, %P_2075 {q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2083 = tile.add_modq %P_2076 {q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_fbc_268433441 = tile.recombine_modq %C_2077, %C_2078, %C_2079, %C_2080, %C_2081, %C_2082, %C_2083 {radix = 128 : i64, q = 268433441 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc1 = tile.reshape %rs2_fbc0_fbc_268433441 {shape = [16], q = 268433441 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_2084, %a_2085, %a_2086, %a_2087 = tile.decompose %fbcD__268435361__268435009__268434977__268433377 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_2088, %b_2089, %b_2090, %b_2091 = tile.decompose %rs2_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_2092 = tile.matmul_modq %a_2084, %b_2088 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2093 = tile.matmul_modq %a_2084, %b_2089 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2094 = tile.matmul_modq %a_2084, %b_2090 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2095 = tile.matmul_modq %a_2084, %b_2091 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2096 = tile.matmul_modq %a_2085, %b_2088 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2097 = tile.matmul_modq %a_2085, %b_2089 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2098 = tile.matmul_modq %a_2085, %b_2090 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2099 = tile.matmul_modq %a_2085, %b_2091 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2100 = tile.matmul_modq %a_2086, %b_2088 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2101 = tile.matmul_modq %a_2086, %b_2089 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2102 = tile.matmul_modq %a_2086, %b_2090 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2103 = tile.matmul_modq %a_2086, %b_2091 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2104 = tile.matmul_modq %a_2087, %b_2088 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2105 = tile.matmul_modq %a_2087, %b_2089 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2106 = tile.matmul_modq %a_2087, %b_2090 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2107 = tile.matmul_modq %a_2087, %b_2091 {q = 268433377 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_2108 = tile.add_modq %P_2092 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2109 = tile.add_modq %P_2093, %P_2096 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2110 = tile.add_modq %P_2094, %P_2097, %P_2100 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2111 = tile.add_modq %P_2095, %P_2098, %P_2101, %P_2104 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2112 = tile.add_modq %P_2099, %P_2102, %P_2105 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2113 = tile.add_modq %P_2103, %P_2106 {q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2114 = tile.add_modq %P_2107 {q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_fbc_268433377 = tile.recombine_modq %C_2108, %C_2109, %C_2110, %C_2111, %C_2112, %C_2113, %C_2114 {radix = 128 : i64, q = 268433377 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc2 = tile.reshape %rs2_fbc0_fbc_268433377 {shape = [16], q = 268433377 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_2115, %a_2116, %a_2117, %a_2118 = tile.decompose %fbcD__268435361__268435009__268434977__268433057 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_2119, %b_2120, %b_2121, %b_2122 = tile.decompose %rs2_fbc0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_2123 = tile.matmul_modq %a_2115, %b_2119 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2124 = tile.matmul_modq %a_2115, %b_2120 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2125 = tile.matmul_modq %a_2115, %b_2121 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2126 = tile.matmul_modq %a_2115, %b_2122 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2127 = tile.matmul_modq %a_2116, %b_2119 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2128 = tile.matmul_modq %a_2116, %b_2120 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2129 = tile.matmul_modq %a_2116, %b_2121 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2130 = tile.matmul_modq %a_2116, %b_2122 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2131 = tile.matmul_modq %a_2117, %b_2119 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2132 = tile.matmul_modq %a_2117, %b_2120 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2133 = tile.matmul_modq %a_2117, %b_2121 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2134 = tile.matmul_modq %a_2117, %b_2122 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2135 = tile.matmul_modq %a_2118, %b_2119 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2136 = tile.matmul_modq %a_2118, %b_2120 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2137 = tile.matmul_modq %a_2118, %b_2121 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2138 = tile.matmul_modq %a_2118, %b_2122 {q = 268433057 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_2139 = tile.add_modq %P_2123 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2140 = tile.add_modq %P_2124, %P_2127 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2141 = tile.add_modq %P_2125, %P_2128, %P_2131 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2142 = tile.add_modq %P_2126, %P_2129, %P_2132, %P_2135 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2143 = tile.add_modq %P_2130, %P_2133, %P_2136 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2144 = tile.add_modq %P_2134, %P_2137 {q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2145 = tile.add_modq %P_2138 {q = 268433057 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_fbc0_fbc_268433057 = tile.recombine_modq %C_2139, %C_2140, %C_2141, %C_2142, %C_2143, %C_2144, %C_2145 {radix = 128 : i64, q = 268433057 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_2146, %a_2147, %a_2148, %a_2149, %a_2150, %a_2151 = tile.decompose %crnsF__268433569__268433441__268433377__268433057__38 {radix = 128 : i64, nlimbs = 6 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_2152, %b_2153, %b_2154, %b_2155 = tile.decompose %rs2_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %wP_2156 = tile.matmul_wide %a_2146, %b_2152 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2157 = tile.matmul_wide %a_2146, %b_2153 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2158 = tile.matmul_wide %a_2146, %b_2154 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2159 = tile.matmul_wide %a_2146, %b_2155 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2160 = tile.matmul_wide %a_2147, %b_2152 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2161 = tile.matmul_wide %a_2147, %b_2153 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2162 = tile.matmul_wide %a_2147, %b_2154 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2163 = tile.matmul_wide %a_2147, %b_2155 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2164 = tile.matmul_wide %a_2148, %b_2152 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2165 = tile.matmul_wide %a_2148, %b_2153 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2166 = tile.matmul_wide %a_2148, %b_2154 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2167 = tile.matmul_wide %a_2148, %b_2155 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2168 = tile.matmul_wide %a_2149, %b_2152 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2169 = tile.matmul_wide %a_2149, %b_2153 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2170 = tile.matmul_wide %a_2149, %b_2154 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2171 = tile.matmul_wide %a_2149, %b_2155 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2172 = tile.matmul_wide %a_2150, %b_2152 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2173 = tile.matmul_wide %a_2150, %b_2153 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2174 = tile.matmul_wide %a_2150, %b_2154 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2175 = tile.matmul_wide %a_2150, %b_2155 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2176 = tile.matmul_wide %a_2151, %b_2152 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2177 = tile.matmul_wide %a_2151, %b_2153 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2178 = tile.matmul_wide %a_2151, %b_2154 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %wP_2179 = tile.matmul_wide %a_2151, %b_2155 : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %rs2_etQ0_s = tile.recombine_wide %wP_2156, %wP_2157, %wP_2158, %wP_2159, %wP_2160, %wP_2161, %wP_2162, %wP_2163, %wP_2164, %wP_2165, %wP_2166, %wP_2167, %wP_2168, %wP_2169, %wP_2170, %wP_2171, %wP_2172, %wP_2173, %wP_2174, %wP_2175, %wP_2176, %wP_2177, %wP_2178, %wP_2179 {radix = 128 : i64, exps = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8]} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_k = tile.fixed_slice %rs2_etQ0_s {shift = 38 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_neg = tile.fixed_slice %rs2_etQ0_s {shift = 37 : i64, mask = 1 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %a_2180, %a_2181, %a_2182, %a_2183 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_2184, %b_2185, %b_2186, %b_2187 = tile.decompose %rs2_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_2188 = tile.matmul_modq %a_2180, %b_2184 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2189 = tile.matmul_modq %a_2180, %b_2185 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2190 = tile.matmul_modq %a_2180, %b_2186 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2191 = tile.matmul_modq %a_2180, %b_2187 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2192 = tile.matmul_modq %a_2181, %b_2184 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2193 = tile.matmul_modq %a_2181, %b_2185 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2194 = tile.matmul_modq %a_2181, %b_2186 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2195 = tile.matmul_modq %a_2181, %b_2187 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2196 = tile.matmul_modq %a_2182, %b_2184 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2197 = tile.matmul_modq %a_2182, %b_2185 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2198 = tile.matmul_modq %a_2182, %b_2186 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2199 = tile.matmul_modq %a_2182, %b_2187 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2200 = tile.matmul_modq %a_2183, %b_2184 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2201 = tile.matmul_modq %a_2183, %b_2185 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2202 = tile.matmul_modq %a_2183, %b_2186 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2203 = tile.matmul_modq %a_2183, %b_2187 {q = 268435361 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_2204 = tile.add_modq %P_2188 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2205 = tile.add_modq %P_2189, %P_2192 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2206 = tile.add_modq %P_2190, %P_2193, %P_2196 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2207 = tile.add_modq %P_2191, %P_2194, %P_2197, %P_2200 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2208 = tile.add_modq %P_2195, %P_2198, %P_2201 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2209 = tile.add_modq %P_2199, %P_2202 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2210 = tile.add_modq %P_2203 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_xE_268435361 = tile.recombine_modq %C_2204, %C_2205, %C_2206, %C_2207, %C_2208, %C_2209, %C_2210 {radix = 128 : i64, q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_kg_268435361 = tile.mul_modq %rs2_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_acc_268435361 = tile.add_modq %rs2_etQ0_xE_268435361, %rs2_etQ0_kg_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_negM_268435361 = tile.mul_modq %rs2_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_accs_268435361 = tile.sub_modq %rs2_etQ0_acc_268435361, %rs2_etQ0_negM_268435361 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0 = tile.reshape %rs2_etQ0_accs_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_2211, %a_2212, %a_2213, %a_2214 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_2215, %b_2216, %b_2217, %b_2218 = tile.decompose %rs2_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_2219 = tile.matmul_modq %a_2211, %b_2215 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2220 = tile.matmul_modq %a_2211, %b_2216 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2221 = tile.matmul_modq %a_2211, %b_2217 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2222 = tile.matmul_modq %a_2211, %b_2218 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2223 = tile.matmul_modq %a_2212, %b_2215 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2224 = tile.matmul_modq %a_2212, %b_2216 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2225 = tile.matmul_modq %a_2212, %b_2217 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2226 = tile.matmul_modq %a_2212, %b_2218 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2227 = tile.matmul_modq %a_2213, %b_2215 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2228 = tile.matmul_modq %a_2213, %b_2216 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2229 = tile.matmul_modq %a_2213, %b_2217 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2230 = tile.matmul_modq %a_2213, %b_2218 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2231 = tile.matmul_modq %a_2214, %b_2215 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2232 = tile.matmul_modq %a_2214, %b_2216 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2233 = tile.matmul_modq %a_2214, %b_2217 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2234 = tile.matmul_modq %a_2214, %b_2218 {q = 268435009 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_2235 = tile.add_modq %P_2219 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2236 = tile.add_modq %P_2220, %P_2223 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2237 = tile.add_modq %P_2221, %P_2224, %P_2227 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2238 = tile.add_modq %P_2222, %P_2225, %P_2228, %P_2231 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2239 = tile.add_modq %P_2226, %P_2229, %P_2232 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2240 = tile.add_modq %P_2230, %P_2233 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2241 = tile.add_modq %P_2234 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_xE_268435009 = tile.recombine_modq %C_2235, %C_2236, %C_2237, %C_2238, %C_2239, %C_2240, %C_2241 {radix = 128 : i64, q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_kg_268435009 = tile.mul_modq %rs2_etQ0_k, %crnsg__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_acc_268435009 = tile.add_modq %rs2_etQ0_xE_268435009, %rs2_etQ0_kg_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_negM_268435009 = tile.mul_modq %rs2_etQ0_neg, %crnsMmod__268433569__268433441__268433377__268433057__268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x1xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_accs_268435009 = tile.sub_modq %rs2_etQ0_acc_268435009, %rs2_etQ0_negM_268435009 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ1 = tile.reshape %rs2_etQ0_accs_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_2242, %a_2243, %a_2244, %a_2245 = tile.decompose %crnsE__268433569__268433441__268433377__268433057__268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x4xi32>) -> (tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>, tensor<1x4xi8>)
    %b_2246, %b_2247, %b_2248, %b_2249 = tile.decompose %rs2_etQ0_RES {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<4x16xi32>) -> (tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>, tensor<4x16xi8>)
    %P_2250 = tile.matmul_modq %a_2242, %b_2246 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2251 = tile.matmul_modq %a_2242, %b_2247 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2252 = tile.matmul_modq %a_2242, %b_2248 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2253 = tile.matmul_modq %a_2242, %b_2249 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2254 = tile.matmul_modq %a_2243, %b_2246 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2255 = tile.matmul_modq %a_2243, %b_2247 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2256 = tile.matmul_modq %a_2243, %b_2248 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2257 = tile.matmul_modq %a_2243, %b_2249 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2258 = tile.matmul_modq %a_2244, %b_2246 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2259 = tile.matmul_modq %a_2244, %b_2247 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2260 = tile.matmul_modq %a_2244, %b_2248 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2261 = tile.matmul_modq %a_2244, %b_2249 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2262 = tile.matmul_modq %a_2245, %b_2246 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2263 = tile.matmul_modq %a_2245, %b_2247 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2264 = tile.matmul_modq %a_2245, %b_2248 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %P_2265 = tile.matmul_modq %a_2245, %b_2249 {q = 268434977 : i64} : (tensor<1x4xi8>, tensor<4x16xi8>) -> tensor<1x16xi32>
    %C_2266 = tile.add_modq %P_2250 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2267 = tile.add_modq %P_2251, %P_2254 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2268 = tile.add_modq %P_2252, %P_2255, %P_2258 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2269 = tile.add_modq %P_2253, %P_2256, %P_2259, %P_2262 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2270 = tile.add_modq %P_2257, %P_2260, %P_2263 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2271 = tile.add_modq %P_2261, %P_2264 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2272 = tile.add_modq %P_2265 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rs2_etQ0_xE_268434977 = tile.recombine_modq %C_2266, %C_2267, %C_2268, %C_2269, %C_2270, %C_2271, %C_2272 {radix = 128 : i64, q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_2273, %a_2274, %a_2275, %a_2276 = tile.decompose %fbcD__268435361__268435009__268434977__268434721 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_2277, %b_2278, %b_2279, %b_2280 = tile.decompose %rl_c2P0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_2281 = tile.matmul_modq %a_2273, %b_2277 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2282 = tile.matmul_modq %a_2273, %b_2278 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2283 = tile.matmul_modq %a_2273, %b_2279 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2284 = tile.matmul_modq %a_2273, %b_2280 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2285 = tile.matmul_modq %a_2274, %b_2277 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2286 = tile.matmul_modq %a_2274, %b_2278 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2287 = tile.matmul_modq %a_2274, %b_2279 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2288 = tile.matmul_modq %a_2274, %b_2280 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2289 = tile.matmul_modq %a_2275, %b_2277 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2290 = tile.matmul_modq %a_2275, %b_2278 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2291 = tile.matmul_modq %a_2275, %b_2279 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2292 = tile.matmul_modq %a_2275, %b_2280 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2293 = tile.matmul_modq %a_2276, %b_2277 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2294 = tile.matmul_modq %a_2276, %b_2278 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2295 = tile.matmul_modq %a_2276, %b_2279 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2296 = tile.matmul_modq %a_2276, %b_2280 {q = 268434721 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_2297 = tile.add_modq %P_2281 {q = 268434721 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2298 = tile.add_modq %P_2282, %P_2285 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2299 = tile.add_modq %P_2283, %P_2286, %P_2289 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2300 = tile.add_modq %P_2284, %P_2287, %P_2290, %P_2293 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2301 = tile.add_modq %P_2288, %P_2291, %P_2294 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2302 = tile.add_modq %P_2292, %P_2295 {q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2303 = tile.add_modq %P_2296 {q = 268434721 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P0_fbc_268434721 = tile.recombine_modq %C_2297, %C_2298, %C_2299, %C_2300, %C_2301, %C_2302, %C_2303 {radix = 128 : i64, q = 268434721 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P0 = tile.reshape %rl_c2P0_fbc_268434721 {shape = [16], q = 268434721 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268434401 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268434401"} : tensor<1x3xi32>
    %a_2304, %a_2305, %a_2306, %a_2307 = tile.decompose %fbcD__268435361__268435009__268434977__268434401 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_2308, %b_2309, %b_2310, %b_2311 = tile.decompose %rl_c2P0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_2312 = tile.matmul_modq %a_2304, %b_2308 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2313 = tile.matmul_modq %a_2304, %b_2309 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2314 = tile.matmul_modq %a_2304, %b_2310 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2315 = tile.matmul_modq %a_2304, %b_2311 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2316 = tile.matmul_modq %a_2305, %b_2308 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2317 = tile.matmul_modq %a_2305, %b_2309 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2318 = tile.matmul_modq %a_2305, %b_2310 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2319 = tile.matmul_modq %a_2305, %b_2311 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2320 = tile.matmul_modq %a_2306, %b_2308 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2321 = tile.matmul_modq %a_2306, %b_2309 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2322 = tile.matmul_modq %a_2306, %b_2310 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2323 = tile.matmul_modq %a_2306, %b_2311 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2324 = tile.matmul_modq %a_2307, %b_2308 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2325 = tile.matmul_modq %a_2307, %b_2309 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2326 = tile.matmul_modq %a_2307, %b_2310 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2327 = tile.matmul_modq %a_2307, %b_2311 {q = 268434401 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_2328 = tile.add_modq %P_2312 {q = 268434401 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2329 = tile.add_modq %P_2313, %P_2316 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2330 = tile.add_modq %P_2314, %P_2317, %P_2320 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2331 = tile.add_modq %P_2315, %P_2318, %P_2321, %P_2324 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2332 = tile.add_modq %P_2319, %P_2322, %P_2325 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2333 = tile.add_modq %P_2323, %P_2326 {q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2334 = tile.add_modq %P_2327 {q = 268434401 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P0_fbc_268434401 = tile.recombine_modq %C_2328, %C_2329, %C_2330, %C_2331, %C_2332, %C_2333, %C_2334 {radix = 128 : i64, q = 268434401 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P1 = tile.reshape %rl_c2P0_fbc_268434401 {shape = [16], q = 268434401 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268435361__268435009__268434977__268433761 = tile.const {name = "fbcD_(268435361, 268435009, 268434977)_268433761"} : tensor<1x3xi32>
    %a_2335, %a_2336, %a_2337, %a_2338 = tile.decompose %fbcD__268435361__268435009__268434977__268433761 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_2339, %b_2340, %b_2341, %b_2342 = tile.decompose %rl_c2P0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_2343 = tile.matmul_modq %a_2335, %b_2339 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2344 = tile.matmul_modq %a_2335, %b_2340 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2345 = tile.matmul_modq %a_2335, %b_2341 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2346 = tile.matmul_modq %a_2335, %b_2342 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2347 = tile.matmul_modq %a_2336, %b_2339 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2348 = tile.matmul_modq %a_2336, %b_2340 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2349 = tile.matmul_modq %a_2336, %b_2341 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2350 = tile.matmul_modq %a_2336, %b_2342 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2351 = tile.matmul_modq %a_2337, %b_2339 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2352 = tile.matmul_modq %a_2337, %b_2340 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2353 = tile.matmul_modq %a_2337, %b_2341 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2354 = tile.matmul_modq %a_2337, %b_2342 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2355 = tile.matmul_modq %a_2338, %b_2339 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2356 = tile.matmul_modq %a_2338, %b_2340 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2357 = tile.matmul_modq %a_2338, %b_2341 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_2358 = tile.matmul_modq %a_2338, %b_2342 {q = 268433761 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_2359 = tile.add_modq %P_2343 {q = 268433761 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2360 = tile.add_modq %P_2344, %P_2347 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2361 = tile.add_modq %P_2345, %P_2348, %P_2351 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2362 = tile.add_modq %P_2346, %P_2349, %P_2352, %P_2355 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2363 = tile.add_modq %P_2350, %P_2353, %P_2356 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2364 = tile.add_modq %P_2354, %P_2357 {q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_2365 = tile.add_modq %P_2358 {q = 268433761 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P0_fbc_268433761 = tile.recombine_modq %C_2359, %C_2360, %C_2361, %C_2362, %C_2363, %C_2364, %C_2365 {radix = 128 : i64, q = 268433761 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_c2P2 = tile.reshape %rl_c2P0_fbc_268433761 {shape = [16], q = 268433761 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_2366, %a_2367, %a_2368, %a_2369 = tile.decompose %W_f_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2370, %b_2371, %b_2372, %b_2373 = tile.decompose %rs2_etQ0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2374 = tile.matmul_modq %a_2366, %b_2370 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2375 = tile.matmul_modq %a_2366, %b_2371 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2376 = tile.matmul_modq %a_2366, %b_2372 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2377 = tile.matmul_modq %a_2366, %b_2373 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2378 = tile.matmul_modq %a_2367, %b_2370 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2379 = tile.matmul_modq %a_2367, %b_2371 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2380 = tile.matmul_modq %a_2367, %b_2372 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2381 = tile.matmul_modq %a_2367, %b_2373 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2382 = tile.matmul_modq %a_2368, %b_2370 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2383 = tile.matmul_modq %a_2368, %b_2371 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2384 = tile.matmul_modq %a_2368, %b_2372 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2385 = tile.matmul_modq %a_2368, %b_2373 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2386 = tile.matmul_modq %a_2369, %b_2370 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2387 = tile.matmul_modq %a_2369, %b_2371 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2388 = tile.matmul_modq %a_2369, %b_2372 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2389 = tile.matmul_modq %a_2369, %b_2373 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2390 = tile.add_modq %P_2374 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2391 = tile.add_modq %P_2375, %P_2378 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2392 = tile.add_modq %P_2376, %P_2379, %P_2382 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2393 = tile.add_modq %P_2377, %P_2380, %P_2383, %P_2386 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2394 = tile.add_modq %P_2381, %P_2384, %P_2387 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2395 = tile.add_modq %P_2385, %P_2388 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2396 = tile.add_modq %P_2389 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X0 = tile.recombine_modq %C_2390, %C_2391, %C_2392, %C_2393, %C_2394, %C_2395, %C_2396 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_2397, %a_2398, %a_2399, %a_2400 = tile.decompose %W_f_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2401, %b_2402, %b_2403, %b_2404 = tile.decompose %rs2_etQ1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2405 = tile.matmul_modq %a_2397, %b_2401 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2406 = tile.matmul_modq %a_2397, %b_2402 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2407 = tile.matmul_modq %a_2397, %b_2403 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2408 = tile.matmul_modq %a_2397, %b_2404 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2409 = tile.matmul_modq %a_2398, %b_2401 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2410 = tile.matmul_modq %a_2398, %b_2402 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2411 = tile.matmul_modq %a_2398, %b_2403 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2412 = tile.matmul_modq %a_2398, %b_2404 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2413 = tile.matmul_modq %a_2399, %b_2401 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2414 = tile.matmul_modq %a_2399, %b_2402 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2415 = tile.matmul_modq %a_2399, %b_2403 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2416 = tile.matmul_modq %a_2399, %b_2404 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2417 = tile.matmul_modq %a_2400, %b_2401 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2418 = tile.matmul_modq %a_2400, %b_2402 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2419 = tile.matmul_modq %a_2400, %b_2403 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2420 = tile.matmul_modq %a_2400, %b_2404 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2421 = tile.add_modq %P_2405 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2422 = tile.add_modq %P_2406, %P_2409 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2423 = tile.add_modq %P_2407, %P_2410, %P_2413 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2424 = tile.add_modq %P_2408, %P_2411, %P_2414, %P_2417 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2425 = tile.add_modq %P_2412, %P_2415, %P_2418 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2426 = tile.add_modq %P_2416, %P_2419 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2427 = tile.add_modq %P_2420 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X1 = tile.recombine_modq %C_2421, %C_2422, %C_2423, %C_2424, %C_2425, %C_2426, %C_2427 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_2428, %a_2429, %a_2430, %a_2431 = tile.decompose %W_f_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2432, %b_2433, %b_2434, %b_2435 = tile.decompose %rs2_etQ2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2436 = tile.matmul_modq %a_2428, %b_2432 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2437 = tile.matmul_modq %a_2428, %b_2433 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2438 = tile.matmul_modq %a_2428, %b_2434 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2439 = tile.matmul_modq %a_2428, %b_2435 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2440 = tile.matmul_modq %a_2429, %b_2432 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2441 = tile.matmul_modq %a_2429, %b_2433 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2442 = tile.matmul_modq %a_2429, %b_2434 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2443 = tile.matmul_modq %a_2429, %b_2435 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2444 = tile.matmul_modq %a_2430, %b_2432 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2445 = tile.matmul_modq %a_2430, %b_2433 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2446 = tile.matmul_modq %a_2430, %b_2434 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2447 = tile.matmul_modq %a_2430, %b_2435 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2448 = tile.matmul_modq %a_2431, %b_2432 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2449 = tile.matmul_modq %a_2431, %b_2433 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2450 = tile.matmul_modq %a_2431, %b_2434 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2451 = tile.matmul_modq %a_2431, %b_2435 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2452 = tile.add_modq %P_2436 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2453 = tile.add_modq %P_2437, %P_2440 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2454 = tile.add_modq %P_2438, %P_2441, %P_2444 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2455 = tile.add_modq %P_2439, %P_2442, %P_2445, %P_2448 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2456 = tile.add_modq %P_2443, %P_2446, %P_2449 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2457 = tile.add_modq %P_2447, %P_2450 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2458 = tile.add_modq %P_2451 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X2 = tile.recombine_modq %C_2452, %C_2453, %C_2454, %C_2455, %C_2456, %C_2457, %C_2458 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268434721 = tile.const {name = "W_f_268434721", q = 268434721 : i64} : tensor<16x16xi32>
    %a_2459, %a_2460, %a_2461, %a_2462 = tile.decompose %W_f_268434721 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2463, %b_2464, %b_2465, %b_2466 = tile.decompose %rl_c2P0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2467 = tile.matmul_modq %a_2459, %b_2463 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2468 = tile.matmul_modq %a_2459, %b_2464 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2469 = tile.matmul_modq %a_2459, %b_2465 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2470 = tile.matmul_modq %a_2459, %b_2466 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2471 = tile.matmul_modq %a_2460, %b_2463 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2472 = tile.matmul_modq %a_2460, %b_2464 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2473 = tile.matmul_modq %a_2460, %b_2465 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2474 = tile.matmul_modq %a_2460, %b_2466 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2475 = tile.matmul_modq %a_2461, %b_2463 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2476 = tile.matmul_modq %a_2461, %b_2464 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2477 = tile.matmul_modq %a_2461, %b_2465 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2478 = tile.matmul_modq %a_2461, %b_2466 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2479 = tile.matmul_modq %a_2462, %b_2463 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2480 = tile.matmul_modq %a_2462, %b_2464 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2481 = tile.matmul_modq %a_2462, %b_2465 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2482 = tile.matmul_modq %a_2462, %b_2466 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2483 = tile.add_modq %P_2467 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2484 = tile.add_modq %P_2468, %P_2471 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2485 = tile.add_modq %P_2469, %P_2472, %P_2475 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2486 = tile.add_modq %P_2470, %P_2473, %P_2476, %P_2479 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2487 = tile.add_modq %P_2474, %P_2477, %P_2480 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2488 = tile.add_modq %P_2478, %P_2481 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2489 = tile.add_modq %P_2482 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X3 = tile.recombine_modq %C_2483, %C_2484, %C_2485, %C_2486, %C_2487, %C_2488, %C_2489 {radix = 128 : i64, q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268434401 = tile.const {name = "W_f_268434401", q = 268434401 : i64} : tensor<16x16xi32>
    %a_2490, %a_2491, %a_2492, %a_2493 = tile.decompose %W_f_268434401 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2494, %b_2495, %b_2496, %b_2497 = tile.decompose %rl_c2P1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2498 = tile.matmul_modq %a_2490, %b_2494 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2499 = tile.matmul_modq %a_2490, %b_2495 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2500 = tile.matmul_modq %a_2490, %b_2496 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2501 = tile.matmul_modq %a_2490, %b_2497 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2502 = tile.matmul_modq %a_2491, %b_2494 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2503 = tile.matmul_modq %a_2491, %b_2495 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2504 = tile.matmul_modq %a_2491, %b_2496 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2505 = tile.matmul_modq %a_2491, %b_2497 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2506 = tile.matmul_modq %a_2492, %b_2494 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2507 = tile.matmul_modq %a_2492, %b_2495 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2508 = tile.matmul_modq %a_2492, %b_2496 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2509 = tile.matmul_modq %a_2492, %b_2497 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2510 = tile.matmul_modq %a_2493, %b_2494 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2511 = tile.matmul_modq %a_2493, %b_2495 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2512 = tile.matmul_modq %a_2493, %b_2496 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2513 = tile.matmul_modq %a_2493, %b_2497 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2514 = tile.add_modq %P_2498 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2515 = tile.add_modq %P_2499, %P_2502 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2516 = tile.add_modq %P_2500, %P_2503, %P_2506 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2517 = tile.add_modq %P_2501, %P_2504, %P_2507, %P_2510 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2518 = tile.add_modq %P_2505, %P_2508, %P_2511 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2519 = tile.add_modq %P_2509, %P_2512 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2520 = tile.add_modq %P_2513 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X4 = tile.recombine_modq %C_2514, %C_2515, %C_2516, %C_2517, %C_2518, %C_2519, %C_2520 {radix = 128 : i64, q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_f_268433761 = tile.const {name = "W_f_268433761", q = 268433761 : i64} : tensor<16x16xi32>
    %a_2521, %a_2522, %a_2523, %a_2524 = tile.decompose %W_f_268433761 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2525, %b_2526, %b_2527, %b_2528 = tile.decompose %rl_c2P2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2529 = tile.matmul_modq %a_2521, %b_2525 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2530 = tile.matmul_modq %a_2521, %b_2526 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2531 = tile.matmul_modq %a_2521, %b_2527 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2532 = tile.matmul_modq %a_2521, %b_2528 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2533 = tile.matmul_modq %a_2522, %b_2525 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2534 = tile.matmul_modq %a_2522, %b_2526 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2535 = tile.matmul_modq %a_2522, %b_2527 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2536 = tile.matmul_modq %a_2522, %b_2528 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2537 = tile.matmul_modq %a_2523, %b_2525 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2538 = tile.matmul_modq %a_2523, %b_2526 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2539 = tile.matmul_modq %a_2523, %b_2527 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2540 = tile.matmul_modq %a_2523, %b_2528 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2541 = tile.matmul_modq %a_2524, %b_2525 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2542 = tile.matmul_modq %a_2524, %b_2526 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2543 = tile.matmul_modq %a_2524, %b_2527 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2544 = tile.matmul_modq %a_2524, %b_2528 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2545 = tile.add_modq %P_2529 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2546 = tile.add_modq %P_2530, %P_2533 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2547 = tile.add_modq %P_2531, %P_2534, %P_2537 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2548 = tile.add_modq %P_2532, %P_2535, %P_2538, %P_2541 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2549 = tile.add_modq %P_2536, %P_2539, %P_2542 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2550 = tile.add_modq %P_2540, %P_2543 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2551 = tile.add_modq %P_2544 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_X5 = tile.recombine_modq %C_2545, %C_2546, %C_2547, %C_2548, %C_2549, %C_2550, %C_2551 {radix = 128 : i64, q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_0c = rns.const {name = "rl_evkA_0c", q = 268435361 : i64} : tensor<16xi32>
    %a_2552, %a_2553, %a_2554, %a_2555 = tile.decompose %W_f_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2556, %b_2557, %b_2558, %b_2559 = tile.decompose %rl_evkA_0c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2560 = tile.matmul_modq %a_2552, %b_2556 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2561 = tile.matmul_modq %a_2552, %b_2557 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2562 = tile.matmul_modq %a_2552, %b_2558 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2563 = tile.matmul_modq %a_2552, %b_2559 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2564 = tile.matmul_modq %a_2553, %b_2556 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2565 = tile.matmul_modq %a_2553, %b_2557 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2566 = tile.matmul_modq %a_2553, %b_2558 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2567 = tile.matmul_modq %a_2553, %b_2559 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2568 = tile.matmul_modq %a_2554, %b_2556 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2569 = tile.matmul_modq %a_2554, %b_2557 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2570 = tile.matmul_modq %a_2554, %b_2558 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2571 = tile.matmul_modq %a_2554, %b_2559 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2572 = tile.matmul_modq %a_2555, %b_2556 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2573 = tile.matmul_modq %a_2555, %b_2557 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2574 = tile.matmul_modq %a_2555, %b_2558 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2575 = tile.matmul_modq %a_2555, %b_2559 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2576 = tile.add_modq %P_2560 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2577 = tile.add_modq %P_2561, %P_2564 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2578 = tile.add_modq %P_2562, %P_2565, %P_2568 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2579 = tile.add_modq %P_2563, %P_2566, %P_2569, %P_2572 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2580 = tile.add_modq %P_2567, %P_2570, %P_2573 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2581 = tile.add_modq %P_2571, %P_2574 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2582 = tile.add_modq %P_2575 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_0 = tile.recombine_modq %C_2576, %C_2577, %C_2578, %C_2579, %C_2580, %C_2581, %C_2582 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY0 = tile.mul_modq %rl_X0, %rl_evkA_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_2583, %a_2584, %a_2585, %a_2586 = tile.decompose %W_i_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2587, %b_2588, %b_2589, %b_2590 = tile.decompose %rl_AY0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2591 = tile.matmul_modq %a_2583, %b_2587 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2592 = tile.matmul_modq %a_2583, %b_2588 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2593 = tile.matmul_modq %a_2583, %b_2589 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2594 = tile.matmul_modq %a_2583, %b_2590 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2595 = tile.matmul_modq %a_2584, %b_2587 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2596 = tile.matmul_modq %a_2584, %b_2588 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2597 = tile.matmul_modq %a_2584, %b_2589 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2598 = tile.matmul_modq %a_2584, %b_2590 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2599 = tile.matmul_modq %a_2585, %b_2587 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2600 = tile.matmul_modq %a_2585, %b_2588 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2601 = tile.matmul_modq %a_2585, %b_2589 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2602 = tile.matmul_modq %a_2585, %b_2590 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2603 = tile.matmul_modq %a_2586, %b_2587 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2604 = tile.matmul_modq %a_2586, %b_2588 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2605 = tile.matmul_modq %a_2586, %b_2589 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2606 = tile.matmul_modq %a_2586, %b_2590 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2607 = tile.add_modq %P_2591 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2608 = tile.add_modq %P_2592, %P_2595 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2609 = tile.add_modq %P_2593, %P_2596, %P_2599 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2610 = tile.add_modq %P_2594, %P_2597, %P_2600, %P_2603 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2611 = tile.add_modq %P_2598, %P_2601, %P_2604 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2612 = tile.add_modq %P_2602, %P_2605 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2613 = tile.add_modq %P_2606 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa0 = tile.recombine_modq %C_2607, %C_2608, %C_2609, %C_2610, %C_2611, %C_2612, %C_2613 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_1c = rns.const {name = "rl_evkA_1c", q = 268435009 : i64} : tensor<16xi32>
    %a_2614, %a_2615, %a_2616, %a_2617 = tile.decompose %W_f_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2618, %b_2619, %b_2620, %b_2621 = tile.decompose %rl_evkA_1c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2622 = tile.matmul_modq %a_2614, %b_2618 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2623 = tile.matmul_modq %a_2614, %b_2619 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2624 = tile.matmul_modq %a_2614, %b_2620 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2625 = tile.matmul_modq %a_2614, %b_2621 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2626 = tile.matmul_modq %a_2615, %b_2618 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2627 = tile.matmul_modq %a_2615, %b_2619 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2628 = tile.matmul_modq %a_2615, %b_2620 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2629 = tile.matmul_modq %a_2615, %b_2621 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2630 = tile.matmul_modq %a_2616, %b_2618 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2631 = tile.matmul_modq %a_2616, %b_2619 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2632 = tile.matmul_modq %a_2616, %b_2620 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2633 = tile.matmul_modq %a_2616, %b_2621 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2634 = tile.matmul_modq %a_2617, %b_2618 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2635 = tile.matmul_modq %a_2617, %b_2619 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2636 = tile.matmul_modq %a_2617, %b_2620 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2637 = tile.matmul_modq %a_2617, %b_2621 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2638 = tile.add_modq %P_2622 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2639 = tile.add_modq %P_2623, %P_2626 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2640 = tile.add_modq %P_2624, %P_2627, %P_2630 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2641 = tile.add_modq %P_2625, %P_2628, %P_2631, %P_2634 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2642 = tile.add_modq %P_2629, %P_2632, %P_2635 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2643 = tile.add_modq %P_2633, %P_2636 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2644 = tile.add_modq %P_2637 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_1 = tile.recombine_modq %C_2638, %C_2639, %C_2640, %C_2641, %C_2642, %C_2643, %C_2644 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY1 = tile.mul_modq %rl_X1, %rl_evkA_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_2645, %a_2646, %a_2647, %a_2648 = tile.decompose %W_i_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2649, %b_2650, %b_2651, %b_2652 = tile.decompose %rl_AY1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2653 = tile.matmul_modq %a_2645, %b_2649 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2654 = tile.matmul_modq %a_2645, %b_2650 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2655 = tile.matmul_modq %a_2645, %b_2651 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2656 = tile.matmul_modq %a_2645, %b_2652 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2657 = tile.matmul_modq %a_2646, %b_2649 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2658 = tile.matmul_modq %a_2646, %b_2650 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2659 = tile.matmul_modq %a_2646, %b_2651 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2660 = tile.matmul_modq %a_2646, %b_2652 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2661 = tile.matmul_modq %a_2647, %b_2649 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2662 = tile.matmul_modq %a_2647, %b_2650 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2663 = tile.matmul_modq %a_2647, %b_2651 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2664 = tile.matmul_modq %a_2647, %b_2652 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2665 = tile.matmul_modq %a_2648, %b_2649 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2666 = tile.matmul_modq %a_2648, %b_2650 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2667 = tile.matmul_modq %a_2648, %b_2651 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2668 = tile.matmul_modq %a_2648, %b_2652 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2669 = tile.add_modq %P_2653 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2670 = tile.add_modq %P_2654, %P_2657 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2671 = tile.add_modq %P_2655, %P_2658, %P_2661 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2672 = tile.add_modq %P_2656, %P_2659, %P_2662, %P_2665 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2673 = tile.add_modq %P_2660, %P_2663, %P_2666 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2674 = tile.add_modq %P_2664, %P_2667 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2675 = tile.add_modq %P_2668 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa1 = tile.recombine_modq %C_2669, %C_2670, %C_2671, %C_2672, %C_2673, %C_2674, %C_2675 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_2c = rns.const {name = "rl_evkA_2c", q = 268434977 : i64} : tensor<16xi32>
    %a_2676, %a_2677, %a_2678, %a_2679 = tile.decompose %W_f_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2680, %b_2681, %b_2682, %b_2683 = tile.decompose %rl_evkA_2c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2684 = tile.matmul_modq %a_2676, %b_2680 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2685 = tile.matmul_modq %a_2676, %b_2681 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2686 = tile.matmul_modq %a_2676, %b_2682 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2687 = tile.matmul_modq %a_2676, %b_2683 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2688 = tile.matmul_modq %a_2677, %b_2680 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2689 = tile.matmul_modq %a_2677, %b_2681 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2690 = tile.matmul_modq %a_2677, %b_2682 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2691 = tile.matmul_modq %a_2677, %b_2683 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2692 = tile.matmul_modq %a_2678, %b_2680 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2693 = tile.matmul_modq %a_2678, %b_2681 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2694 = tile.matmul_modq %a_2678, %b_2682 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2695 = tile.matmul_modq %a_2678, %b_2683 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2696 = tile.matmul_modq %a_2679, %b_2680 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2697 = tile.matmul_modq %a_2679, %b_2681 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2698 = tile.matmul_modq %a_2679, %b_2682 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2699 = tile.matmul_modq %a_2679, %b_2683 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2700 = tile.add_modq %P_2684 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2701 = tile.add_modq %P_2685, %P_2688 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2702 = tile.add_modq %P_2686, %P_2689, %P_2692 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2703 = tile.add_modq %P_2687, %P_2690, %P_2693, %P_2696 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2704 = tile.add_modq %P_2691, %P_2694, %P_2697 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2705 = tile.add_modq %P_2695, %P_2698 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2706 = tile.add_modq %P_2699 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_2 = tile.recombine_modq %C_2700, %C_2701, %C_2702, %C_2703, %C_2704, %C_2705, %C_2706 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY2 = tile.mul_modq %rl_X2, %rl_evkA_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_2707, %a_2708, %a_2709, %a_2710 = tile.decompose %W_i_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2711, %b_2712, %b_2713, %b_2714 = tile.decompose %rl_AY2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2715 = tile.matmul_modq %a_2707, %b_2711 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2716 = tile.matmul_modq %a_2707, %b_2712 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2717 = tile.matmul_modq %a_2707, %b_2713 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2718 = tile.matmul_modq %a_2707, %b_2714 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2719 = tile.matmul_modq %a_2708, %b_2711 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2720 = tile.matmul_modq %a_2708, %b_2712 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2721 = tile.matmul_modq %a_2708, %b_2713 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2722 = tile.matmul_modq %a_2708, %b_2714 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2723 = tile.matmul_modq %a_2709, %b_2711 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2724 = tile.matmul_modq %a_2709, %b_2712 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2725 = tile.matmul_modq %a_2709, %b_2713 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2726 = tile.matmul_modq %a_2709, %b_2714 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2727 = tile.matmul_modq %a_2710, %b_2711 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2728 = tile.matmul_modq %a_2710, %b_2712 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2729 = tile.matmul_modq %a_2710, %b_2713 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2730 = tile.matmul_modq %a_2710, %b_2714 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2731 = tile.add_modq %P_2715 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2732 = tile.add_modq %P_2716, %P_2719 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2733 = tile.add_modq %P_2717, %P_2720, %P_2723 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2734 = tile.add_modq %P_2718, %P_2721, %P_2724, %P_2727 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2735 = tile.add_modq %P_2722, %P_2725, %P_2728 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2736 = tile.add_modq %P_2726, %P_2729 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2737 = tile.add_modq %P_2730 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa2 = tile.recombine_modq %C_2731, %C_2732, %C_2733, %C_2734, %C_2735, %C_2736, %C_2737 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_3c = rns.const {name = "rl_evkA_3c", q = 268434721 : i64} : tensor<16xi32>
    %a_2738, %a_2739, %a_2740, %a_2741 = tile.decompose %W_f_268434721 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2742, %b_2743, %b_2744, %b_2745 = tile.decompose %rl_evkA_3c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2746 = tile.matmul_modq %a_2738, %b_2742 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2747 = tile.matmul_modq %a_2738, %b_2743 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2748 = tile.matmul_modq %a_2738, %b_2744 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2749 = tile.matmul_modq %a_2738, %b_2745 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2750 = tile.matmul_modq %a_2739, %b_2742 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2751 = tile.matmul_modq %a_2739, %b_2743 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2752 = tile.matmul_modq %a_2739, %b_2744 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2753 = tile.matmul_modq %a_2739, %b_2745 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2754 = tile.matmul_modq %a_2740, %b_2742 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2755 = tile.matmul_modq %a_2740, %b_2743 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2756 = tile.matmul_modq %a_2740, %b_2744 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2757 = tile.matmul_modq %a_2740, %b_2745 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2758 = tile.matmul_modq %a_2741, %b_2742 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2759 = tile.matmul_modq %a_2741, %b_2743 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2760 = tile.matmul_modq %a_2741, %b_2744 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2761 = tile.matmul_modq %a_2741, %b_2745 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2762 = tile.add_modq %P_2746 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2763 = tile.add_modq %P_2747, %P_2750 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2764 = tile.add_modq %P_2748, %P_2751, %P_2754 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2765 = tile.add_modq %P_2749, %P_2752, %P_2755, %P_2758 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2766 = tile.add_modq %P_2753, %P_2756, %P_2759 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2767 = tile.add_modq %P_2757, %P_2760 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2768 = tile.add_modq %P_2761 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_3 = tile.recombine_modq %C_2762, %C_2763, %C_2764, %C_2765, %C_2766, %C_2767, %C_2768 {radix = 128 : i64, q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY3 = tile.mul_modq %rl_X3, %rl_evkA_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268434721 = tile.const {name = "W_i_268434721", q = 268434721 : i64} : tensor<16x16xi32>
    %a_2769, %a_2770, %a_2771, %a_2772 = tile.decompose %W_i_268434721 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2773, %b_2774, %b_2775, %b_2776 = tile.decompose %rl_AY3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2777 = tile.matmul_modq %a_2769, %b_2773 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2778 = tile.matmul_modq %a_2769, %b_2774 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2779 = tile.matmul_modq %a_2769, %b_2775 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2780 = tile.matmul_modq %a_2769, %b_2776 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2781 = tile.matmul_modq %a_2770, %b_2773 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2782 = tile.matmul_modq %a_2770, %b_2774 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2783 = tile.matmul_modq %a_2770, %b_2775 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2784 = tile.matmul_modq %a_2770, %b_2776 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2785 = tile.matmul_modq %a_2771, %b_2773 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2786 = tile.matmul_modq %a_2771, %b_2774 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2787 = tile.matmul_modq %a_2771, %b_2775 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2788 = tile.matmul_modq %a_2771, %b_2776 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2789 = tile.matmul_modq %a_2772, %b_2773 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2790 = tile.matmul_modq %a_2772, %b_2774 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2791 = tile.matmul_modq %a_2772, %b_2775 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2792 = tile.matmul_modq %a_2772, %b_2776 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2793 = tile.add_modq %P_2777 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2794 = tile.add_modq %P_2778, %P_2781 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2795 = tile.add_modq %P_2779, %P_2782, %P_2785 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2796 = tile.add_modq %P_2780, %P_2783, %P_2786, %P_2789 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2797 = tile.add_modq %P_2784, %P_2787, %P_2790 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2798 = tile.add_modq %P_2788, %P_2791 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2799 = tile.add_modq %P_2792 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa3 = tile.recombine_modq %C_2793, %C_2794, %C_2795, %C_2796, %C_2797, %C_2798, %C_2799 {radix = 128 : i64, q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_4c = rns.const {name = "rl_evkA_4c", q = 268434401 : i64} : tensor<16xi32>
    %a_2800, %a_2801, %a_2802, %a_2803 = tile.decompose %W_f_268434401 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2804, %b_2805, %b_2806, %b_2807 = tile.decompose %rl_evkA_4c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2808 = tile.matmul_modq %a_2800, %b_2804 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2809 = tile.matmul_modq %a_2800, %b_2805 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2810 = tile.matmul_modq %a_2800, %b_2806 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2811 = tile.matmul_modq %a_2800, %b_2807 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2812 = tile.matmul_modq %a_2801, %b_2804 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2813 = tile.matmul_modq %a_2801, %b_2805 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2814 = tile.matmul_modq %a_2801, %b_2806 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2815 = tile.matmul_modq %a_2801, %b_2807 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2816 = tile.matmul_modq %a_2802, %b_2804 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2817 = tile.matmul_modq %a_2802, %b_2805 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2818 = tile.matmul_modq %a_2802, %b_2806 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2819 = tile.matmul_modq %a_2802, %b_2807 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2820 = tile.matmul_modq %a_2803, %b_2804 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2821 = tile.matmul_modq %a_2803, %b_2805 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2822 = tile.matmul_modq %a_2803, %b_2806 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2823 = tile.matmul_modq %a_2803, %b_2807 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2824 = tile.add_modq %P_2808 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2825 = tile.add_modq %P_2809, %P_2812 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2826 = tile.add_modq %P_2810, %P_2813, %P_2816 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2827 = tile.add_modq %P_2811, %P_2814, %P_2817, %P_2820 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2828 = tile.add_modq %P_2815, %P_2818, %P_2821 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2829 = tile.add_modq %P_2819, %P_2822 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2830 = tile.add_modq %P_2823 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_4 = tile.recombine_modq %C_2824, %C_2825, %C_2826, %C_2827, %C_2828, %C_2829, %C_2830 {radix = 128 : i64, q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY4 = tile.mul_modq %rl_X4, %rl_evkA_4 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268434401 = tile.const {name = "W_i_268434401", q = 268434401 : i64} : tensor<16x16xi32>
    %a_2831, %a_2832, %a_2833, %a_2834 = tile.decompose %W_i_268434401 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2835, %b_2836, %b_2837, %b_2838 = tile.decompose %rl_AY4 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2839 = tile.matmul_modq %a_2831, %b_2835 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2840 = tile.matmul_modq %a_2831, %b_2836 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2841 = tile.matmul_modq %a_2831, %b_2837 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2842 = tile.matmul_modq %a_2831, %b_2838 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2843 = tile.matmul_modq %a_2832, %b_2835 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2844 = tile.matmul_modq %a_2832, %b_2836 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2845 = tile.matmul_modq %a_2832, %b_2837 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2846 = tile.matmul_modq %a_2832, %b_2838 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2847 = tile.matmul_modq %a_2833, %b_2835 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2848 = tile.matmul_modq %a_2833, %b_2836 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2849 = tile.matmul_modq %a_2833, %b_2837 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2850 = tile.matmul_modq %a_2833, %b_2838 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2851 = tile.matmul_modq %a_2834, %b_2835 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2852 = tile.matmul_modq %a_2834, %b_2836 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2853 = tile.matmul_modq %a_2834, %b_2837 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2854 = tile.matmul_modq %a_2834, %b_2838 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2855 = tile.add_modq %P_2839 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2856 = tile.add_modq %P_2840, %P_2843 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2857 = tile.add_modq %P_2841, %P_2844, %P_2847 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2858 = tile.add_modq %P_2842, %P_2845, %P_2848, %P_2851 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2859 = tile.add_modq %P_2846, %P_2849, %P_2852 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2860 = tile.add_modq %P_2850, %P_2853 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2861 = tile.add_modq %P_2854 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa4 = tile.recombine_modq %C_2855, %C_2856, %C_2857, %C_2858, %C_2859, %C_2860, %C_2861 {radix = 128 : i64, q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_5c = rns.const {name = "rl_evkA_5c", q = 268433761 : i64} : tensor<16xi32>
    %a_2862, %a_2863, %a_2864, %a_2865 = tile.decompose %W_f_268433761 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2866, %b_2867, %b_2868, %b_2869 = tile.decompose %rl_evkA_5c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2870 = tile.matmul_modq %a_2862, %b_2866 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2871 = tile.matmul_modq %a_2862, %b_2867 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2872 = tile.matmul_modq %a_2862, %b_2868 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2873 = tile.matmul_modq %a_2862, %b_2869 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2874 = tile.matmul_modq %a_2863, %b_2866 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2875 = tile.matmul_modq %a_2863, %b_2867 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2876 = tile.matmul_modq %a_2863, %b_2868 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2877 = tile.matmul_modq %a_2863, %b_2869 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2878 = tile.matmul_modq %a_2864, %b_2866 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2879 = tile.matmul_modq %a_2864, %b_2867 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2880 = tile.matmul_modq %a_2864, %b_2868 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2881 = tile.matmul_modq %a_2864, %b_2869 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2882 = tile.matmul_modq %a_2865, %b_2866 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2883 = tile.matmul_modq %a_2865, %b_2867 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2884 = tile.matmul_modq %a_2865, %b_2868 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2885 = tile.matmul_modq %a_2865, %b_2869 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2886 = tile.add_modq %P_2870 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2887 = tile.add_modq %P_2871, %P_2874 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2888 = tile.add_modq %P_2872, %P_2875, %P_2878 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2889 = tile.add_modq %P_2873, %P_2876, %P_2879, %P_2882 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2890 = tile.add_modq %P_2877, %P_2880, %P_2883 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2891 = tile.add_modq %P_2881, %P_2884 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2892 = tile.add_modq %P_2885 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkA_5 = tile.recombine_modq %C_2886, %C_2887, %C_2888, %C_2889, %C_2890, %C_2891, %C_2892 {radix = 128 : i64, q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_AY5 = tile.mul_modq %rl_X5, %rl_evkA_5 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %W_i_268433761 = tile.const {name = "W_i_268433761", q = 268433761 : i64} : tensor<16x16xi32>
    %a_2893, %a_2894, %a_2895, %a_2896 = tile.decompose %W_i_268433761 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2897, %b_2898, %b_2899, %b_2900 = tile.decompose %rl_AY5 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2901 = tile.matmul_modq %a_2893, %b_2897 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2902 = tile.matmul_modq %a_2893, %b_2898 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2903 = tile.matmul_modq %a_2893, %b_2899 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2904 = tile.matmul_modq %a_2893, %b_2900 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2905 = tile.matmul_modq %a_2894, %b_2897 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2906 = tile.matmul_modq %a_2894, %b_2898 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2907 = tile.matmul_modq %a_2894, %b_2899 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2908 = tile.matmul_modq %a_2894, %b_2900 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2909 = tile.matmul_modq %a_2895, %b_2897 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2910 = tile.matmul_modq %a_2895, %b_2898 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2911 = tile.matmul_modq %a_2895, %b_2899 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2912 = tile.matmul_modq %a_2895, %b_2900 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2913 = tile.matmul_modq %a_2896, %b_2897 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2914 = tile.matmul_modq %a_2896, %b_2898 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2915 = tile.matmul_modq %a_2896, %b_2899 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2916 = tile.matmul_modq %a_2896, %b_2900 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2917 = tile.add_modq %P_2901 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2918 = tile.add_modq %P_2902, %P_2905 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2919 = tile.add_modq %P_2903, %P_2906, %P_2909 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2920 = tile.add_modq %P_2904, %P_2907, %P_2910, %P_2913 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2921 = tile.add_modq %P_2908, %P_2911, %P_2914 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2922 = tile.add_modq %P_2912, %P_2915 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2923 = tile.add_modq %P_2916 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Aa5 = tile.recombine_modq %C_2917, %C_2918, %C_2919, %C_2920, %C_2921, %C_2922, %C_2923 {radix = 128 : i64, q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_0c = rns.const {name = "rl_evkB_0c", q = 268435361 : i64} : tensor<16xi32>
    %a_2924, %a_2925, %a_2926, %a_2927 = tile.decompose %W_f_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2928, %b_2929, %b_2930, %b_2931 = tile.decompose %rl_evkB_0c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2932 = tile.matmul_modq %a_2924, %b_2928 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2933 = tile.matmul_modq %a_2924, %b_2929 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2934 = tile.matmul_modq %a_2924, %b_2930 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2935 = tile.matmul_modq %a_2924, %b_2931 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2936 = tile.matmul_modq %a_2925, %b_2928 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2937 = tile.matmul_modq %a_2925, %b_2929 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2938 = tile.matmul_modq %a_2925, %b_2930 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2939 = tile.matmul_modq %a_2925, %b_2931 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2940 = tile.matmul_modq %a_2926, %b_2928 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2941 = tile.matmul_modq %a_2926, %b_2929 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2942 = tile.matmul_modq %a_2926, %b_2930 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2943 = tile.matmul_modq %a_2926, %b_2931 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2944 = tile.matmul_modq %a_2927, %b_2928 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2945 = tile.matmul_modq %a_2927, %b_2929 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2946 = tile.matmul_modq %a_2927, %b_2930 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2947 = tile.matmul_modq %a_2927, %b_2931 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2948 = tile.add_modq %P_2932 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2949 = tile.add_modq %P_2933, %P_2936 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2950 = tile.add_modq %P_2934, %P_2937, %P_2940 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2951 = tile.add_modq %P_2935, %P_2938, %P_2941, %P_2944 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2952 = tile.add_modq %P_2939, %P_2942, %P_2945 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2953 = tile.add_modq %P_2943, %P_2946 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2954 = tile.add_modq %P_2947 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_0 = tile.recombine_modq %C_2948, %C_2949, %C_2950, %C_2951, %C_2952, %C_2953, %C_2954 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY0 = tile.mul_modq %rl_X0, %rl_evkB_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_2955, %a_2956, %a_2957, %a_2958 = tile.decompose %W_i_268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2959, %b_2960, %b_2961, %b_2962 = tile.decompose %rl_BY0 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2963 = tile.matmul_modq %a_2955, %b_2959 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2964 = tile.matmul_modq %a_2955, %b_2960 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2965 = tile.matmul_modq %a_2955, %b_2961 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2966 = tile.matmul_modq %a_2955, %b_2962 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2967 = tile.matmul_modq %a_2956, %b_2959 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2968 = tile.matmul_modq %a_2956, %b_2960 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2969 = tile.matmul_modq %a_2956, %b_2961 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2970 = tile.matmul_modq %a_2956, %b_2962 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2971 = tile.matmul_modq %a_2957, %b_2959 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2972 = tile.matmul_modq %a_2957, %b_2960 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2973 = tile.matmul_modq %a_2957, %b_2961 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2974 = tile.matmul_modq %a_2957, %b_2962 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2975 = tile.matmul_modq %a_2958, %b_2959 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2976 = tile.matmul_modq %a_2958, %b_2960 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2977 = tile.matmul_modq %a_2958, %b_2961 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2978 = tile.matmul_modq %a_2958, %b_2962 {q = 268435361 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_2979 = tile.add_modq %P_2963 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_2980 = tile.add_modq %P_2964, %P_2967 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2981 = tile.add_modq %P_2965, %P_2968, %P_2971 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2982 = tile.add_modq %P_2966, %P_2969, %P_2972, %P_2975 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2983 = tile.add_modq %P_2970, %P_2973, %P_2976 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2984 = tile.add_modq %P_2974, %P_2977 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_2985 = tile.add_modq %P_2978 {q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba0 = tile.recombine_modq %C_2979, %C_2980, %C_2981, %C_2982, %C_2983, %C_2984, %C_2985 {radix = 128 : i64, q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_1c = rns.const {name = "rl_evkB_1c", q = 268435009 : i64} : tensor<16xi32>
    %a_2986, %a_2987, %a_2988, %a_2989 = tile.decompose %W_f_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_2990, %b_2991, %b_2992, %b_2993 = tile.decompose %rl_evkB_1c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_2994 = tile.matmul_modq %a_2986, %b_2990 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2995 = tile.matmul_modq %a_2986, %b_2991 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2996 = tile.matmul_modq %a_2986, %b_2992 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2997 = tile.matmul_modq %a_2986, %b_2993 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2998 = tile.matmul_modq %a_2987, %b_2990 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_2999 = tile.matmul_modq %a_2987, %b_2991 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3000 = tile.matmul_modq %a_2987, %b_2992 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3001 = tile.matmul_modq %a_2987, %b_2993 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3002 = tile.matmul_modq %a_2988, %b_2990 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3003 = tile.matmul_modq %a_2988, %b_2991 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3004 = tile.matmul_modq %a_2988, %b_2992 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3005 = tile.matmul_modq %a_2988, %b_2993 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3006 = tile.matmul_modq %a_2989, %b_2990 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3007 = tile.matmul_modq %a_2989, %b_2991 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3008 = tile.matmul_modq %a_2989, %b_2992 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3009 = tile.matmul_modq %a_2989, %b_2993 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_3010 = tile.add_modq %P_2994 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_3011 = tile.add_modq %P_2995, %P_2998 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3012 = tile.add_modq %P_2996, %P_2999, %P_3002 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3013 = tile.add_modq %P_2997, %P_3000, %P_3003, %P_3006 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3014 = tile.add_modq %P_3001, %P_3004, %P_3007 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3015 = tile.add_modq %P_3005, %P_3008 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3016 = tile.add_modq %P_3009 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_1 = tile.recombine_modq %C_3010, %C_3011, %C_3012, %C_3013, %C_3014, %C_3015, %C_3016 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY1 = tile.mul_modq %rl_X1, %rl_evkB_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_3017, %a_3018, %a_3019, %a_3020 = tile.decompose %W_i_268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_3021, %b_3022, %b_3023, %b_3024 = tile.decompose %rl_BY1 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_3025 = tile.matmul_modq %a_3017, %b_3021 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3026 = tile.matmul_modq %a_3017, %b_3022 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3027 = tile.matmul_modq %a_3017, %b_3023 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3028 = tile.matmul_modq %a_3017, %b_3024 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3029 = tile.matmul_modq %a_3018, %b_3021 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3030 = tile.matmul_modq %a_3018, %b_3022 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3031 = tile.matmul_modq %a_3018, %b_3023 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3032 = tile.matmul_modq %a_3018, %b_3024 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3033 = tile.matmul_modq %a_3019, %b_3021 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3034 = tile.matmul_modq %a_3019, %b_3022 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3035 = tile.matmul_modq %a_3019, %b_3023 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3036 = tile.matmul_modq %a_3019, %b_3024 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3037 = tile.matmul_modq %a_3020, %b_3021 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3038 = tile.matmul_modq %a_3020, %b_3022 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3039 = tile.matmul_modq %a_3020, %b_3023 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3040 = tile.matmul_modq %a_3020, %b_3024 {q = 268435009 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_3041 = tile.add_modq %P_3025 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_3042 = tile.add_modq %P_3026, %P_3029 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3043 = tile.add_modq %P_3027, %P_3030, %P_3033 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3044 = tile.add_modq %P_3028, %P_3031, %P_3034, %P_3037 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3045 = tile.add_modq %P_3032, %P_3035, %P_3038 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3046 = tile.add_modq %P_3036, %P_3039 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3047 = tile.add_modq %P_3040 {q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba1 = tile.recombine_modq %C_3041, %C_3042, %C_3043, %C_3044, %C_3045, %C_3046, %C_3047 {radix = 128 : i64, q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_2c = rns.const {name = "rl_evkB_2c", q = 268434977 : i64} : tensor<16xi32>
    %a_3048, %a_3049, %a_3050, %a_3051 = tile.decompose %W_f_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_3052, %b_3053, %b_3054, %b_3055 = tile.decompose %rl_evkB_2c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_3056 = tile.matmul_modq %a_3048, %b_3052 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3057 = tile.matmul_modq %a_3048, %b_3053 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3058 = tile.matmul_modq %a_3048, %b_3054 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3059 = tile.matmul_modq %a_3048, %b_3055 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3060 = tile.matmul_modq %a_3049, %b_3052 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3061 = tile.matmul_modq %a_3049, %b_3053 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3062 = tile.matmul_modq %a_3049, %b_3054 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3063 = tile.matmul_modq %a_3049, %b_3055 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3064 = tile.matmul_modq %a_3050, %b_3052 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3065 = tile.matmul_modq %a_3050, %b_3053 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3066 = tile.matmul_modq %a_3050, %b_3054 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3067 = tile.matmul_modq %a_3050, %b_3055 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3068 = tile.matmul_modq %a_3051, %b_3052 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3069 = tile.matmul_modq %a_3051, %b_3053 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3070 = tile.matmul_modq %a_3051, %b_3054 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3071 = tile.matmul_modq %a_3051, %b_3055 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_3072 = tile.add_modq %P_3056 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_3073 = tile.add_modq %P_3057, %P_3060 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3074 = tile.add_modq %P_3058, %P_3061, %P_3064 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3075 = tile.add_modq %P_3059, %P_3062, %P_3065, %P_3068 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3076 = tile.add_modq %P_3063, %P_3066, %P_3069 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3077 = tile.add_modq %P_3067, %P_3070 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3078 = tile.add_modq %P_3071 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_2 = tile.recombine_modq %C_3072, %C_3073, %C_3074, %C_3075, %C_3076, %C_3077, %C_3078 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY2 = tile.mul_modq %rl_X2, %rl_evkB_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_3079, %a_3080, %a_3081, %a_3082 = tile.decompose %W_i_268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_3083, %b_3084, %b_3085, %b_3086 = tile.decompose %rl_BY2 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_3087 = tile.matmul_modq %a_3079, %b_3083 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3088 = tile.matmul_modq %a_3079, %b_3084 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3089 = tile.matmul_modq %a_3079, %b_3085 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3090 = tile.matmul_modq %a_3079, %b_3086 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3091 = tile.matmul_modq %a_3080, %b_3083 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3092 = tile.matmul_modq %a_3080, %b_3084 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3093 = tile.matmul_modq %a_3080, %b_3085 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3094 = tile.matmul_modq %a_3080, %b_3086 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3095 = tile.matmul_modq %a_3081, %b_3083 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3096 = tile.matmul_modq %a_3081, %b_3084 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3097 = tile.matmul_modq %a_3081, %b_3085 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3098 = tile.matmul_modq %a_3081, %b_3086 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3099 = tile.matmul_modq %a_3082, %b_3083 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3100 = tile.matmul_modq %a_3082, %b_3084 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3101 = tile.matmul_modq %a_3082, %b_3085 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3102 = tile.matmul_modq %a_3082, %b_3086 {q = 268434977 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_3103 = tile.add_modq %P_3087 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_3104 = tile.add_modq %P_3088, %P_3091 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3105 = tile.add_modq %P_3089, %P_3092, %P_3095 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3106 = tile.add_modq %P_3090, %P_3093, %P_3096, %P_3099 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3107 = tile.add_modq %P_3094, %P_3097, %P_3100 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3108 = tile.add_modq %P_3098, %P_3101 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3109 = tile.add_modq %P_3102 {q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba2 = tile.recombine_modq %C_3103, %C_3104, %C_3105, %C_3106, %C_3107, %C_3108, %C_3109 {radix = 128 : i64, q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_3c = rns.const {name = "rl_evkB_3c", q = 268434721 : i64} : tensor<16xi32>
    %a_3110, %a_3111, %a_3112, %a_3113 = tile.decompose %W_f_268434721 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_3114, %b_3115, %b_3116, %b_3117 = tile.decompose %rl_evkB_3c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_3118 = tile.matmul_modq %a_3110, %b_3114 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3119 = tile.matmul_modq %a_3110, %b_3115 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3120 = tile.matmul_modq %a_3110, %b_3116 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3121 = tile.matmul_modq %a_3110, %b_3117 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3122 = tile.matmul_modq %a_3111, %b_3114 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3123 = tile.matmul_modq %a_3111, %b_3115 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3124 = tile.matmul_modq %a_3111, %b_3116 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3125 = tile.matmul_modq %a_3111, %b_3117 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3126 = tile.matmul_modq %a_3112, %b_3114 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3127 = tile.matmul_modq %a_3112, %b_3115 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3128 = tile.matmul_modq %a_3112, %b_3116 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3129 = tile.matmul_modq %a_3112, %b_3117 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3130 = tile.matmul_modq %a_3113, %b_3114 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3131 = tile.matmul_modq %a_3113, %b_3115 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3132 = tile.matmul_modq %a_3113, %b_3116 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3133 = tile.matmul_modq %a_3113, %b_3117 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_3134 = tile.add_modq %P_3118 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_3135 = tile.add_modq %P_3119, %P_3122 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3136 = tile.add_modq %P_3120, %P_3123, %P_3126 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3137 = tile.add_modq %P_3121, %P_3124, %P_3127, %P_3130 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3138 = tile.add_modq %P_3125, %P_3128, %P_3131 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3139 = tile.add_modq %P_3129, %P_3132 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3140 = tile.add_modq %P_3133 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_3 = tile.recombine_modq %C_3134, %C_3135, %C_3136, %C_3137, %C_3138, %C_3139, %C_3140 {radix = 128 : i64, q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY3 = tile.mul_modq %rl_X3, %rl_evkB_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_3141, %a_3142, %a_3143, %a_3144 = tile.decompose %W_i_268434721 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_3145, %b_3146, %b_3147, %b_3148 = tile.decompose %rl_BY3 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_3149 = tile.matmul_modq %a_3141, %b_3145 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3150 = tile.matmul_modq %a_3141, %b_3146 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3151 = tile.matmul_modq %a_3141, %b_3147 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3152 = tile.matmul_modq %a_3141, %b_3148 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3153 = tile.matmul_modq %a_3142, %b_3145 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3154 = tile.matmul_modq %a_3142, %b_3146 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3155 = tile.matmul_modq %a_3142, %b_3147 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3156 = tile.matmul_modq %a_3142, %b_3148 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3157 = tile.matmul_modq %a_3143, %b_3145 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3158 = tile.matmul_modq %a_3143, %b_3146 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3159 = tile.matmul_modq %a_3143, %b_3147 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3160 = tile.matmul_modq %a_3143, %b_3148 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3161 = tile.matmul_modq %a_3144, %b_3145 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3162 = tile.matmul_modq %a_3144, %b_3146 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3163 = tile.matmul_modq %a_3144, %b_3147 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3164 = tile.matmul_modq %a_3144, %b_3148 {q = 268434721 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_3165 = tile.add_modq %P_3149 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_3166 = tile.add_modq %P_3150, %P_3153 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3167 = tile.add_modq %P_3151, %P_3154, %P_3157 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3168 = tile.add_modq %P_3152, %P_3155, %P_3158, %P_3161 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3169 = tile.add_modq %P_3156, %P_3159, %P_3162 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3170 = tile.add_modq %P_3160, %P_3163 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3171 = tile.add_modq %P_3164 {q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba3 = tile.recombine_modq %C_3165, %C_3166, %C_3167, %C_3168, %C_3169, %C_3170, %C_3171 {radix = 128 : i64, q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_4c = rns.const {name = "rl_evkB_4c", q = 268434401 : i64} : tensor<16xi32>
    %a_3172, %a_3173, %a_3174, %a_3175 = tile.decompose %W_f_268434401 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_3176, %b_3177, %b_3178, %b_3179 = tile.decompose %rl_evkB_4c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_3180 = tile.matmul_modq %a_3172, %b_3176 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3181 = tile.matmul_modq %a_3172, %b_3177 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3182 = tile.matmul_modq %a_3172, %b_3178 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3183 = tile.matmul_modq %a_3172, %b_3179 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3184 = tile.matmul_modq %a_3173, %b_3176 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3185 = tile.matmul_modq %a_3173, %b_3177 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3186 = tile.matmul_modq %a_3173, %b_3178 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3187 = tile.matmul_modq %a_3173, %b_3179 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3188 = tile.matmul_modq %a_3174, %b_3176 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3189 = tile.matmul_modq %a_3174, %b_3177 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3190 = tile.matmul_modq %a_3174, %b_3178 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3191 = tile.matmul_modq %a_3174, %b_3179 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3192 = tile.matmul_modq %a_3175, %b_3176 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3193 = tile.matmul_modq %a_3175, %b_3177 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3194 = tile.matmul_modq %a_3175, %b_3178 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3195 = tile.matmul_modq %a_3175, %b_3179 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_3196 = tile.add_modq %P_3180 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_3197 = tile.add_modq %P_3181, %P_3184 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3198 = tile.add_modq %P_3182, %P_3185, %P_3188 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3199 = tile.add_modq %P_3183, %P_3186, %P_3189, %P_3192 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3200 = tile.add_modq %P_3187, %P_3190, %P_3193 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3201 = tile.add_modq %P_3191, %P_3194 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3202 = tile.add_modq %P_3195 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_4 = tile.recombine_modq %C_3196, %C_3197, %C_3198, %C_3199, %C_3200, %C_3201, %C_3202 {radix = 128 : i64, q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY4 = tile.mul_modq %rl_X4, %rl_evkB_4 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_3203, %a_3204, %a_3205, %a_3206 = tile.decompose %W_i_268434401 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_3207, %b_3208, %b_3209, %b_3210 = tile.decompose %rl_BY4 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_3211 = tile.matmul_modq %a_3203, %b_3207 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3212 = tile.matmul_modq %a_3203, %b_3208 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3213 = tile.matmul_modq %a_3203, %b_3209 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3214 = tile.matmul_modq %a_3203, %b_3210 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3215 = tile.matmul_modq %a_3204, %b_3207 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3216 = tile.matmul_modq %a_3204, %b_3208 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3217 = tile.matmul_modq %a_3204, %b_3209 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3218 = tile.matmul_modq %a_3204, %b_3210 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3219 = tile.matmul_modq %a_3205, %b_3207 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3220 = tile.matmul_modq %a_3205, %b_3208 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3221 = tile.matmul_modq %a_3205, %b_3209 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3222 = tile.matmul_modq %a_3205, %b_3210 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3223 = tile.matmul_modq %a_3206, %b_3207 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3224 = tile.matmul_modq %a_3206, %b_3208 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3225 = tile.matmul_modq %a_3206, %b_3209 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3226 = tile.matmul_modq %a_3206, %b_3210 {q = 268434401 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_3227 = tile.add_modq %P_3211 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_3228 = tile.add_modq %P_3212, %P_3215 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3229 = tile.add_modq %P_3213, %P_3216, %P_3219 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3230 = tile.add_modq %P_3214, %P_3217, %P_3220, %P_3223 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3231 = tile.add_modq %P_3218, %P_3221, %P_3224 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3232 = tile.add_modq %P_3222, %P_3225 {q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3233 = tile.add_modq %P_3226 {q = 268434401 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba4 = tile.recombine_modq %C_3227, %C_3228, %C_3229, %C_3230, %C_3231, %C_3232, %C_3233 {radix = 128 : i64, q = 268434401 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_5c = rns.const {name = "rl_evkB_5c", q = 268433761 : i64} : tensor<16xi32>
    %a_3234, %a_3235, %a_3236, %a_3237 = tile.decompose %W_f_268433761 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_3238, %b_3239, %b_3240, %b_3241 = tile.decompose %rl_evkB_5c {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_3242 = tile.matmul_modq %a_3234, %b_3238 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3243 = tile.matmul_modq %a_3234, %b_3239 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3244 = tile.matmul_modq %a_3234, %b_3240 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3245 = tile.matmul_modq %a_3234, %b_3241 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3246 = tile.matmul_modq %a_3235, %b_3238 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3247 = tile.matmul_modq %a_3235, %b_3239 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3248 = tile.matmul_modq %a_3235, %b_3240 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3249 = tile.matmul_modq %a_3235, %b_3241 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3250 = tile.matmul_modq %a_3236, %b_3238 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3251 = tile.matmul_modq %a_3236, %b_3239 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3252 = tile.matmul_modq %a_3236, %b_3240 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3253 = tile.matmul_modq %a_3236, %b_3241 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3254 = tile.matmul_modq %a_3237, %b_3238 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3255 = tile.matmul_modq %a_3237, %b_3239 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3256 = tile.matmul_modq %a_3237, %b_3240 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3257 = tile.matmul_modq %a_3237, %b_3241 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_3258 = tile.add_modq %P_3242 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_3259 = tile.add_modq %P_3243, %P_3246 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3260 = tile.add_modq %P_3244, %P_3247, %P_3250 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3261 = tile.add_modq %P_3245, %P_3248, %P_3251, %P_3254 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3262 = tile.add_modq %P_3249, %P_3252, %P_3255 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3263 = tile.add_modq %P_3253, %P_3256 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3264 = tile.add_modq %P_3257 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_evkB_5 = tile.recombine_modq %C_3258, %C_3259, %C_3260, %C_3261, %C_3262, %C_3263, %C_3264 {radix = 128 : i64, q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %rl_BY5 = tile.mul_modq %rl_X5, %rl_evkB_5 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %a_3265, %a_3266, %a_3267, %a_3268 = tile.decompose %W_i_268433761 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16x16xi32>) -> (tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>, tensor<16x16xi8>)
    %b_3269, %b_3270, %b_3271, %b_3272 = tile.decompose %rl_BY5 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<16xi32>) -> (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
    %P_3273 = tile.matmul_modq %a_3265, %b_3269 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3274 = tile.matmul_modq %a_3265, %b_3270 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3275 = tile.matmul_modq %a_3265, %b_3271 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3276 = tile.matmul_modq %a_3265, %b_3272 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3277 = tile.matmul_modq %a_3266, %b_3269 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3278 = tile.matmul_modq %a_3266, %b_3270 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3279 = tile.matmul_modq %a_3266, %b_3271 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3280 = tile.matmul_modq %a_3266, %b_3272 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3281 = tile.matmul_modq %a_3267, %b_3269 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3282 = tile.matmul_modq %a_3267, %b_3270 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3283 = tile.matmul_modq %a_3267, %b_3271 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3284 = tile.matmul_modq %a_3267, %b_3272 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3285 = tile.matmul_modq %a_3268, %b_3269 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3286 = tile.matmul_modq %a_3268, %b_3270 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3287 = tile.matmul_modq %a_3268, %b_3271 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %P_3288 = tile.matmul_modq %a_3268, %b_3272 {q = 268433761 : i64} : (tensor<16x16xi8>, tensor<16xi8>) -> tensor<16xi32>
    %C_3289 = tile.add_modq %P_3273 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C_3290 = tile.add_modq %P_3274, %P_3277 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3291 = tile.add_modq %P_3275, %P_3278, %P_3281 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3292 = tile.add_modq %P_3276, %P_3279, %P_3282, %P_3285 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3293 = tile.add_modq %P_3280, %P_3283, %P_3286 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3294 = tile.add_modq %P_3284, %P_3287 {q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %C_3295 = tile.add_modq %P_3288 {q = 268433761 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %rl_Ba5 = tile.recombine_modq %C_3289, %C_3290, %C_3291, %C_3292, %C_3293, %C_3294, %C_3295 {radix = 128 : i64, q = 268433761 : i64} : (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
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
    %a_3296, %a_3297, %a_3298, %a_3299 = tile.decompose %fbcD__268434721__268434401__268433761__268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_3300, %b_3301, %b_3302, %b_3303 = tile.decompose %rl_AxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_3304 = tile.matmul_modq %a_3296, %b_3300 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3305 = tile.matmul_modq %a_3296, %b_3301 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3306 = tile.matmul_modq %a_3296, %b_3302 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3307 = tile.matmul_modq %a_3296, %b_3303 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3308 = tile.matmul_modq %a_3297, %b_3300 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3309 = tile.matmul_modq %a_3297, %b_3301 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3310 = tile.matmul_modq %a_3297, %b_3302 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3311 = tile.matmul_modq %a_3297, %b_3303 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3312 = tile.matmul_modq %a_3298, %b_3300 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3313 = tile.matmul_modq %a_3298, %b_3301 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3314 = tile.matmul_modq %a_3298, %b_3302 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3315 = tile.matmul_modq %a_3298, %b_3303 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3316 = tile.matmul_modq %a_3299, %b_3300 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3317 = tile.matmul_modq %a_3299, %b_3301 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3318 = tile.matmul_modq %a_3299, %b_3302 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3319 = tile.matmul_modq %a_3299, %b_3303 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_3320 = tile.add_modq %P_3304 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3321 = tile.add_modq %P_3305, %P_3308 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3322 = tile.add_modq %P_3306, %P_3309, %P_3312 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3323 = tile.add_modq %P_3307, %P_3310, %P_3313, %P_3316 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3324 = tile.add_modq %P_3311, %P_3314, %P_3317 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3325 = tile.add_modq %P_3315, %P_3318 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3326 = tile.add_modq %P_3319 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_AxP0_fbc_268435361 = tile.recombine_modq %C_3320, %C_3321, %C_3322, %C_3323, %C_3324, %C_3325, %C_3326 {radix = 128 : i64, q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_AxP0 = tile.reshape %rl_AxP0_fbc_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268434721__268434401__268433761__268435009 = tile.const {name = "fbcD_(268434721, 268434401, 268433761)_268435009"} : tensor<1x3xi32>
    %a_3327, %a_3328, %a_3329, %a_3330 = tile.decompose %fbcD__268434721__268434401__268433761__268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_3331, %b_3332, %b_3333, %b_3334 = tile.decompose %rl_AxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_3335 = tile.matmul_modq %a_3327, %b_3331 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3336 = tile.matmul_modq %a_3327, %b_3332 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3337 = tile.matmul_modq %a_3327, %b_3333 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3338 = tile.matmul_modq %a_3327, %b_3334 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3339 = tile.matmul_modq %a_3328, %b_3331 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3340 = tile.matmul_modq %a_3328, %b_3332 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3341 = tile.matmul_modq %a_3328, %b_3333 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3342 = tile.matmul_modq %a_3328, %b_3334 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3343 = tile.matmul_modq %a_3329, %b_3331 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3344 = tile.matmul_modq %a_3329, %b_3332 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3345 = tile.matmul_modq %a_3329, %b_3333 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3346 = tile.matmul_modq %a_3329, %b_3334 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3347 = tile.matmul_modq %a_3330, %b_3331 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3348 = tile.matmul_modq %a_3330, %b_3332 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3349 = tile.matmul_modq %a_3330, %b_3333 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3350 = tile.matmul_modq %a_3330, %b_3334 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_3351 = tile.add_modq %P_3335 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3352 = tile.add_modq %P_3336, %P_3339 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3353 = tile.add_modq %P_3337, %P_3340, %P_3343 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3354 = tile.add_modq %P_3338, %P_3341, %P_3344, %P_3347 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3355 = tile.add_modq %P_3342, %P_3345, %P_3348 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3356 = tile.add_modq %P_3346, %P_3349 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3357 = tile.add_modq %P_3350 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_AxP0_fbc_268435009 = tile.recombine_modq %C_3351, %C_3352, %C_3353, %C_3354, %C_3355, %C_3356, %C_3357 {radix = 128 : i64, q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_AxP1 = tile.reshape %rl_AxP0_fbc_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %fbcD__268434721__268434401__268433761__268434977 = tile.const {name = "fbcD_(268434721, 268434401, 268433761)_268434977"} : tensor<1x3xi32>
    %a_3358, %a_3359, %a_3360, %a_3361 = tile.decompose %fbcD__268434721__268434401__268433761__268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_3362, %b_3363, %b_3364, %b_3365 = tile.decompose %rl_AxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_3366 = tile.matmul_modq %a_3358, %b_3362 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3367 = tile.matmul_modq %a_3358, %b_3363 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3368 = tile.matmul_modq %a_3358, %b_3364 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3369 = tile.matmul_modq %a_3358, %b_3365 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3370 = tile.matmul_modq %a_3359, %b_3362 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3371 = tile.matmul_modq %a_3359, %b_3363 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3372 = tile.matmul_modq %a_3359, %b_3364 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3373 = tile.matmul_modq %a_3359, %b_3365 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3374 = tile.matmul_modq %a_3360, %b_3362 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3375 = tile.matmul_modq %a_3360, %b_3363 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3376 = tile.matmul_modq %a_3360, %b_3364 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3377 = tile.matmul_modq %a_3360, %b_3365 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3378 = tile.matmul_modq %a_3361, %b_3362 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3379 = tile.matmul_modq %a_3361, %b_3363 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3380 = tile.matmul_modq %a_3361, %b_3364 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3381 = tile.matmul_modq %a_3361, %b_3365 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_3382 = tile.add_modq %P_3366 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3383 = tile.add_modq %P_3367, %P_3370 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3384 = tile.add_modq %P_3368, %P_3371, %P_3374 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3385 = tile.add_modq %P_3369, %P_3372, %P_3375, %P_3378 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3386 = tile.add_modq %P_3373, %P_3376, %P_3379 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3387 = tile.add_modq %P_3377, %P_3380 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3388 = tile.add_modq %P_3381 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_AxP0_fbc_268434977 = tile.recombine_modq %C_3382, %C_3383, %C_3384, %C_3385, %C_3386, %C_3387, %C_3388 {radix = 128 : i64, q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
    %a_3389, %a_3390, %a_3391, %a_3392 = tile.decompose %fbcD__268434721__268434401__268433761__268435361 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_3393, %b_3394, %b_3395, %b_3396 = tile.decompose %rl_BxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_3397 = tile.matmul_modq %a_3389, %b_3393 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3398 = tile.matmul_modq %a_3389, %b_3394 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3399 = tile.matmul_modq %a_3389, %b_3395 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3400 = tile.matmul_modq %a_3389, %b_3396 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3401 = tile.matmul_modq %a_3390, %b_3393 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3402 = tile.matmul_modq %a_3390, %b_3394 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3403 = tile.matmul_modq %a_3390, %b_3395 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3404 = tile.matmul_modq %a_3390, %b_3396 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3405 = tile.matmul_modq %a_3391, %b_3393 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3406 = tile.matmul_modq %a_3391, %b_3394 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3407 = tile.matmul_modq %a_3391, %b_3395 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3408 = tile.matmul_modq %a_3391, %b_3396 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3409 = tile.matmul_modq %a_3392, %b_3393 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3410 = tile.matmul_modq %a_3392, %b_3394 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3411 = tile.matmul_modq %a_3392, %b_3395 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3412 = tile.matmul_modq %a_3392, %b_3396 {q = 268435361 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_3413 = tile.add_modq %P_3397 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3414 = tile.add_modq %P_3398, %P_3401 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3415 = tile.add_modq %P_3399, %P_3402, %P_3405 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3416 = tile.add_modq %P_3400, %P_3403, %P_3406, %P_3409 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3417 = tile.add_modq %P_3404, %P_3407, %P_3410 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3418 = tile.add_modq %P_3408, %P_3411 {q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3419 = tile.add_modq %P_3412 {q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_BxP0_fbc_268435361 = tile.recombine_modq %C_3413, %C_3414, %C_3415, %C_3416, %C_3417, %C_3418, %C_3419 {radix = 128 : i64, q = 268435361 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_BxP0 = tile.reshape %rl_BxP0_fbc_268435361 {shape = [16], q = 268435361 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_3420, %a_3421, %a_3422, %a_3423 = tile.decompose %fbcD__268434721__268434401__268433761__268435009 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_3424, %b_3425, %b_3426, %b_3427 = tile.decompose %rl_BxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_3428 = tile.matmul_modq %a_3420, %b_3424 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3429 = tile.matmul_modq %a_3420, %b_3425 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3430 = tile.matmul_modq %a_3420, %b_3426 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3431 = tile.matmul_modq %a_3420, %b_3427 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3432 = tile.matmul_modq %a_3421, %b_3424 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3433 = tile.matmul_modq %a_3421, %b_3425 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3434 = tile.matmul_modq %a_3421, %b_3426 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3435 = tile.matmul_modq %a_3421, %b_3427 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3436 = tile.matmul_modq %a_3422, %b_3424 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3437 = tile.matmul_modq %a_3422, %b_3425 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3438 = tile.matmul_modq %a_3422, %b_3426 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3439 = tile.matmul_modq %a_3422, %b_3427 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3440 = tile.matmul_modq %a_3423, %b_3424 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3441 = tile.matmul_modq %a_3423, %b_3425 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3442 = tile.matmul_modq %a_3423, %b_3426 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3443 = tile.matmul_modq %a_3423, %b_3427 {q = 268435009 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_3444 = tile.add_modq %P_3428 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3445 = tile.add_modq %P_3429, %P_3432 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3446 = tile.add_modq %P_3430, %P_3433, %P_3436 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3447 = tile.add_modq %P_3431, %P_3434, %P_3437, %P_3440 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3448 = tile.add_modq %P_3435, %P_3438, %P_3441 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3449 = tile.add_modq %P_3439, %P_3442 {q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3450 = tile.add_modq %P_3443 {q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_BxP0_fbc_268435009 = tile.recombine_modq %C_3444, %C_3445, %C_3446, %C_3447, %C_3448, %C_3449, %C_3450 {radix = 128 : i64, q = 268435009 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_BxP1 = tile.reshape %rl_BxP0_fbc_268435009 {shape = [16], q = 268435009 : i64} : (tensor<1x16xi32>) -> tensor<16xi32>
    %a_3451, %a_3452, %a_3453, %a_3454 = tile.decompose %fbcD__268434721__268434401__268433761__268434977 {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<1x3xi32>) -> (tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>, tensor<1x3xi8>)
    %b_3455, %b_3456, %b_3457, %b_3458 = tile.decompose %rl_BxP0_Y {radix = 128 : i64, nlimbs = 4 : i64} : (tensor<3x16xi32>) -> (tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>, tensor<3x16xi8>)
    %P_3459 = tile.matmul_modq %a_3451, %b_3455 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3460 = tile.matmul_modq %a_3451, %b_3456 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3461 = tile.matmul_modq %a_3451, %b_3457 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3462 = tile.matmul_modq %a_3451, %b_3458 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3463 = tile.matmul_modq %a_3452, %b_3455 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3464 = tile.matmul_modq %a_3452, %b_3456 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3465 = tile.matmul_modq %a_3452, %b_3457 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3466 = tile.matmul_modq %a_3452, %b_3458 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3467 = tile.matmul_modq %a_3453, %b_3455 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3468 = tile.matmul_modq %a_3453, %b_3456 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3469 = tile.matmul_modq %a_3453, %b_3457 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3470 = tile.matmul_modq %a_3453, %b_3458 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3471 = tile.matmul_modq %a_3454, %b_3455 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3472 = tile.matmul_modq %a_3454, %b_3456 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3473 = tile.matmul_modq %a_3454, %b_3457 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %P_3474 = tile.matmul_modq %a_3454, %b_3458 {q = 268434977 : i64} : (tensor<1x3xi8>, tensor<3x16xi8>) -> tensor<1x16xi32>
    %C_3475 = tile.add_modq %P_3459 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3476 = tile.add_modq %P_3460, %P_3463 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3477 = tile.add_modq %P_3461, %P_3464, %P_3467 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3478 = tile.add_modq %P_3462, %P_3465, %P_3468, %P_3471 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3479 = tile.add_modq %P_3466, %P_3469, %P_3472 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3480 = tile.add_modq %P_3470, %P_3473 {q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %C_3481 = tile.add_modq %P_3474 {q = 268434977 : i64} : (tensor<1x16xi32>) -> tensor<1x16xi32>
    %rl_BxP0_fbc_268434977 = tile.recombine_modq %C_3475, %C_3476, %C_3477, %C_3478, %C_3479, %C_3480, %C_3481 {radix = 128 : i64, q = 268434977 : i64} : (tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
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
