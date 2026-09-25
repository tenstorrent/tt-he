module {
  func.func @ntt_tiled(%x: tensor<4096xi32>) -> tensor<4096xi32> {
    %0 = ntt.reshape %x : tensor<4096xi32> to tensor<8x512xi32>
    %1 = ntt.const {name = "W0_T"} : tensor<8x8xi32>
    %0__t1, %0__t2, %0__t3, %0__t4, %0__t5, %0__t6, %0__t7, %0__t8, %0__t9, %0__t10, %0__t11, %0__t12, %0__t13, %0__t14, %0__t15, %0__t16, %0__t17, %0__t18, %0__t19, %0__t20, %0__t21, %0__t22, %0__t23, %0__t24, %0__t25, %0__t26, %0__t27, %0__t28, %0__t29, %0__t30, %0__t31, %0__t32 = ntt.split %0 {axis = 1 : index, parts = 32 : index} : tensor<8x512xi32> -> (tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>)
    %2__t33 = ntt.matmul_modq %0__t1, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t34 = ntt.const {name = "T0_T__r0"} : tensor<16x8xi32>
    %2__t35 = ntt.mul_modq %2__t33, %3__t34 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t36 = ntt.matmul_modq %0__t2, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t37 = ntt.const {name = "T0_T__r1"} : tensor<16x8xi32>
    %2__t38 = ntt.mul_modq %2__t36, %3__t37 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t39 = ntt.matmul_modq %0__t3, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t40 = ntt.const {name = "T0_T__r2"} : tensor<16x8xi32>
    %2__t41 = ntt.mul_modq %2__t39, %3__t40 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t42 = ntt.matmul_modq %0__t4, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t43 = ntt.const {name = "T0_T__r3"} : tensor<16x8xi32>
    %2__t44 = ntt.mul_modq %2__t42, %3__t43 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t45 = ntt.matmul_modq %0__t5, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t46 = ntt.const {name = "T0_T__r4"} : tensor<16x8xi32>
    %2__t47 = ntt.mul_modq %2__t45, %3__t46 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t48 = ntt.matmul_modq %0__t6, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t49 = ntt.const {name = "T0_T__r5"} : tensor<16x8xi32>
    %2__t50 = ntt.mul_modq %2__t48, %3__t49 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t51 = ntt.matmul_modq %0__t7, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t52 = ntt.const {name = "T0_T__r6"} : tensor<16x8xi32>
    %2__t53 = ntt.mul_modq %2__t51, %3__t52 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t54 = ntt.matmul_modq %0__t8, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t55 = ntt.const {name = "T0_T__r7"} : tensor<16x8xi32>
    %2__t56 = ntt.mul_modq %2__t54, %3__t55 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t57 = ntt.matmul_modq %0__t9, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t58 = ntt.const {name = "T0_T__r8"} : tensor<16x8xi32>
    %2__t59 = ntt.mul_modq %2__t57, %3__t58 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t60 = ntt.matmul_modq %0__t10, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t61 = ntt.const {name = "T0_T__r9"} : tensor<16x8xi32>
    %2__t62 = ntt.mul_modq %2__t60, %3__t61 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t63 = ntt.matmul_modq %0__t11, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t64 = ntt.const {name = "T0_T__r10"} : tensor<16x8xi32>
    %2__t65 = ntt.mul_modq %2__t63, %3__t64 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t66 = ntt.matmul_modq %0__t12, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t67 = ntt.const {name = "T0_T__r11"} : tensor<16x8xi32>
    %2__t68 = ntt.mul_modq %2__t66, %3__t67 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t69 = ntt.matmul_modq %0__t13, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t70 = ntt.const {name = "T0_T__r12"} : tensor<16x8xi32>
    %2__t71 = ntt.mul_modq %2__t69, %3__t70 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t72 = ntt.matmul_modq %0__t14, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t73 = ntt.const {name = "T0_T__r13"} : tensor<16x8xi32>
    %2__t74 = ntt.mul_modq %2__t72, %3__t73 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t75 = ntt.matmul_modq %0__t15, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t76 = ntt.const {name = "T0_T__r14"} : tensor<16x8xi32>
    %2__t77 = ntt.mul_modq %2__t75, %3__t76 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t78 = ntt.matmul_modq %0__t16, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t79 = ntt.const {name = "T0_T__r15"} : tensor<16x8xi32>
    %2__t80 = ntt.mul_modq %2__t78, %3__t79 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t81 = ntt.matmul_modq %0__t17, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t82 = ntt.const {name = "T0_T__r16"} : tensor<16x8xi32>
    %2__t83 = ntt.mul_modq %2__t81, %3__t82 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t84 = ntt.matmul_modq %0__t18, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t85 = ntt.const {name = "T0_T__r17"} : tensor<16x8xi32>
    %2__t86 = ntt.mul_modq %2__t84, %3__t85 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t87 = ntt.matmul_modq %0__t19, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t88 = ntt.const {name = "T0_T__r18"} : tensor<16x8xi32>
    %2__t89 = ntt.mul_modq %2__t87, %3__t88 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t90 = ntt.matmul_modq %0__t20, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t91 = ntt.const {name = "T0_T__r19"} : tensor<16x8xi32>
    %2__t92 = ntt.mul_modq %2__t90, %3__t91 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t93 = ntt.matmul_modq %0__t21, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t94 = ntt.const {name = "T0_T__r20"} : tensor<16x8xi32>
    %2__t95 = ntt.mul_modq %2__t93, %3__t94 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t96 = ntt.matmul_modq %0__t22, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t97 = ntt.const {name = "T0_T__r21"} : tensor<16x8xi32>
    %2__t98 = ntt.mul_modq %2__t96, %3__t97 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t99 = ntt.matmul_modq %0__t23, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t100 = ntt.const {name = "T0_T__r22"} : tensor<16x8xi32>
    %2__t101 = ntt.mul_modq %2__t99, %3__t100 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t102 = ntt.matmul_modq %0__t24, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t103 = ntt.const {name = "T0_T__r23"} : tensor<16x8xi32>
    %2__t104 = ntt.mul_modq %2__t102, %3__t103 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t105 = ntt.matmul_modq %0__t25, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t106 = ntt.const {name = "T0_T__r24"} : tensor<16x8xi32>
    %2__t107 = ntt.mul_modq %2__t105, %3__t106 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t108 = ntt.matmul_modq %0__t26, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t109 = ntt.const {name = "T0_T__r25"} : tensor<16x8xi32>
    %2__t110 = ntt.mul_modq %2__t108, %3__t109 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t111 = ntt.matmul_modq %0__t27, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t112 = ntt.const {name = "T0_T__r26"} : tensor<16x8xi32>
    %2__t113 = ntt.mul_modq %2__t111, %3__t112 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t114 = ntt.matmul_modq %0__t28, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t115 = ntt.const {name = "T0_T__r27"} : tensor<16x8xi32>
    %2__t116 = ntt.mul_modq %2__t114, %3__t115 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t117 = ntt.matmul_modq %0__t29, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t118 = ntt.const {name = "T0_T__r28"} : tensor<16x8xi32>
    %2__t119 = ntt.mul_modq %2__t117, %3__t118 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t120 = ntt.matmul_modq %0__t30, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t121 = ntt.const {name = "T0_T__r29"} : tensor<16x8xi32>
    %2__t122 = ntt.mul_modq %2__t120, %3__t121 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t123 = ntt.matmul_modq %0__t31, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t124 = ntt.const {name = "T0_T__r30"} : tensor<16x8xi32>
    %2__t125 = ntt.mul_modq %2__t123, %3__t124 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %2__t126 = ntt.matmul_modq %0__t32, %1 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %3__t127 = ntt.const {name = "T0_T__r31"} : tensor<16x8xi32>
    %2__t128 = ntt.mul_modq %2__t126, %3__t127 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %6__t129, %6__t130, %6__t131, %6__t132, %6__t133, %6__t134, %6__t135, %6__t136, %6__t137, %6__t138, %6__t139, %6__t140, %6__t141, %6__t142, %6__t143, %6__t144, %6__t145, %6__t146, %6__t147, %6__t148, %6__t149, %6__t150, %6__t151, %6__t152, %6__t153, %6__t154, %6__t155, %6__t156, %6__t157, %6__t158, %6__t159, %6__t160 = ntt.all_to_all %2__t35, %2__t38, %2__t41, %2__t44, %2__t47, %2__t50, %2__t53, %2__t56, %2__t59, %2__t62, %2__t65, %2__t68, %2__t71, %2__t74, %2__t77, %2__t80, %2__t83, %2__t86, %2__t89, %2__t92, %2__t95, %2__t98, %2__t101, %2__t104, %2__t107, %2__t110, %2__t113, %2__t116, %2__t119, %2__t122, %2__t125, %2__t128 {reshape_to = [8, 512], parts = 32 : index} : (tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>) -> (tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>)
    %7 = ntt.const {name = "W1_T"} : tensor<8x8xi32>
    %8__t161 = ntt.matmul_modq %6__t129, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t162 = ntt.const {name = "T1_T__r0"} : tensor<16x8xi32>
    %8__t163 = ntt.mul_modq %8__t161, %9__t162 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t164 = ntt.matmul_modq %6__t130, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t165 = ntt.const {name = "T1_T__r1"} : tensor<16x8xi32>
    %8__t166 = ntt.mul_modq %8__t164, %9__t165 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t167 = ntt.matmul_modq %6__t131, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t168 = ntt.const {name = "T1_T__r2"} : tensor<16x8xi32>
    %8__t169 = ntt.mul_modq %8__t167, %9__t168 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t170 = ntt.matmul_modq %6__t132, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t171 = ntt.const {name = "T1_T__r3"} : tensor<16x8xi32>
    %8__t172 = ntt.mul_modq %8__t170, %9__t171 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t173 = ntt.matmul_modq %6__t133, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t174 = ntt.const {name = "T1_T__r4"} : tensor<16x8xi32>
    %8__t175 = ntt.mul_modq %8__t173, %9__t174 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t176 = ntt.matmul_modq %6__t134, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t177 = ntt.const {name = "T1_T__r5"} : tensor<16x8xi32>
    %8__t178 = ntt.mul_modq %8__t176, %9__t177 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t179 = ntt.matmul_modq %6__t135, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t180 = ntt.const {name = "T1_T__r6"} : tensor<16x8xi32>
    %8__t181 = ntt.mul_modq %8__t179, %9__t180 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t182 = ntt.matmul_modq %6__t136, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t183 = ntt.const {name = "T1_T__r7"} : tensor<16x8xi32>
    %8__t184 = ntt.mul_modq %8__t182, %9__t183 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t185 = ntt.matmul_modq %6__t137, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t186 = ntt.const {name = "T1_T__r8"} : tensor<16x8xi32>
    %8__t187 = ntt.mul_modq %8__t185, %9__t186 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t188 = ntt.matmul_modq %6__t138, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t189 = ntt.const {name = "T1_T__r9"} : tensor<16x8xi32>
    %8__t190 = ntt.mul_modq %8__t188, %9__t189 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t191 = ntt.matmul_modq %6__t139, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t192 = ntt.const {name = "T1_T__r10"} : tensor<16x8xi32>
    %8__t193 = ntt.mul_modq %8__t191, %9__t192 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t194 = ntt.matmul_modq %6__t140, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t195 = ntt.const {name = "T1_T__r11"} : tensor<16x8xi32>
    %8__t196 = ntt.mul_modq %8__t194, %9__t195 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t197 = ntt.matmul_modq %6__t141, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t198 = ntt.const {name = "T1_T__r12"} : tensor<16x8xi32>
    %8__t199 = ntt.mul_modq %8__t197, %9__t198 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t200 = ntt.matmul_modq %6__t142, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t201 = ntt.const {name = "T1_T__r13"} : tensor<16x8xi32>
    %8__t202 = ntt.mul_modq %8__t200, %9__t201 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t203 = ntt.matmul_modq %6__t143, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t204 = ntt.const {name = "T1_T__r14"} : tensor<16x8xi32>
    %8__t205 = ntt.mul_modq %8__t203, %9__t204 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t206 = ntt.matmul_modq %6__t144, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t207 = ntt.const {name = "T1_T__r15"} : tensor<16x8xi32>
    %8__t208 = ntt.mul_modq %8__t206, %9__t207 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t209 = ntt.matmul_modq %6__t145, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t210 = ntt.const {name = "T1_T__r16"} : tensor<16x8xi32>
    %8__t211 = ntt.mul_modq %8__t209, %9__t210 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t212 = ntt.matmul_modq %6__t146, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t213 = ntt.const {name = "T1_T__r17"} : tensor<16x8xi32>
    %8__t214 = ntt.mul_modq %8__t212, %9__t213 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t215 = ntt.matmul_modq %6__t147, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t216 = ntt.const {name = "T1_T__r18"} : tensor<16x8xi32>
    %8__t217 = ntt.mul_modq %8__t215, %9__t216 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t218 = ntt.matmul_modq %6__t148, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t219 = ntt.const {name = "T1_T__r19"} : tensor<16x8xi32>
    %8__t220 = ntt.mul_modq %8__t218, %9__t219 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t221 = ntt.matmul_modq %6__t149, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t222 = ntt.const {name = "T1_T__r20"} : tensor<16x8xi32>
    %8__t223 = ntt.mul_modq %8__t221, %9__t222 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t224 = ntt.matmul_modq %6__t150, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t225 = ntt.const {name = "T1_T__r21"} : tensor<16x8xi32>
    %8__t226 = ntt.mul_modq %8__t224, %9__t225 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t227 = ntt.matmul_modq %6__t151, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t228 = ntt.const {name = "T1_T__r22"} : tensor<16x8xi32>
    %8__t229 = ntt.mul_modq %8__t227, %9__t228 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t230 = ntt.matmul_modq %6__t152, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t231 = ntt.const {name = "T1_T__r23"} : tensor<16x8xi32>
    %8__t232 = ntt.mul_modq %8__t230, %9__t231 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t233 = ntt.matmul_modq %6__t153, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t234 = ntt.const {name = "T1_T__r24"} : tensor<16x8xi32>
    %8__t235 = ntt.mul_modq %8__t233, %9__t234 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t236 = ntt.matmul_modq %6__t154, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t237 = ntt.const {name = "T1_T__r25"} : tensor<16x8xi32>
    %8__t238 = ntt.mul_modq %8__t236, %9__t237 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t239 = ntt.matmul_modq %6__t155, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t240 = ntt.const {name = "T1_T__r26"} : tensor<16x8xi32>
    %8__t241 = ntt.mul_modq %8__t239, %9__t240 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t242 = ntt.matmul_modq %6__t156, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t243 = ntt.const {name = "T1_T__r27"} : tensor<16x8xi32>
    %8__t244 = ntt.mul_modq %8__t242, %9__t243 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t245 = ntt.matmul_modq %6__t157, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t246 = ntt.const {name = "T1_T__r28"} : tensor<16x8xi32>
    %8__t247 = ntt.mul_modq %8__t245, %9__t246 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t248 = ntt.matmul_modq %6__t158, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t249 = ntt.const {name = "T1_T__r29"} : tensor<16x8xi32>
    %8__t250 = ntt.mul_modq %8__t248, %9__t249 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t251 = ntt.matmul_modq %6__t159, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t252 = ntt.const {name = "T1_T__r30"} : tensor<16x8xi32>
    %8__t253 = ntt.mul_modq %8__t251, %9__t252 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %8__t254 = ntt.matmul_modq %6__t160, %7 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %9__t255 = ntt.const {name = "T1_T__r31"} : tensor<16x8xi32>
    %8__t256 = ntt.mul_modq %8__t254, %9__t255 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %12__t257, %12__t258, %12__t259, %12__t260, %12__t261, %12__t262, %12__t263, %12__t264, %12__t265, %12__t266, %12__t267, %12__t268, %12__t269, %12__t270, %12__t271, %12__t272, %12__t273, %12__t274, %12__t275, %12__t276, %12__t277, %12__t278, %12__t279, %12__t280, %12__t281, %12__t282, %12__t283, %12__t284, %12__t285, %12__t286, %12__t287, %12__t288 = ntt.all_to_all %8__t163, %8__t166, %8__t169, %8__t172, %8__t175, %8__t178, %8__t181, %8__t184, %8__t187, %8__t190, %8__t193, %8__t196, %8__t199, %8__t202, %8__t205, %8__t208, %8__t211, %8__t214, %8__t217, %8__t220, %8__t223, %8__t226, %8__t229, %8__t232, %8__t235, %8__t238, %8__t241, %8__t244, %8__t247, %8__t250, %8__t253, %8__t256 {reshape_to = [8, 512], parts = 32 : index} : (tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>) -> (tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>)
    %13 = ntt.const {name = "W2_T"} : tensor<8x8xi32>
    %14__t289 = ntt.matmul_modq %12__t257, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t290 = ntt.const {name = "T2_T__r0"} : tensor<16x8xi32>
    %14__t291 = ntt.mul_modq %14__t289, %15__t290 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t292 = ntt.matmul_modq %12__t258, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t293 = ntt.const {name = "T2_T__r1"} : tensor<16x8xi32>
    %14__t294 = ntt.mul_modq %14__t292, %15__t293 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t295 = ntt.matmul_modq %12__t259, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t296 = ntt.const {name = "T2_T__r2"} : tensor<16x8xi32>
    %14__t297 = ntt.mul_modq %14__t295, %15__t296 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t298 = ntt.matmul_modq %12__t260, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t299 = ntt.const {name = "T2_T__r3"} : tensor<16x8xi32>
    %14__t300 = ntt.mul_modq %14__t298, %15__t299 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t301 = ntt.matmul_modq %12__t261, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t302 = ntt.const {name = "T2_T__r4"} : tensor<16x8xi32>
    %14__t303 = ntt.mul_modq %14__t301, %15__t302 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t304 = ntt.matmul_modq %12__t262, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t305 = ntt.const {name = "T2_T__r5"} : tensor<16x8xi32>
    %14__t306 = ntt.mul_modq %14__t304, %15__t305 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t307 = ntt.matmul_modq %12__t263, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t308 = ntt.const {name = "T2_T__r6"} : tensor<16x8xi32>
    %14__t309 = ntt.mul_modq %14__t307, %15__t308 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t310 = ntt.matmul_modq %12__t264, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t311 = ntt.const {name = "T2_T__r7"} : tensor<16x8xi32>
    %14__t312 = ntt.mul_modq %14__t310, %15__t311 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t313 = ntt.matmul_modq %12__t265, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t314 = ntt.const {name = "T2_T__r8"} : tensor<16x8xi32>
    %14__t315 = ntt.mul_modq %14__t313, %15__t314 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t316 = ntt.matmul_modq %12__t266, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t317 = ntt.const {name = "T2_T__r9"} : tensor<16x8xi32>
    %14__t318 = ntt.mul_modq %14__t316, %15__t317 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t319 = ntt.matmul_modq %12__t267, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t320 = ntt.const {name = "T2_T__r10"} : tensor<16x8xi32>
    %14__t321 = ntt.mul_modq %14__t319, %15__t320 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t322 = ntt.matmul_modq %12__t268, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t323 = ntt.const {name = "T2_T__r11"} : tensor<16x8xi32>
    %14__t324 = ntt.mul_modq %14__t322, %15__t323 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t325 = ntt.matmul_modq %12__t269, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t326 = ntt.const {name = "T2_T__r12"} : tensor<16x8xi32>
    %14__t327 = ntt.mul_modq %14__t325, %15__t326 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t328 = ntt.matmul_modq %12__t270, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t329 = ntt.const {name = "T2_T__r13"} : tensor<16x8xi32>
    %14__t330 = ntt.mul_modq %14__t328, %15__t329 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t331 = ntt.matmul_modq %12__t271, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t332 = ntt.const {name = "T2_T__r14"} : tensor<16x8xi32>
    %14__t333 = ntt.mul_modq %14__t331, %15__t332 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t334 = ntt.matmul_modq %12__t272, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t335 = ntt.const {name = "T2_T__r15"} : tensor<16x8xi32>
    %14__t336 = ntt.mul_modq %14__t334, %15__t335 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t337 = ntt.matmul_modq %12__t273, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t338 = ntt.const {name = "T2_T__r16"} : tensor<16x8xi32>
    %14__t339 = ntt.mul_modq %14__t337, %15__t338 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t340 = ntt.matmul_modq %12__t274, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t341 = ntt.const {name = "T2_T__r17"} : tensor<16x8xi32>
    %14__t342 = ntt.mul_modq %14__t340, %15__t341 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t343 = ntt.matmul_modq %12__t275, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t344 = ntt.const {name = "T2_T__r18"} : tensor<16x8xi32>
    %14__t345 = ntt.mul_modq %14__t343, %15__t344 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t346 = ntt.matmul_modq %12__t276, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t347 = ntt.const {name = "T2_T__r19"} : tensor<16x8xi32>
    %14__t348 = ntt.mul_modq %14__t346, %15__t347 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t349 = ntt.matmul_modq %12__t277, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t350 = ntt.const {name = "T2_T__r20"} : tensor<16x8xi32>
    %14__t351 = ntt.mul_modq %14__t349, %15__t350 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t352 = ntt.matmul_modq %12__t278, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t353 = ntt.const {name = "T2_T__r21"} : tensor<16x8xi32>
    %14__t354 = ntt.mul_modq %14__t352, %15__t353 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t355 = ntt.matmul_modq %12__t279, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t356 = ntt.const {name = "T2_T__r22"} : tensor<16x8xi32>
    %14__t357 = ntt.mul_modq %14__t355, %15__t356 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t358 = ntt.matmul_modq %12__t280, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t359 = ntt.const {name = "T2_T__r23"} : tensor<16x8xi32>
    %14__t360 = ntt.mul_modq %14__t358, %15__t359 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t361 = ntt.matmul_modq %12__t281, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t362 = ntt.const {name = "T2_T__r24"} : tensor<16x8xi32>
    %14__t363 = ntt.mul_modq %14__t361, %15__t362 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t364 = ntt.matmul_modq %12__t282, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t365 = ntt.const {name = "T2_T__r25"} : tensor<16x8xi32>
    %14__t366 = ntt.mul_modq %14__t364, %15__t365 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t367 = ntt.matmul_modq %12__t283, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t368 = ntt.const {name = "T2_T__r26"} : tensor<16x8xi32>
    %14__t369 = ntt.mul_modq %14__t367, %15__t368 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t370 = ntt.matmul_modq %12__t284, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t371 = ntt.const {name = "T2_T__r27"} : tensor<16x8xi32>
    %14__t372 = ntt.mul_modq %14__t370, %15__t371 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t373 = ntt.matmul_modq %12__t285, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t374 = ntt.const {name = "T2_T__r28"} : tensor<16x8xi32>
    %14__t375 = ntt.mul_modq %14__t373, %15__t374 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t376 = ntt.matmul_modq %12__t286, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t377 = ntt.const {name = "T2_T__r29"} : tensor<16x8xi32>
    %14__t378 = ntt.mul_modq %14__t376, %15__t377 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t379 = ntt.matmul_modq %12__t287, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t380 = ntt.const {name = "T2_T__r30"} : tensor<16x8xi32>
    %14__t381 = ntt.mul_modq %14__t379, %15__t380 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %14__t382 = ntt.matmul_modq %12__t288, %13 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %15__t383 = ntt.const {name = "T2_T__r31"} : tensor<16x8xi32>
    %14__t384 = ntt.mul_modq %14__t382, %15__t383 {q = 167772161 : i32} : (tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<16x8xi32>
    %18__t385, %18__t386, %18__t387, %18__t388, %18__t389, %18__t390, %18__t391, %18__t392, %18__t393, %18__t394, %18__t395, %18__t396, %18__t397, %18__t398, %18__t399, %18__t400, %18__t401, %18__t402, %18__t403, %18__t404, %18__t405, %18__t406, %18__t407, %18__t408, %18__t409, %18__t410, %18__t411, %18__t412, %18__t413, %18__t414, %18__t415, %18__t416 = ntt.all_to_all %14__t291, %14__t294, %14__t297, %14__t300, %14__t303, %14__t306, %14__t309, %14__t312, %14__t315, %14__t318, %14__t321, %14__t324, %14__t327, %14__t330, %14__t333, %14__t336, %14__t339, %14__t342, %14__t345, %14__t348, %14__t351, %14__t354, %14__t357, %14__t360, %14__t363, %14__t366, %14__t369, %14__t372, %14__t375, %14__t378, %14__t381, %14__t384 {reshape_to = [8, 512], parts = 32 : index} : (tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>) -> (tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>, tensor<8x16xi32>)
    %19 = ntt.const {name = "W3_f_T"} : tensor<8x8xi32>
    %20__t417 = ntt.matmul_modq %18__t385, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t418 = ntt.matmul_modq %18__t386, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t419 = ntt.matmul_modq %18__t387, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t420 = ntt.matmul_modq %18__t388, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t421 = ntt.matmul_modq %18__t389, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t422 = ntt.matmul_modq %18__t390, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t423 = ntt.matmul_modq %18__t391, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t424 = ntt.matmul_modq %18__t392, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t425 = ntt.matmul_modq %18__t393, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t426 = ntt.matmul_modq %18__t394, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t427 = ntt.matmul_modq %18__t395, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t428 = ntt.matmul_modq %18__t396, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t429 = ntt.matmul_modq %18__t397, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t430 = ntt.matmul_modq %18__t398, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t431 = ntt.matmul_modq %18__t399, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t432 = ntt.matmul_modq %18__t400, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t433 = ntt.matmul_modq %18__t401, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t434 = ntt.matmul_modq %18__t402, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t435 = ntt.matmul_modq %18__t403, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t436 = ntt.matmul_modq %18__t404, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t437 = ntt.matmul_modq %18__t405, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t438 = ntt.matmul_modq %18__t406, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t439 = ntt.matmul_modq %18__t407, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t440 = ntt.matmul_modq %18__t408, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t441 = ntt.matmul_modq %18__t409, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t442 = ntt.matmul_modq %18__t410, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t443 = ntt.matmul_modq %18__t411, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t444 = ntt.matmul_modq %18__t412, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t445 = ntt.matmul_modq %18__t413, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t446 = ntt.matmul_modq %18__t414, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t447 = ntt.matmul_modq %18__t415, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20__t448 = ntt.matmul_modq %18__t416, %19 {q = 167772161 : i32, transpose_a} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<16x8xi32>
    %20 = ntt.concat %20__t417, %20__t418, %20__t419, %20__t420, %20__t421, %20__t422, %20__t423, %20__t424, %20__t425, %20__t426, %20__t427, %20__t428, %20__t429, %20__t430, %20__t431, %20__t432, %20__t433, %20__t434, %20__t435, %20__t436, %20__t437, %20__t438, %20__t439, %20__t440, %20__t441, %20__t442, %20__t443, %20__t444, %20__t445, %20__t446, %20__t447, %20__t448 {axis = 0 : index} : (tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>) -> tensor<512x8xi32>
    %24 = ntt.reshape %20 : tensor<512x8xi32> to tensor<4096xi32>
    return %24 : tensor<4096xi32>
  }
}
