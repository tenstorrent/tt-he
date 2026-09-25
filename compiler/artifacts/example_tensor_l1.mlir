module {
  func.func @tensor_l1(%c0_0: tensor<16xi32>, %c1_0: tensor<16xi32>, %d0_0: tensor<16xi32>, %d1_0: tensor<16xi32>, %c0_1: tensor<16xi32>, %c1_1: tensor<16xi32>, %d0_1: tensor<16xi32>, %d1_1: tensor<16xi32>, %c0_2: tensor<16xi32>, %c1_2: tensor<16xi32>, %d0_2: tensor<16xi32>, %d1_2: tensor<16xi32>, %c0_3: tensor<16xi32>, %c1_3: tensor<16xi32>, %d0_3: tensor<16xi32>, %d1_3: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %C0_0 = rns.ntt %c0_0 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C1_0 = rns.ntt %c1_0 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %D0_0 = rns.ntt %d0_0 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %D1_0 = rns.ntt %d1_0 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %E0_0 = rns.mul %C0_0, %D0_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E2_0 = rns.mul %C1_0, %D1_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T0_0 = rns.mul %C0_0, %D1_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T1_0 = rns.mul %C1_0, %D0_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E1_0 = rns.add %T0_0, %T1_0 {q = 268435361 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_0 = rns.intt %E0_0 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_0 = rns.intt %E1_0 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_0 = rns.intt %E2_0 {N = 16 : i64, q = 268435361 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C0_1 = rns.ntt %c0_1 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C1_1 = rns.ntt %c1_1 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %D0_1 = rns.ntt %d0_1 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %D1_1 = rns.ntt %d1_1 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %E0_1 = rns.mul %C0_1, %D0_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E2_1 = rns.mul %C1_1, %D1_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T0_1 = rns.mul %C0_1, %D1_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T1_1 = rns.mul %C1_1, %D0_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E1_1 = rns.add %T0_1, %T1_1 {q = 268435009 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_1 = rns.intt %E0_1 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_1 = rns.intt %E1_1 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_1 = rns.intt %E2_1 {N = 16 : i64, q = 268435009 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C0_2 = rns.ntt %c0_2 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C1_2 = rns.ntt %c1_2 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %D0_2 = rns.ntt %d0_2 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %D1_2 = rns.ntt %d1_2 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %E0_2 = rns.mul %C0_2, %D0_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E2_2 = rns.mul %C1_2, %D1_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T0_2 = rns.mul %C0_2, %D1_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T1_2 = rns.mul %C1_2, %D0_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E1_2 = rns.add %T0_2, %T1_2 {q = 268434977 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_2 = rns.intt %E0_2 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_2 = rns.intt %E1_2 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_2 = rns.intt %E2_2 {N = 16 : i64, q = 268434977 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C0_3 = rns.ntt %c0_3 {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %C1_3 = rns.ntt %c1_3 {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %D0_3 = rns.ntt %d0_3 {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %D1_3 = rns.ntt %d1_3 {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %E0_3 = rns.mul %C0_3, %D0_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E2_3 = rns.mul %C1_3, %D1_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T0_3 = rns.mul %C0_3, %D1_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %T1_3 = rns.mul %C1_3, %D0_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %E1_3 = rns.add %T0_3, %T1_3 {q = 268434721 : i64} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %e0_3 = rns.intt %E0_3 {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e1_3 = rns.intt %E1_3 {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    %e2_3 = rns.intt %E2_3 {N = 16 : i64, q = 268434721 : i64} : (tensor<16xi32>) -> tensor<16xi32>
    return %e0_0, %e0_1, %e0_2, %e0_3, %e1_0, %e1_1, %e1_2, %e1_3, %e2_0, %e2_1, %e2_2, %e2_3 : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }
}
