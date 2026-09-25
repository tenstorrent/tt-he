// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BFV homomorphic-encryption demo on Tenstorrent.
//   encrypt two integers  ->  HE add & HE multiply ON DEVICE  ->  decrypt  ->  print.
// All device/TT-Metalium control is hidden behind he::HEContext (see he_api.hpp / he_api.cpp).
//   ./build/he_demo [a] [b]
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "he_api.hpp"

int main(int argc, char** argv) {
    // Quiet the TT-Metalium device logs so the demo output is clean. The logger initializes at
    // library load (before main) and reads TT_METAL_LOGGER_LEVEL from the environment, so we set it
    // and re-exec ourselves once for it to take effect. Set HE_VERBOSE=1 to keep the full logs.
    if (!std::getenv("HE_VERBOSE") && !std::getenv("HE_QUIET_")) {
        setenv("TT_METAL_LOGGER_LEVEL", "FATAL", 1);
        setenv("HE_QUIET_", "1", 1);
        execv("/proc/self/exe", argv);   // env inherited; on return execv failed — fall through
    }

    int64_t a = (argc > 1) ? std::atoll(argv[1]) : 6;
    int64_t b = (argc > 2) ? std::atoll(argv[2]) : 7;

    printf("\n===== BFV homomorphic encryption on Tenstorrent =====\n\n");
    printf("plaintext inputs:      a = %lld ,  b = %lld\n", (long long)a, (long long)b);

    uint64_t seed = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 0;
    he::HEContext he(seed);
    const uint32_t t = he.plaintext_modulus();
    printf("scheme:                BFV, N=1024, ciphertext modulus ~84-bit, plaintext modulus t=%u\n\n", t);

    printf("[1] encrypting a and b into ciphertexts ...\n");
    he::Ciphertext ca = he.encrypt(a);
    he::Ciphertext cb = he.encrypt(b);
    printf("    ct_a, ct_b ready (each a pair of polynomials in R_q).\n\n");

    printf("[2] HE ADD on device:       ct_sum = ct_a (+) ct_b\n");
    he::Ciphertext csum = he.add(ca, cb);

    printf("[3] HE MULTIPLY on device:  ct_prod = ct_a (x) ct_b   (tensor -> rescale -> relinearize)\n\n");
    he::Ciphertext cprod = he.multiply(ca, cb);

    printf("[4] decrypting results ...\n\n");
    int64_t dec_sum  = he.decrypt(csum);
    int64_t dec_prod = he.decrypt(cprod);

    printf("-----------------------------------------------------\n");
    printf("  decrypt(ct_a (+) ct_b) = %-6lld   (expected %lld)   %s\n",
           (long long)dec_sum,  (long long)((a + b) % t),
           dec_sum  == (a + b) % t ? "OK" : "MISMATCH");
    printf("  decrypt(ct_a (x) ct_b) = %-6lld   (expected %lld)   %s\n",
           (long long)dec_prod, (long long)((a * b) % t),
           dec_prod == (a * b) % t ? "OK" : "MISMATCH");
    printf("-----------------------------------------------------\n\n");

    bool ok = (dec_sum == (a + b) % t) && (dec_prod == (a * b) % t);
    printf("%s\n\n", ok ? "All homomorphic operations decrypted correctly." : "FAILED.");
    return ok ? 0 : 1;
}
