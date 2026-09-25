// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal BFV homomorphic-encryption API — hides all TT-Metalium device control.
// Encrypt integers, do HE add / HE multiply ON THE TENSTORRENT DEVICE, decrypt.
// (Messages are encoded as constant polynomials, so results live in coefficient 0;
//  keep values so that a*b < t = 12289.)
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace he {

// A ciphertext is opaque to the caller (RNS residues live inside). Copyable/movable.
struct Ciphertext {
    std::vector<std::vector<uint32_t>> c0, c1;   // base-Q residues (internal representation)
};

class HEContext {
public:
    explicit HEContext(uint64_t seed = 0);       // params + keygen + open device
    ~HEContext();
    HEContext(HEContext&&) noexcept;

    Ciphertext encrypt(int64_t value);           // host
    int64_t    decrypt(const Ciphertext& ct);    // host

    Ciphertext add(const Ciphertext& a, const Ciphertext& b);        // ON DEVICE (HE add)
    Ciphertext multiply(const Ciphertext& a, const Ciphertext& b);   // ON DEVICE (tensor→rescale→relin)

    uint32_t plaintext_modulus() const;          // t

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace he
