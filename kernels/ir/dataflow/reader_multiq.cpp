// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for mul_modq_multiq: loads the 4 per-element modulus param tiles once (q, q_inv,
// q_low16, q_high16 -> c_2..c_5), then streams a[i]->c_0, b[i]->c_1 for i in [0,n_tiles).
// Args: 0=a,1=b,2=q,3=q_inv,4=q_low,5=q_high,6=n_tiles.
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t qi_addr = get_arg_val<uint32_t>(3);
    const uint32_t ql_addr = get_arg_val<uint32_t>(4);
    const uint32_t qh_addr = get_arg_val<uint32_t>(5);
    const uint32_t n = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_q = tt::CBIndex::c_2, cb_qi = tt::CBIndex::c_3, cb_ql = tt::CBIndex::c_4, cb_qh = tt::CBIndex::c_5;
    const uint32_t ts = get_tile_size(cb_a);
    auto gen = [&](uint32_t addr) { return InterleavedAddrGenFast<true>{.bank_base_address = addr, .page_size = ts, .data_format = DataFormat::UInt32}; };
    auto load1 = [&](uint32_t cb, uint32_t addr) {
        cb_reserve_back(cb, 1); auto g = gen(addr); noc_async_read_tile(0, g, get_write_ptr(cb)); noc_async_read_barrier(); cb_push_back(cb, 1);
    };
    load1(cb_q, q_addr); load1(cb_qi, qi_addr); load1(cb_ql, ql_addr); load1(cb_qh, qh_addr);

    const auto ga = gen(a_addr), gb = gen(b_addr);
    for (uint32_t i = 0; i < n; ++i) {
        cb_reserve_back(cb_a, 1); cb_reserve_back(cb_b, 1);
        noc_async_read_tile(i, ga, get_write_ptr(cb_a));
        noc_async_read_tile(i, gb, get_write_ptr(cb_b));
        noc_async_read_barrier();
        cb_push_back(cb_a, 1); cb_push_back(cb_b, 1);
    }
}
