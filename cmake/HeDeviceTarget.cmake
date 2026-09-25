# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# HeDeviceTarget.cmake — one helper for the repo's device programs.
#
# Every device target is the same three steps: an executable, link TT::Metalium, and point
# OVERRIDE_KERNEL_PREFIX at the kernel directory it JIT-compiles from. Written out longhand that is
# ~4 lines x 71 targets of identical boilerplate, where the only thing that varies is the kernel
# directory and the occasional extra define.
#
#   he_device_target(vector_modmul SOURCES host/tests/vector/vector_modmul.cpp TEST hw)
#   he_device_target(test_ir_rns   SOURCES host/tests/ir/test_ir_rns.cpp KERNELS kernels/ir TEST hw)
#   he_device_target(he_demo       SOURCES host/examples/he_demo.cpp host/lib/he/he_api.cpp TEST hw)
#   he_device_target(bfv_relin_q3  SOURCES host/tests/he/bfv_relin.cpp DEFINES CFG_LQ=3 CFG_LP=3 TEST hw)
#   he_device_target(ckks_ctmul    SOURCES host/benchmarks/ckks_ctmul.cpp TEST perf)
#
# OVERRIDE_KERNEL_PREFIX must keep its trailing slash: kernels are located by string concatenation
# in the host code, so dropping it fails at JIT time on device, not at build time.
#
# TEST <label...> additionally registers the target with ctest under those labels. One label per
# host/ subfolder, so where a program lives decides when it runs:
#   hw      host/tests/       correctness -- the gate; `ctest -L '^hw$' -LE '^fixture$'`
#   perf    host/benchmarks/  THE benchmark, ckks_ctmul; `ctest -L '^perf$'`
#   study   host/experiments/ finished exp_* studies + legacy/; `ctest -L '^study$'`
#   nohw    host-only, no device
#   fixture needs a generated artifact the repo does NOT ship -- only the four PIR tests that
#           want db65536.pir / db65536.fold / query_small.bin. Every other case file IS tracked,
#           so those tests belong in the gate; 19 carried this label wrongly and were skipped.
# Anchor the selectors: a bare `-L hw` also matches `nohw`. tools/run_all.sh wraps all of this.
# ARGS <arg...> are passed to the program when ctest runs it (e.g. a specific fixture file).

function(he_device_target name)
    # Host-only configure (CI without hardware): device targets are skipped entirely, so the project
    # configures and builds with no tt-metalium present.
    if(HE_HOST_ONLY)
        return()
    endif()
    # NB: TEST is multi-value. It must not be a one-value keyword: ${ARGN} expands unquoted, so a
    # quoted "hw;fixture" would be split and the keyword would silently capture only "hw".
    cmake_parse_arguments(HE "" "KERNELS;KERNELS_IR" "SOURCES;DEFINES;TEST;ARGS" ${ARGN})

    if(NOT HE_SOURCES)
        message(FATAL_ERROR "he_device_target(${name}): SOURCES is required")
    endif()
    # The validated tree. Experiment trees are selected explicitly with KERNELS
    # (kernels/experiments/<exp>); kernels/ir is the compiler/PIR tree.
    if(NOT HE_KERNELS)
        set(HE_KERNELS "kernels/shared")
    endif()
    # KERNELS is resolved against HE_KERNELS_BASE (the caller sets it; defaults to this directory),
    # so the kernel tree can live anywhere relative to the CMakeLists that defines the targets.
    if(NOT DEFINED HE_KERNELS_BASE)
        set(HE_KERNELS_BASE "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    get_filename_component(_he_kdir "${HE_KERNELS_BASE}/${HE_KERNELS}" ABSOLUTE)
    if(NOT IS_DIRECTORY "${_he_kdir}")
        message(FATAL_ERROR
            "he_device_target(${name}): kernel dir '${_he_kdir}' does not exist. "
            "A wrong OVERRIDE_KERNEL_PREFIX only fails at runtime, so this is checked here.")
    endif()

    # KERNELS_IR: a second, independent kernel root. The PIR engine resolves some kernels through
    # OVERRIDE_KERNEL_PREFIX and others through OVERRIDE_KERNEL_PREFIX_IR, and both macros have empty
    # fallbacks in the header -- so a target that needs the IR root but omits it COMPILES CLEANLY and then
    # dies at the first scan_mtile, after expand has already run. Same IS_DIRECTORY guard as above, for
    # the same reason: a wrong prefix is otherwise a runtime-only failure.
    if(HE_KERNELS_IR)
        get_filename_component(_he_kdir_ir "${HE_KERNELS_BASE}/${HE_KERNELS_IR}" ABSOLUTE)
        if(NOT IS_DIRECTORY "${_he_kdir_ir}")
            message(FATAL_ERROR
                "he_device_target(${name}): IR kernel dir '${_he_kdir_ir}' does not exist. "
                "A wrong OVERRIDE_KERNEL_PREFIX_IR only fails at runtime, so this is checked here.")
        endif()
    endif()

    add_executable(${name} ${HE_SOURCES})
    target_link_libraries(${name} PRIVATE TT::Metalium)
    # Repo-root-relative includes (e.g. "cpu/montgomery.hpp") plus any shared-header dirs, so a
    # source file's depth in the tree does not decide whether its #includes resolve.
    if(DEFINED HE_INCLUDE_DIRS)
        target_include_directories(${name} PRIVATE ${HE_INCLUDE_DIRS})
    endif()
    target_compile_definitions(${name} PRIVATE
        OVERRIDE_KERNEL_PREFIX="${_he_kdir}/"
        ${HE_DEFINES}
    )
    if(HE_KERNELS_IR)
        target_compile_definitions(${name} PRIVATE OVERRIDE_KERNEL_PREFIX_IR="${_he_kdir_ir}/")
    endif()

    if(HE_TEST)
        # Run from HE_RUN_DIR: several programs load .bin fixtures by a path relative to their cwd,
        # so this must be the directory those fixtures sit in, not wherever CMakeLists happens to live.
        if(NOT DEFINED HE_RUN_DIR)
            set(HE_RUN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        endif()
        # ARGS matters for targets that share a source but differ by compile-time defines: they
        # cannot all default to the same fixture path, so the right one is named here.
        add_test(NAME ${name} COMMAND ${name} ${HE_ARGS} WORKING_DIRECTORY ${HE_RUN_DIR})
        set_tests_properties(${name} PROPERTIES LABELS "${HE_TEST}" TIMEOUT 300)
    endif()
endfunction()
