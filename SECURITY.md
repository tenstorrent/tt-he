# Security Policy

## Cryptographic status — read this first

**This project is research and engineering work, not a production cryptographic
library.** It implements BFV homomorphic encryption and PIR primitives to
demonstrate and measure what Tenstorrent hardware can do with them. It has not
been audited, it makes no constant-time or side-channel guarantees, and its
parameter sets were chosen to exercise the hardware rather than to meet a
specific security level. Do not use it to protect real secrets.

Reports about the *implementation* — a kernel that computes the wrong thing, a
key or plaintext that leaks through an API, a memory-safety defect — are in
scope and welcome. Reports that the project lacks a property it never claimed
(constant-time execution, a formally analyzed parameter set) are better raised
as a GitHub Issue than as a vulnerability report.

The `main` branch is the only supported version. Fixes land there; there are no
backports to tags or older commits.

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
responsibly using GitHub's private vulnerability reporting feature.

**Do not report security vulnerabilities through public GitHub issues.**

### How to Report

1. Navigate to the **Security** tab of this repository
2. Click **"Report a vulnerability"**
3. Provide a detailed description of the vulnerability, including:
   - Steps to reproduce the issue
   - Potential impact
   - Any suggested fixes (if applicable)

For this project specifically, please also name the affected file or kernel and
the conditions needed to trigger the issue.

For detailed instructions on privately reporting a security vulnerability, see [GitHub's documentation](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-a-vulnerability/privately-reporting-a-security-vulnerability).

### Questions or Discussion

If you have questions about the vulnerability or need to start a conversation
about it, please contact **ospo@tenstorrent.com** directly.

## Our Security Process

1. **Report**: Submit a vulnerability report through GitHub's Security tab
2. **Acknowledgment**: Tenstorrent will respond within **2 business days**
3. **Triage**: Our team will assess the issue and update its priority and risk level
4. **Fix Development**: A fix will be developed in a private branch, with reporter feedback when possible
5. **Disclosure**: A security advisory will be published once the fix is patched and merged to the main branch

If we conclude a report is not a vulnerability, we will say so and explain why,
rather than letting it go quiet.

## Scope

In scope: everything in this repository — device kernels under `kernels/`, the
library surface under `host/lib/`, the host programs under `host/examples/`,
`host/tests/`, `host/benchmarks/` and `host/experiments/`, the compiler under
`compiler/`, and the Python reference models under `reference/`.

Out of scope: vulnerabilities in tt-metalium or other dependencies. Report those
to their own projects — for tt-metalium, see
<https://github.com/tenstorrent/tt-metal/security>.

Thank you for helping keep this project and our users safe!
