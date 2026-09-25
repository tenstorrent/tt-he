#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
r"""Generate the demo text database: one row per line, ASCII only.

Row format:  <5-digit index> <Company> - <sentence about that company>

The index leads the row on purpose: a retrieved row is then SELF-IDENTIFYING, so the demo can print
what came back and you can see the index matches what was asked for without trusting the plumbing.
ASCII only because the PIR plaintext modulus is t=257 and each polynomial coefficient carries exactly
one character (see docs/pir/PIR65536_DEMO.md §0).

    python3 gen_db_text.py                      # 65536 rows -> db65536.txt
    python3 gen_db_text.py --rows 1024 --out small.txt
"""
import argparse, random

# well-known companies, cycled so every row names one
COMPANIES = [
    "Tenstorrent", "Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla", "Nvidia", "Intel", "AMD",
    "IBM", "Oracle", "Samsung", "Sony", "Toyota", "Honda", "Siemens", "Bosch", "Philips", "Nokia",
    "Ericsson", "Qualcomm", "Broadcom", "Cisco", "Dell", "HP", "Lenovo", "Adobe", "Salesforce", "SAP",
    "Netflix", "Spotify", "Uber", "Airbnb", "Stripe", "Shopify", "Square", "PayPal", "Visa", "Mastercard",
    "Goldman Sachs", "JPMorgan", "BlackRock", "Berkshire Hathaway", "Walmart", "Costco", "Target", "IKEA",
    "Nike", "Adidas", "Boeing", "Airbus", "SpaceX", "Lockheed Martin", "General Electric", "3M",
    "Johnson & Johnson", "Pfizer", "Moderna", "Roche", "Novartis", "AstraZeneca", "Unilever", "Nestle",
    "Coca-Cola", "PepsiCo", "McDonalds", "Starbucks", "Disney", "Warner Bros", "Comcast", "Verizon",
    "AT&T", "Vodafone", "Shell", "BP", "ExxonMobil", "Chevron", "Maersk", "FedEx", "UPS", "DHL",
    "Volkswagen", "BMW", "Mercedes-Benz", "Ferrari", "Porsche", "Hyundai", "Kia", "Michelin",
    "TSMC", "ASML", "Applied Materials", "Micron", "Western Digital", "Seagate", "Arm", "Cadence",
    "Synopsys", "Palantir", "Snowflake", "Databricks", "OpenAI", "Anthropic", "DeepMind", "Cloudflare",
]

TEMPLATES = [
    "{c} was founded by people who wanted to build something that had not existed before.",
    "{c} employs engineers across many countries and ships products used every single day.",
    "{c} is often cited in business schools as a case study in scaling an organisation.",
    "{c} spends a large share of its revenue on research and long horizon development.",
    "{c} became widely known after a product launch that changed customer expectations.",
    "{c} operates supply chains that span several continents and thousands of suppliers.",
    "{c} publishes an annual report that analysts read closely for signals about the sector.",
    "{c} has weathered downturns by reinvesting when competitors chose to retrench.",
    "{c} maintains research labs whose output appears in peer reviewed conferences.",
    "{c} built its reputation on reliability rather than on being first to market.",
    "{c} runs data centres whose power draw is measured in tens of megawatts.",
    "{c} counts governments, universities and small businesses among its customers.",
    "{c} is frequently named when people discuss the future of its industry.",
    "{c} has an engineering culture that prizes measurement over intuition.",
    "{c} learned expensive lessons from an early product that did not find its market.",
    "{c} contributes to open standards that competitors also depend on.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=65536, help="database rows (hypercube capacity)")
    ap.add_argument("--maxlen", type=int, default=2048, help="max chars/row = polynomial degree n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="db65536.txt")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    idx_w = max(5, len(str(a.rows - 1)))
    longest = 0
    with open(a.out, "w", encoding="ascii") as f:
        for i in range(a.rows):
            comp = COMPANIES[i % len(COMPANIES)]
            line = f"{i:0{idx_w}d} {comp} - {rng.choice(TEMPLATES).format(c=comp)}"
            # keep every row inside one plaintext polynomial, and ASCII-clean
            line = line.encode("ascii", "replace").decode("ascii")[: a.maxlen]
            longest = max(longest, len(line))
            f.write(line + "\n")

    print(f"wrote {a.out}: {a.rows} rows, longest={longest} chars (limit {a.maxlen}), ASCII only")
    print(f"  row 0     : {open(a.out).readline().rstrip()}")

if __name__ == "__main__":
    main()
