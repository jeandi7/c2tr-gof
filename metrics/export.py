# metrics/export.py

import csv


def init_gof_csv(filename: str):
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Session", "Mode", "Iteration", "Status",
            "Energy_J", "Comprehension", "Phi_E", "Ci", "CPJ", "Rg", "Pattern", "Error",
        ])


def append_gof_row(
    filename: str,
    session: str,
    mode: str,
    iteration: int,
    accepted: bool,
    energy_j: float,
    comprehension: int,
    phi_e: float,
    ci: float,
    cpj: float,
    rg: float,
    pattern: str,
    errors: list,
    warmup: bool = False,
):
    if warmup:
        status = "WARMUP"
    elif accepted:
        status = "ACCEPTED"
    else:
        status = "REJECTED"
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            session, mode, iteration,
            status,
            round(energy_j, 6),
            comprehension,
            round(phi_e, 8),
            round(ci, 6),
            round(cpj, 8),
            round(rg, 6),
            pattern,
            errors[0] if errors else "",
        ])
