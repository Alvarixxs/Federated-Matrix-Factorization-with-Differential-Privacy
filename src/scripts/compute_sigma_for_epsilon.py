import argparse
import json
import os
from opacus.accountants import RDPAccountant


def compute_epsilon(sigma, sample_rate, rounds, delta):
    """Calcula el epsilon para una configuración dada."""
    accountant = RDPAccountant()
    for _ in range(rounds):
        accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
    eps, _ = accountant.get_privacy_spent(delta=delta)
    return eps


def binary_search_sigma(target_eps, sample_rate, rounds, delta, tol=0.01,
                        sigma_low=0.1, sigma_high=100.0, max_iter=100):
    """Busca el sigma tal que epsilon = target_eps mediante bisección."""

    # Comprobar límites
    eps_low = compute_epsilon(sigma_low, sample_rate, rounds, delta)
    eps_high = compute_epsilon(sigma_high, sample_rate, rounds, delta)

    if eps_low < target_eps:
        # Ni con el sigma mínimo conseguimos eps > target — no hay solución
        return None
    if eps_high > target_eps:
        # Ni con el sigma máximo conseguimos eps < target — aumentar sigma_high
        return None

    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        eps_mid = compute_epsilon(sigma_mid, sample_rate, rounds, delta)

        if abs(eps_mid - target_eps) / target_eps < tol:
            return sigma_mid

        if eps_mid > target_eps:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid

    return sigma_mid


def main(args):
    results = {}

    for n_clients in args.n_clients_list:
        effective_sample_rate = args.sample_rate
        # Nota: sample_rate se mantiene igual — controla la probabilidad
        # de selección, no cambia con |C|

        results[n_clients] = {}
        for target_eps in args.target_epsilons:
            sigma = binary_search_sigma(
                target_eps=target_eps,
                sample_rate=effective_sample_rate,
                rounds=args.rounds,
                delta=args.delta,
            )
            results[n_clients][target_eps] = sigma
            status = f"{sigma:.4f}" if sigma is not None else "no viable"
            print(f"  |C|={n_clients}, eps={target_eps}: sigma = {status}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResultados guardados en {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_clients_list", type=int, nargs="+", required=True)
    ap.add_argument("--target_epsilons", type=float, nargs="+", required=True)
    ap.add_argument("--sample_rate", type=float, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    main(args)