import os


def build_base_path(args, model):
    path_parts = [
        "experiments",
        args.dataset,
        f"k{args.k}_"
        f"lr{args.lr}_"
        f"reg{args.reg}_"
        f"bs{args.batch_size}_"
        f"r{args.rounds}",
        f"seed{args.seed}",
        model,
    ]

    if model in ("FL", "FL_DP"):
        path_parts.append(f"le{args.local_epochs}")
        if getattr(args, "n_clients", None) is not None:
            path_parts.append(f"nc{args.n_clients}")

    if model == "FL_DP":
        path_parts.append(f"noise{args.noise_multiplier}")

    return os.path.join(*path_parts)