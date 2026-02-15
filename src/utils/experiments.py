
import os


def build_base_path(args, model):
    return os.path.join(
        "experiments",
        args.dataset,
        f"k{args.k}_"
        f"lr{args.lr}_"
        f"reg{args.reg}_"
        f"bs{args.batch_size}_"
        f"r{args.rounds}",
        model,
        f"noise{args.noise_multiplier}" if model == "FL_DP" else ""
    )