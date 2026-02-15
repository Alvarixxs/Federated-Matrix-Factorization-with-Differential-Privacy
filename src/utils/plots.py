import os


def build_results_path(args):
    return os.path.join(
        "results",
        args.dataset,
        f"k{args.k}_"
        f"lr{args.lr}_"
        f"reg{args.reg}_"
        f"bs{args.batch_size}_"
        f"r{args.rounds}",
    )

def save_figure(fig, save_path, name):
    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, f"{name}.png")
    fig.savefig(fig_path)