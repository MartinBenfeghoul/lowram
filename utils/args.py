import argparse


def list_of_strings(arg):
    return arg.split(",")

def parse_layers(layer_spec: str | None, num_layers: int) -> list[int]:
    if layer_spec is None or layer_spec == "all":
        return list(range(num_layers))

    layers: set[int] = set()
    for part in layer_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", maxsplit=1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))

    bad = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if bad:
        raise ValueError(f"Layer indices out of range: {bad}")
    return sorted(layers)


def add_common_training_args(
    parser,
    *,
    model_name: str,
    seq_len: int,
    max_batches: int,
    lr: float = 1e-3,
    batch_size: int | None = None,
):
    """Add the arguments every training script shares.

    Defaults are per-script, so each caller passes its own.
    """
    parser.add_argument("--model_name", type=str, default=model_name)
    parser.add_argument("--seq_len", type=int, default=seq_len)
    parser.add_argument("--max_batches", type=int, default=max_batches)
    parser.add_argument("--lr", type=float, default=lr)
    if batch_size is not None:
        parser.add_argument("--batch_size", type=int, default=batch_size)
    return parser


def add_cache_args(parser):
    """Add the key/value cache flags shared by the evaluation scripts."""
    # key cache
    parser.add_argument(
        "-kc",
        "--k_cache_type",
        choices=["baseline", "xkv", "turboquant"],
        default="xkv",
    )
    parser.add_argument("-r", "--comp_ratio", type=float, default=2.0)
    parser.add_argument(
        "--xkv_layer_group_size",
        type=int,
        default=4,
        help="Number of adjacent layers to jointly compress when --k_cache_type=xkv.",
    )
    parser.add_argument(
        "--xkv_svd_backend",
        choices=["cholqr", "linalg"],
        default="cholqr",
    )
    parser.add_argument("--selective_reconstruction", action="store_true")
    parser.add_argument(
        "--k_quantise_a",
        action="store_true",
        help="Quantise low-rank key-cache A factors with TurboQuant.",
    )
    parser.add_argument(
        "--k_quantise_b",
        action="store_true",
        help="Quantise low-rank key-cache B factors with TurboQuant.",
    )
    parser.add_argument(
        "--k_compressor_bits",
        type=int,
        default=4,
        help="Bits per rotated coordinate for TurboQuant key cache or low-rank factor quantisation.",
    )

    # value cache
    parser.add_argument(
        "-vc",
        "--v_cache_type",
        type=str,
        default="mlp",
        help="Value cache backend: baseline, mlp, linear, xkv, or turboquant.",
    )
    parser.add_argument(
        "--target_cr",
        type=float,
        default=None,
        help="Sets MLP or linear value-cache target compression ratio.",
    )
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument(
        "--v_residual_sink_tokens",
        type=int,
        default=0,
        help="Reserve MLP residual rows for the first N non-padding tokens per sequence and KV head.",
    )
    parser.add_argument("--meta_weights_path", type=str, default=None)
    parser.add_argument(
        "--value_mlp_weights_path",
        type=str,
        default=None,
        help="Path to plain pretrained value-cache MLP weights from train_value_mlps.py.",
    )
    parser.add_argument(
        "--use_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a linear residual W_linear to the MLP, initialised as pinv(W_k) @ W_v from the model's projection weights. Use --no-use_residual to disable.",
    )
    parser.add_argument(
        "--attn_predictor_path",
        type=str,
        default=None,
        help="Path to a checkpoint from train_attention_predictor.py. Providing it enables the shared CNN attention predictor that guides value residual selection.",
    )
    parser.add_argument(
        "--eviction_keep_ratio",
        type=float,
        default=1.0,
        help="Fraction of prefill KV tokens to retain per layer. Values below 1 enable attention-predictor physical eviction.",
    )
    parser.add_argument(
        "--v_turboquant_residuals",
        action="store_true",
        help="Quantise stored MLP value residuals with TurboQuant.",
    )
    parser.add_argument(
        "--v_compressor_bits",
        type=int,
        default=2,
        help="Bits per rotated residual coordinate for TurboQuant residual coding.",
    )
    return parser


def validate_cache_args(parser, args) -> None:
    """Reject flag combinations argparse cannot express on its own."""
    if args.v_cache_type in ("mlp", "linear") and args.target_cr is None:
        parser.error(
            f"--target_cr is required when --v_cache_type={args.v_cache_type}."
        )
    if args.eviction_keep_ratio < 1 and args.attn_predictor_path is None:
        parser.error(
            "--eviction_keep_ratio < 1 requires --attn_predictor_path."
        )
    if (
        args.meta_weights_path is not None
        and args.value_mlp_weights_path is not None
    ):
        parser.error(
            "--meta_weights_path and --value_mlp_weights_path are mutually "
            "exclusive."
        )
