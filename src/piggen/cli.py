import logging

import click

from .model import pIgGen

log_format = "%(asctime)s - %(levelname)s - %(message)s"
date_format = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@click.command()
@click.option("--developable", is_flag=True, help="Use the developable model")
@click.option(
    "--heavy_chain_file",
    default=None,
    help="File containing heavy chain sequences to generate from.",
)
@click.option(
    "--light_chain_file",
    default=None,
    help="File containing light chain sequences to generate heavy chains from.",
)
@click.option(
    "--initial_sequence", default=None, help="Initial sequence to generate from."
)
@click.option(
    "--n_sequences",
    default=1,
    help="Number of sequences to generate, per input sequence if applicable.",
)
@click.option("--top_p", default=0.95, help="Top-p sampling value")
@click.option("--temp", default=1.2, help="Temperature for generation")
@click.option(
    "--bottom_n_percent",
    default=5,
    type=int,
    help="""Bottom n percent of sequences to discard based on likelihood.
        Only applicable if n_sequences > 100.""",
)
@click.option("--backwards", is_flag=True, help="Generate sequences in reverse")
@click.option(
    "--output_file",
    default=None,
    help="File to save the generated sequences",
    required=True,
)
@click.option(
    "--separate_chains",
    is_flag=True,
    help="Output VH and VL sequences separately, requires anarci.",
)
@click.option(
    "--cache_dir",
    default=None,
    help="Directory to cache the model in if not using default, handled by huggingface.",
)
@click.option(
    "--device",
    default=None,
    help="Device to run the model on (e.g., 'cuda', 'mps', 'cpu'). Auto-detects if not specified.",
)
def generate(
    developable,
    heavy_chain_file,
    light_chain_file,
    initial_sequence,
    n_sequences,
    top_p,
    temp,
    bottom_n_percent,
    backwards,
    output_file,
    separate_chains,
    cache_dir,
    device,
):
    model_name = (
        "ollieturnbull/p-IgGen-developable" if developable else "ollieturnbull/p-IgGen"
    )
    model = pIgGen(model_name=model_name, cache_dir=cache_dir, device=device)
    if heavy_chain_file and light_chain_file:
        raise click.UsageError(
            "Specify only one of --heavy_chain_file or --light_chain_file."
        )

    sequences = []

    if heavy_chain_file:
        with open(heavy_chain_file) as f:
            heavy_chains = [line.strip() for line in f]
        for i, heavy_chain in enumerate(heavy_chains):
            if heavy_chain:  # Skip empty lines
                generated = model.generate_light_chain(
                    heavy_chain=heavy_chain,
                    num_return_sequences=n_sequences,
                    top_p=top_p,
                    temp=temp,
                    batch_size=1,
                )
                # Extract light chains and pair with index
                generated_light_chains = [seq[len(heavy_chain) :] for seq in generated]
                sequences.extend(
                    [(i, light_seq) for light_seq in generated_light_chains]
                )

    elif light_chain_file:
        with open(light_chain_file) as f:
            light_chains = [line.strip() for line in f]
        for i, light_chain in enumerate(light_chains):
            if light_chain:  # Skip empty lines
                generated = model.generate_heavy_chain(
                    light_chain=light_chain,
                    num_return_sequences=n_sequences,
                    top_p=top_p,
                    temp=temp,
                    batch_size=1,
                )
                # Extract heavy chains and pair with index
                generated_heavy_chains = [seq[: -len(light_chain)] for seq in generated]
                sequences.extend(
                    [(i, heavy_seq) for heavy_seq in generated_heavy_chains]
                )

    else:
        sequences = model.generate(
            num_return_sequences=n_sequences,
            top_p=top_p,
            temp=temp,
            batch_size=1,
            prompt=initial_sequence,
            discard_bottom_n_percent=bottom_n_percent,
            separated_output=separate_chains,
            backwards=backwards,
        )

    with open(output_file, "w") as f:
        for seq in sequences:
            if light_chain_file or heavy_chain_file:
                # write out the index of the prompted heavy chain
                f.write(str(seq[0]) + ", " + seq[1] + "\n")
            else:
                # if separated output, write out the VH and VL sequences
                if separate_chains:
                    f.write(seq[0] + ", " + seq[1] + "\n")
                else:
                    f.write(seq + "\n")


@click.command()
@click.option("--developable", is_flag=True, help="Use the developable model")
@click.option(
    "--sequence_file",
    required=True,
    help="""File containing sequences to calculate log likelihoods,
        separated by new lines.""",
)
@click.option("--batch_size", default=1, help="Batch size for processing sequences")
@click.option("--output_file", required=True, help="File to save the log likelihoods")
@click.option(
    "--cache_dir",
    default=None,
    help="Directory to cache the model in if not using default, handled by huggingface.",
)
@click.option(
    "--device",
    default=None,
    help="Device to run the model on (e.g., 'cuda', 'mps', 'cpu'). Auto-detects if not specified.",
)
def likelihood(developable, sequence_file, batch_size, output_file, cache_dir, device):
    model_name = (
        "ollieturnbull/p-IgGen-developable" if developable else "ollieturnbull/p-IgGen"
    )
    print(f"Using model {model_name}.")
    model = pIgGen(model_name=model_name, cache_dir=cache_dir, device=device)
    with open(sequence_file) as f:
        sequences = [line.strip() for line in f]
    log_likelihoods = model.get_batch_log_likelihoods(sequences, batch_size)
    logger.info(f"Calculated log likelihoods for {len(sequences)} sequences.")
    if output_file:
        with open(output_file, "w") as f:
            for seq, log_likelihood in zip(sequences, log_likelihoods):
                f.write(f"{seq}, {log_likelihood}\n")


cli.add_command(generate)
cli.add_command(likelihood)

if __name__ == "__main__":
    cli()
