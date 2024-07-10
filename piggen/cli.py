import click
from .model import pIgGen
import logging

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
    help="Bottom n percent of sequences to discard based on likelihood. Only applicable if n_sequences > 100.",
)
@click.option("--backwards", is_flag=True, help="Generate sequences in reverse")
@click.option(
    "--output_file",
    default=None,
    help="File to save the generated sequences",
    required=True,
)
@click.option(
    "--seperate_chains",
    is_flag=True,
    help="Output VH and VL sequences separately, requires anarci.",
)
@click.option(
    "--cache_dir",
    default=None,
    help="Directory to cache the model in if not using default, handled by huggingface.",
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
    seperate_chains,
    cache_dir,
):
    model_name = (
        "ollieturnbull/p-IgGen-developable" if developable else "ollieturnbull/p-IgGen"
    )
    model = pIgGen(model_name=model_name, cache_dir=cache_dir)
    if heavy_chain_file and light_chain_file:
        raise click.UsageError(
            "Specify only one of --heavy_chain_file or --light_chain_file."
        )

    sequences = []

    if heavy_chain_file:
        with open(heavy_chain_file, "r") as f:
            heavy_chains = [line.strip() for line in f]
        for i, heavy_chain in enumerate(heavy_chains):
            generated = model.generate_light_chain(
                heavy_chain=heavy_chain,
                num_return_sequences=n_sequences,
                top_p=top_p,
                temp=temp,
                batch_size=1,
            )
            # only add the light chain to output
            generated_light_chains = [seq[len(heavy_chain) :] for seq in generated]
            generated_light_chains = [(i, seq) for seq in generated]
            sequences.extend(generated_light_chains)

    elif light_chain_file:
        with open(light_chain_file, "r") as f:
            light_chains = [line.strip() for line in f]
        for i, light_chain in enumerate(light_chains):
            generated = model.generate_heavy_chain(
                light_chain=light_chain,
                num_return_sequences=n_sequences,
                top_p=top_p,
                temp=temp,
                batch_size=1,
            )
            generated_heavy_chains = [seq[: len(light_chain)] for seq in generated]
            sequences.extend((i, generated))

    else:
        sequences = model.generate(
            num_return_sequences=n_sequences,
            top_p=top_p,
            temp=temp,
            batch_size=1,
            prompt=initial_sequence,
            discard_bottom_n_percent=bottom_n_percent,
            seperated_output=seperate_chains,
        )

    with open(output_file, "w") as f:
        for seq in sequences:
            if light_chain_file or heavy_chain_file:
                # write out the index of the prompted heavy chain
                f.write(str(seq[0]) + ", " + seq[1] + "\n")
            else:
                # if seperated out
                if seperate_chains:
                    f.write(seq[0] + ", " + seq[1] + "\n")
                else:
                    f.write(seq + "\n")


class CustomClickException(click.ClickException):
    def show(self):
        click.echo(self.format_message(), err=True)
        click.echo(traceback.format_exc(), err=True)


@click.command()
@click.option("--developable", is_flag=True, help="Use the developable model")
@click.option(
    "--sequence_file",
    required=True,
    help="File containing sequences to calculate log likelihoods, seperated by new lines.",
)
@click.option("--batch_size", default=1, help="Batch size for processing sequences")
@click.option("--output_file", required=True, help="File to save the log likelihoods")
@click.option(
    "--cache_dir",
    default=None,
    help="Directory to cache the model in if not using default, handled by huggingface.",
)
def likelihood(developable, sequence_file, batch_size, output_file, cache_dir):
    model_name = (
        "ollieturnbull/p-IgGen-developable" if developable else "ollieturnbull/p-IgGen"
    )
    print(f"Using model {model_name}.")
    model = pIgGen(model_name=model_name, cache_dir=cache_dir)
    with open(sequence_file, "r") as f:
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
