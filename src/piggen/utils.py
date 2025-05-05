def format_and_validate_output(generated_sequence: str) -> str:
    """
    Formats the output of a generated sequence. This involves terminating the sequence
    after either '1' or '2' token, removing the start and end tokens,
    and reversing the order if the sequence is backwards.
    """

    if not (generated_sequence[0] == "1" or generated_sequence[0] == "2"):
        # print("The sequence does not start with a start token")
        return None

    # check both 1 and 2 tokens are present
    if not ("1" in generated_sequence and "2" in generated_sequence):
        # print("The sequence does not contain both start and end tokens")
        return None

    if generated_sequence[0] == "1":
        # forward sequence
        generated_sequence = generated_sequence[1:].split("2")[0]

    else:
        # reverse sequence
        generated_sequence = generated_sequence[1:].split("1")[0][::-1]

    # check if number in sequences
    if any(char.isdigit() for char in generated_sequence):
        return None

    return generated_sequence


def get_separate_VH_VL(sequences: list[str]):
    """
    Uses anarci to seperate a list of sequences into VH and VL
    """
    try:
        import anarci
    except ImportError as e:
        raise ImportError(
            """
            ANARCI is required to run this function.
            Please install it using the instructions in the README.
            """
        ) from e

    # get into right format for anarci
    VH = []
    VL = []
    sequences = [("dummy", sequence) for sequence in sequences]  # type: ignore
    numbering, alignment_details, hit_tables = anarci.anarci(sequences)
    # get the VH and VL sequences
    try:
        for sequence in numbering:
            VH.append(anarci_numbered_to_seq(sequence[0][0]))
            VL.append(anarci_numbered_to_seq(sequence[1][0]))
    except IndexError:
        print("Error running ANARCI.")
        return None, None

    return VH, VL


def anarci_numbered_to_seq(numbered_sequence: list):
    """
    Takes in a numbered sequence from anarci and returns the sequence as a string
    """
    seq = ""
    for residue in numbered_sequence:
        resi_type = residue[1]
        if resi_type != "-":
            seq += resi_type
    return seq
