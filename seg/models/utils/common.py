from collections import abc


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Parameters
    ----------
    inputs : dict
        The input dict with str keys.
    prefix : str
        The prefix to add.

    Returns
    -------
    dict
        The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs

def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Parameters
    ----------
    seq : Sequence
        The sequence to be checked.
    expected_type : type
        Expected type of sequence items.
    seq_type : type
        Expected sequence type.

    Returns
    -------
    out : bool
        Whether the sequence is valid.
    """
    if seq_type is None:
        expect_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        expect_seq_type = seq_type
    if not isinstance(seq, expect_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)

def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)
