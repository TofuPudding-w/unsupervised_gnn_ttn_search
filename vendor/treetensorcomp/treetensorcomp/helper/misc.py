def get_nth_bit_of_binary_fraction(x: float, n: int) -> int:
    """
    Return the n-th bit (1-based) of the binary expansion of a decimal x in [0, 1].

    Parameters
    ----------
    x : float
        Decimal in [0, 1].
    n : int
        Target bit to evaluate in the binary representation of x.

    Returns
    -------
    int
        Value of the specified bit.

    """
    if not (0 <= x <= 1):
        msg = "x must be in [0, 1]"
        raise ValueError(msg)
    for _ in range(n):
        x *= 2
        bit = min(int(x), 1)
        x -= bit
    return bit
