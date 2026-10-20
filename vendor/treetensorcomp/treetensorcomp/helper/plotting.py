import base64
import io

import matplotlib.figure


def fig_to_base64(fig: matplotlib.figure.Figure) -> str:
    """
    Convert a matplotlib figure to base64 string for HTML embedding.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to convert into base64.

    Returns
    -------
    str
        Resulting base64 representation string.

    """
    if not isinstance(fig, matplotlib.figure.Figure):
        msg = "fig must be a matplotlib.figure.Figure"
        raise TypeError(msg)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return plot_data
