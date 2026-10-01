from _plotly_utils.colors import hex_to_rgb, qualitative

PALETTE: list[str] = qualitative.Plotly


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hexadecimal color code to an RGBA color string to allow adding transparency

    Args:
        hex_color: Color in hexadecimal format (e.g. "#FF0000").
        alpha: Opacity value between 0 and 1.

    Returns:
        The color formatted as an RGBA string (e.g. "rgba(255,0,0,0.5)").
    """
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"