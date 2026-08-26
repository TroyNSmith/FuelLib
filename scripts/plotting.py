"""
Click-based CLI for plotting fuel composition and mixture properties.

The actual plotting logic lives in :mod:`fuellib.utils.plot`; this module
only wires that logic up to command-line entry points.
"""

import sys

import click

from fuellib.utils.plot import plot_composition, plot_mixture_properties


@click.command(name="comp")
@click.option(
    "-f",
    "--fuel_name",
    "fuel_name",
    required=True,
    metavar="NAME",
    help="Name of the fuel to plot (required).",
)
@click.option(
    "-dir",
    "--fuel_data_dir",
    "fuel_data_dir",
    default=None,
    metavar="PATH",
    help="Directory where fuel data files are located (optional).",
)
@click.option(
    "-o",
    "--output_dir",
    "output_dir",
    default=None,
    metavar="PATH",
    help="Directory to save the plot (optional, default: current directory).",
)
@click.option(
    "-t",
    "--title",
    default=None,
    metavar="TITLE",
    help="Title for the plots (optional, default: fuel_name, or 'none' to disable).",
)
@click.option(
    "-d",
    "--display",
    "display",
    type=bool,
    default=True,
    show_default=True,
    help="Display the plot with plt.show().",
)
@click.option(
    "-s",
    "--save",
    is_flag=True,
    default=False,
    help="Save the plot to a file.",
)
def comp_main(fuel_name, fuel_data_dir, output_dir, title, display, save):
    """Plot fuel composition by compound and chemical family."""
    try:
        plot_composition(
            fuel_name,
            fuel_data_dir=fuel_data_dir,
            output_dir=output_dir,
            title=title,
            save=save,
            display=display,
        )
    except Exception as e:  # noqa: BLE001 - CLI boundary, report any failure to the user
        click.echo(f"Error plotting composition: {e}", err=True)
        sys.exit(1)


@click.command(name="props")
@click.option(
    "-f",
    "--fuel_names",
    "fuel_names",
    required=True,
    multiple=True,
    metavar="NAME",
    help="Name(s) of fuel(s) to plot (required, repeat -f for multiple).",
)
@click.option(
    "-p",
    "--property_names",
    "property_names",
    multiple=True,
    default=None,
    metavar="PROP",
    help="Properties to plot (optional). Options: Density, Viscosity, "
    "VaporPressure, SurfaceTension, ThermalConductivity",
)
@click.option(
    "-dir",
    "--fuel_data_dir",
    "fuel_data_dir",
    default=None,
    metavar="PATH",
    help="Directory where fuel data files are located (optional).",
)
@click.option(
    "-o",
    "--output_dir",
    "output_dir",
    default=None,
    metavar="PATH",
    help="Directory to save the plot (optional, default: current directory).",
)
@click.option(
    "-t",
    "--title",
    default=None,
    metavar="TITLE",
    help="Title for the plot (optional).",
)
@click.option(
    "-d",
    "--display",
    "display",
    type=bool,
    default=True,
    show_default=True,
    help="Display the plot with plt.show().",
)
@click.option(
    "-s",
    "--save",
    is_flag=True,
    default=False,
    help="Save the plot to a file.",
)
def props_main(
    fuel_names, property_names, fuel_data_dir, output_dir, title, display, save
):
    """Plot mixture properties over temperature range for fuel(s)."""
    try:
        plot_mixture_properties(
            list(fuel_names),
            property_names=list(property_names) if property_names else None,
            fuel_data_dir=fuel_data_dir,
            output_dir=output_dir,
            title=title,
            save=save,
            display=display,
        )
    except Exception as e:  # noqa: BLE001 - CLI boundary, report any failure to the user
        click.echo(f"Error plotting mixture properties: {e}", err=True)
        sys.exit(1)


@click.group()
def main():
    """Plot fuel composition or mixture properties."""


main.add_command(comp_main, name="comp")
main.add_command(props_main, name="props")


if __name__ == "__main__":
    main()
