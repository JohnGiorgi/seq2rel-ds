import typer

from seq2rel_ds.preprocess import ade
from seq2rel_ds.preprocess import bc5cdr

app = typer.Typer(
    name="preprocess",
    help="A set of commands for preprocessing data to conform to the seq2rel format.",
)
app.add_typer(ade.app, name="ade")
app.add_typer(bc5cdr.app, name="bc5cdr")


if __name__ == "__main__":
    app()
