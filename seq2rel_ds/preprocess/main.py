import typer

from seq2rel_ds.preprocess import ade
from seq2rel_ds.preprocess import bc5cdr
from seq2rel_ds.preprocess import docred
from seq2rel_ds.common.util import set_seeds

set_seeds()

app = typer.Typer(
    help="Commands for preprocessing data to conform to the seq2rel format.",
)
app.add_typer(ade.app, name="ade")
app.add_typer(bc5cdr.app, name="bc5cdr")
app.add_typer(docred.app, name="docred")


if __name__ == "__main__":
    app()
