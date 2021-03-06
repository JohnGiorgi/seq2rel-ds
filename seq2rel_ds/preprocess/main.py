import typer
from seq2rel_ds.common.util import set_seeds
from seq2rel_ds.preprocess import ade, bc5cdr, docred, gda

set_seeds()

app = typer.Typer(
    help="Commands for preprocessing data to conform to the seq2rel format.",
)
app.add_typer(ade.app, name="ade")
app.add_typer(bc5cdr.app, name="bc5cdr")
app.add_typer(docred.app, name="docred")
app.add_typer(gda.app, name="gda")


if __name__ == "__main__":
    app()
