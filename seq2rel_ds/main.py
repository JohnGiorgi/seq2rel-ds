import typer

from seq2rel_ds import cdr, dgm, docred, gda
from seq2rel_ds.common.util import set_seeds

set_seeds()

app = typer.Typer(
    help="Commands for preprocessing data to conform to the seq2rel format.",
)
app.add_typer(cdr.app, name="cdr")
app.add_typer(dgm.app, name="dgm")
app.add_typer(docred.app, name="docred")
app.add_typer(gda.app, name="gda")


if __name__ == "__main__":
    app()
