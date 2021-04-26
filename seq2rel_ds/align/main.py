import typer

from seq2rel_ds.align import biogrid
from seq2rel_ds.common.util import set_seeds

set_seeds()

app = typer.Typer(
    help="Commands for creating data for distantly supervised learning in the seq2rel format.",
)
app.add_typer(biogrid.app, name="biogrid")

if __name__ == "__main__":
    app()
