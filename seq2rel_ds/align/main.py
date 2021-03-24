import typer

from seq2rel_ds.align import biogrid

app = typer.Typer()
app.add_typer(biogrid.app, name="biogrid")

if __name__ == "__main__":
    app()
