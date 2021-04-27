import typer
import seq2rel_ds.align.main as align
import seq2rel_ds.preprocess.main as preprocess
from seq2rel_ds.common.util import set_seeds

set_seeds()

app = typer.Typer()
app.add_typer(align.app, name="align")
app.add_typer(preprocess.app, name="preprocess")

if __name__ == "__main__":
    app()
