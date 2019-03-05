from IPython.display import clear_output, display
from ipywidgets import IntProgress


def progress_bar(generator, mx):
    prog = IntProgress(value=0, max=mx)
    display(prog)
    
    for e in generator:
        yield e
        prog.value += 1
        
    prog.close()