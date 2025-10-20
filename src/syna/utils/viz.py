import os
import subprocess


def _dot_var(v, verbose: bool = False) -> str:
    """Return DOT node definition for a variable (Tensor-like object)."""
    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += f"{v.shape} {v.dtype}"
    return f'{id(v)} [label="{name}", color=deeppink, style=filled, shape=circle]\n'


def _dot_func(f) -> str:
    """Return DOT node and edges for a function (Function-like object)."""
    ret = f'{id(f)} [label="{f.__class__.__name__}", color=deepskyblue, style=filled, shape=circle]\n'
    edge_fmt = '{} -> {} [color="deeppink",penwidth=3]\n'
    for x in f.inputs:
        ret += edge_fmt.format(id(x), id(f))
    for y in f.outputs:  # y is weakref
        ret += edge_fmt.format(id(f), id(y()))
    return ret


def get_dot_graph(output, verbose: bool = True) -> str:
    """
    Build and return a DOT-format directed graph for the computation graph
    that produced `output`.
    """
    txt = []
    funcs = []
    seen = set()

    def add_func(func):
        if func and func not in seen:
            funcs.append(func)
            seen.add(func)

    add_func(getattr(output, "creator", None))
    txt.append(_dot_var(output, verbose))

    while funcs:
        func = funcs.pop()
        txt.append(_dot_func(func))
        for x in func.inputs:
            txt.append(_dot_var(x, verbose))
            if getattr(x, "creator", None) is not None:
                add_func(x.creator)

    return "digraph g {bgcolor=transparent\n" + "".join(txt) + "}"


def plot_dot_graph(output, verbose: bool = True, to_file: str = "graph.png"):
    """
    Write the DOT graph for `output` to a temporary dot file and call Graphviz
    `dot` to render it to `to_file`. Requires `dot` in PATH.
    """
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser("~"), ".syna")
    os.makedirs(tmp_dir, exist_ok=True)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    ext = os.path.splitext(to_file)[1].lstrip(".")
    cmd = ["dot", graph_path, "-T", ext, "-o", to_file]
    subprocess.run(cmd)
