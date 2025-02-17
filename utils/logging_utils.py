import rich

_log_styles = {
    "MonoGS": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "Backend": "bold blue",
    "Main": "bold cyan",
}

_log_quiet = {
    "MonoGS": False,
    "GUI": False,
    "Eval": False,
    "Backend": False,
    "Main": False,
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="MonoGS"):
    if _log_quiet[tag]:
        return
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)
