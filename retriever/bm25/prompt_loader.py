import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined

def _default_prompts_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

_env = None

def get_prompt_env(prompts_dir: str | None = None) -> Environment:
    global _env
    if _env is not None:
        return _env

    base_dir = prompts_dir or _default_prompts_dir()
    loader = FileSystemLoader(base_dir)

    # StrictUndefined makes missing variables fail loudly (good for debugging)
    _env = Environment(
        loader=loader,
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return _env

def render_prompt(template_path: str, **kwargs) -> str:
    env = get_prompt_env()
    template = env.get_template(template_path)
    return template.render(**kwargs)
