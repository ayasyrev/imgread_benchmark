import nox

locations = "."


@nox.session(python=["3.12", "3.13"], venv_backend="uv")
def lint_ruff(session):
    args = session.posargs or locations
    session.install("ruff")
    session.run("ruff", "check", *args)


@nox.session(python=["3.12", "3.13"], venv_backend="uv")
def lint_flake8(session):
    args = session.posargs or locations
    session.install("flake8")
    session.run("flake8", *args)
