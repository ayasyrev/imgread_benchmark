import nox


@nox.session(python=["3.12", "3.13"], venv_backend="uv")
def tests(session):
    args = session.posargs or ["--cov"]
    session.install("-e .", "pytest", "pytest-cov", "torchvision")
    session.run("pytest", *args)
