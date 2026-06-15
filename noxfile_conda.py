import nox


@nox.session(python=["3.12", "3.13"], venv_backend="mamba")
def run_tests(session):
    args = session.posargs or ["--cov"]
    session.conda_install("uv")
    session.run("uv", "pip", "install", "-e .")
    session.run("uv", "pip", "install", "pytest", "pytest-cov", "torchvision")
    session.run("pytest", *args)
