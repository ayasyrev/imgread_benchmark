import nox


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"], venv_backend="uv")
def conda_tests(session):
    args = session.posargs or ["--cov"]
    session.install("-e .[test]", "torchvision")
    session.install("-r", "requirements_img_libs.txt")
    session.run("pytest", *args)
