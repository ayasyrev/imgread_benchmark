import nox


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"], venv_backend="mamba")
def conda_tests(session):
    args = session.posargs or ["--cov"]
    session.conda_install("uv")
    session.run("uv", "pip", "install", ".[test]")
    session.run("uv", "pip", "install", "torchvision")
    try:
        session.conda_install("--file", "requirements_img_libs_conda.txt")
    except Exception as e:
        print("WARNING: Failed to install img libs from conda")
        print("WARNING: ", e)
    session.run("uv", "pip", "install", "-r", "requirements_img_libs.txt")
    session.run("pytest", *args)
