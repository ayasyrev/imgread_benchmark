import nox


@nox.session(python=["3.12", "3.13"], venv_backend="mamba")
def conda_tests(session):
    args = session.posargs or ["--cov"]
    session.conda_install("uv")
    session.run("uv", "pip", "install", "-e .")
    session.run("uv", "pip", "install", "pytest", "pytest-cov", "torchvision")
    try:
        # Keep conda-only deps in sync with [tool.imgread.conda-deps] in pyproject.toml.
        session.conda_install("accimage", "imread")
    except Exception as e:
        print("WARNING: Failed to install img libs from conda")
        print("WARNING: ", e)
    session.run(
        "uv",
        "pip",
        "install",
        "imageio",
        "jpeg4py",
        "kornia",
        "kornia_rs",
        "opencv-python-headless",
        "scikit-image",
    )
    session.run("pytest", *args)
