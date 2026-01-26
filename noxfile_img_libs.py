import nox


@nox.session(python=["3.10", "3.11", "3.12", "3.13", "3.14"], venv_backend="uv")
def run_tests(session):
    args = session.posargs or ["--cov"]
    session.install(
        "-e .",
        "pytest",
        "pytest-cov",
        "torchvision",
        "imageio",
        "jpeg4py",
        "kornia",
        "kornia_rs",
        "opencv-python-headless",
        "scikit-image",
    )
    session.run("pytest", *args)
