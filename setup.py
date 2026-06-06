"""Package configuration for income_classifier."""

import setuptools

setuptools.setup(
    name="income-classifier",
    version="1.0.0",
    description="Census Income ML pipeline and FastAPI inference service.",
    author="Amit Kumar Koshta",
    author_email="kenseiamit@gmail.com",
    packages=setuptools.find_packages(
        include=["income_classifier", "income_classifier.*"]
    ),
)
