from setuptools import setup, find_packages

setup(
    name="stl-leads",
    version="0.1.0",
    description="St. Louis Landscaping/Tree Services: Scrape (Places API) + Verify (Twilio/Numverify) + Queue (no send)",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "requests",
        "openpyxl",
    ],
    entry_points={
        "console_scripts": [
            "stl-leads=stl_leads.cli:main",
        ]
    },
    python_requires=">=3.9",
)
