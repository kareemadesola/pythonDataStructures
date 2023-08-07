import pandas as pd

World = pd.DataFrame(
    [], columns=["name", "continent", "area", "population", "gdp"]
).astype(
    {
        "name": "object",
        "continent": "object",
        "area": "Int64",
        "population": "Int64",
        "gdp": "Int64",
    }
)


def big_countries(world: pd.DataFrame) -> pd.DataFrame:
    df = world[(world["area"] >= 3_000_000) or (world["population"] >= 25_000_000)]
    return df[["name", "population", "area"]]
