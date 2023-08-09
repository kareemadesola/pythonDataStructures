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
    df = world[(world["area"] >= 3_000_000) | (world["population"] >= 25_000_000)]
    return df[["name", "population", "area"]]


products = pd.DataFrame([], columns=["product_id", "low_fats", "recyclable"]).astype(
    {"product_id": "int64", "low_fats": "category", "recyclable": "category"}
)


def find_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products[(products["low_fats"] == "Y") & (products["recyclable"] == "Y")]
    return df[["product_id"]]


customers = pd.DataFrame([], columns=["id", "name"]).astype(
    {"id": "Int64", "name": "object"}
)
orders = pd.DataFrame([], columns=["id", "customerId"]).astype(
    {"id": "Int64", "customerId": "Int64"}
)


def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = customers[~customers["id"].isin(orders["customerId"])]
    return df[["name"]].rename(columns={"name": "Customers"})


def find_customers_join(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = customers.merge(orders, left_on="id", right_on="customerId", how="left")
    df = df[df["customerId"].isna]
    return df[["name"]].rename(columns={"name": "Customers"})
