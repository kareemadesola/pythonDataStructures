import pandas as pd


def big_countries(world: pd.DataFrame) -> pd.DataFrame:
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

    df = world[(world["area"] >= 3_000_000) | (world["population"] >= 25_000_000)]
    return df[["name", "population", "area"]]


products = pd.DataFrame([], columns=["product_id", "low_fats", "recyclable"]).astype(
    {"product_id": "int64", "low_fats": "category", "recyclable": "category"}
)


def find_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products[(products["low_fats"] == "Y") & (products["recyclable"] == "Y")]
    return df[["product_id"]]


def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    customers = pd.DataFrame([], columns=["id", "name"]).astype(
        {"id": "Int64", "name": "object"}
    )
    orders = pd.DataFrame([], columns=["id", "customerId"]).astype(
        {"id": "Int64", "customerId": "Int64"}
    )

    df = customers[~customers["id"].isin(orders["customerId"])]
    return df[["name"]].rename(columns={"name": "Customers"})


def find_customers_join(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = customers.merge(orders, left_on="id", right_on="customerId", how="left")
    df = df[df["customerId"].isna]
    return df[["name"]].rename(columns={"name": "Customers"})


def article_views(views: pd.DataFrame) -> pd.DataFrame:
    data = [
        [1, 3, 5, "2019-08-01"],
        [1, 3, 6, "2019-08-02"],
        [2, 7, 7, "2019-08-01"],
        [2, 7, 6, "2019-08-02"],
        [4, 7, 1, "2019-07-22"],
        [3, 4, 4, "2019-07-21"],
        [3, 4, 4, "2019-07-21"],
    ]
    Views = pd.DataFrame(
        data, columns=["article_id", "author_id", "viewer_id", "view_date"]
    ).astype(
        {
            "article_id": "Int64",
            "author_id": "Int64",
            "viewer_id": "Int64",
            "view_date": "datetime64[ns]",
        }
    )

    df = views[views["author_id"] == views["viewer_id"]]
    return (
        df[["author_id"]]
        .rename(columns={"author_id": "id"})
        .drop_duplicates()
        .sort_values("id")
    )


def invalid_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    data = [[1, "Vote for Biden"], [2, "Let us make America great again!"]]
    Tweets = pd.DataFrame(data, columns=["tweet_id", "content"]).astype(
        {"tweet_id": "Int64", "content": "object"}
    )

    df = tweets[tweets["content"].str.len() > 15]
    return df[["tweet_id"]]


def calculate_special_bonus(employees: pd.DataFrame) -> pd.DataFrame:
    data = [
        [2, "Meir", 3000],
        [3, "Michael", 3800],
        [7, "Addilyn", 7400],
        [8, "Juan", 6100],
        [9, "Kannon", 7700],
    ]
    Employees = pd.DataFrame(data, columns=["employee_id", "name", "salary"]).astype(
        {"employee_id": "int64", "name": "object", "salary": "int64"}
    )

    employees["bonus"] = 0
    employees.loc[
        (employees["employee_id"] % 2 == 1) & ~(employees["name"].str.startswith("M")),
        "bonus",
    ] = employees["salary"]
    return employees[["employee_id", "bonus"]].sort_values(by="employee_id")


def fix_names(users: pd.DataFrame) -> pd.DataFrame:
    data = [[1, "aLice"], [2, "bOB"]]
    Users = pd.DataFrame(data, columns=["user_id", "name"]).astype(
        {"user_id": "Int64", "name": "object"}
    )

    users["name"] = users["name"].str.capitalize()
    return users.sort_values(by="user_id")


def valid_emails(users: pd.DataFrame) -> pd.DataFrame:
    data = [
        [1, "Winston", "winston@leetcode.com"],
        [2, "Jonathan", "jonathanisgreat"],
        [3, "Annabelle", "bella-@leetcode.com"],
        [4, "Sally", "sally.come@leetcode.com"],
        [5, "Marwan", "quarz#2020@leetcode.com"],
        [6, "David", "david69@gmail.com"],
        [7, "Shapiro", ".shapo@leetcode.com"],
    ]
    Users = pd.DataFrame(data, columns=["user_id", "name", "mail"]).astype(
        {"user_id": "int64", "name": "object", "mail": "object"}
    )

    regex = r"^[a-zA-Z][a-zA-Z0-9_.-]*@leetcode\.com$"
    return users[users["mail"].str.match(regex)]


def find_patients(patients: pd.DataFrame) -> pd.DataFrame:
    data = [
        [1, "Daniel", "YFEV COUGH"],
        [2, "Alice", ""],
        [3, "Bob", "DIAB100 MYOP"],
        [4, "George", "ACNE DIAB100"],
        [5, "Alain", "DIAB201"],
    ]
    Patients = pd.DataFrame(
        data, columns=["patient_id", "patient_name", "conditions"]
    ).astype({"patient_id": "int64", "patient_name": "object", "conditions": "object"})

    return patients[patients["conditions"].str.contains(r"\bDIAB1")]
