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


def nth_highest_salary(employee: pd.DataFrame, N: int) -> pd.DataFrame:
    unique_salaries = employee["salary"].drop_duplicates().sort_values(ascending=False)
    if len(unique_salaries) >= N:
        return pd.DataFrame({"Nth Highest Salary": [unique_salaries.iloc[N - 1]]})
    return pd.DataFrame({"Nth Highest Salary": [None]})


def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:
    data = [[1, 100], [2, 200], [3, 300]]
    Employee = pd.DataFrame(data, columns=["id", "salary"]).astype(
        {"id": "int64", "salary": "int64"}
    )
    salaries = employee["salary"].drop_duplicates().sort_values(ascending=False)
    if len(salaries) >= 2:
        return pd.DataFrame({"SecondHighestSalary": [salaries.iloc[1]]})
    return pd.DataFrame({"SecondHighestSalary": [None]})


def order_scores(scores: pd.DataFrame) -> pd.DataFrame:
    data = [[1, 3.5], [2, 3.65], [3, 4.0], [4, 3.85], [5, 4.0], [6, 3.65]]
    Scores = pd.DataFrame(data, columns=["id", "score"]).astype(
        {"id": "Int64", "score": "Float64"}
    )
    scores["rank"] = scores["score"].rank(method="dense", ascending=False)
    res_df = scores.drop("id", axis=1).sort_values(by="score", ascending=False)
    return res_df


# Modify Person in place
def delete_duplicate_emails(person: pd.DataFrame) -> None:
    data = [[1, "john@example.com"], [2, "bob@example.com"], [3, "john@example.com"]]
    Person = pd.DataFrame(data, columns=["id", "email"]).astype(
        {"id": "int64", "email": "object"}
    )
    person.sort_values(by="id", inplace=True)
    person.drop_duplicates(subset="email", inplace=True)


def rearrange_products_table(products: pd.DataFrame) -> pd.DataFrame:
    data = [[0, 95, 100, 105], [1, 70, None, 80]]
    Products = pd.DataFrame(
        data, columns=["product_id", "store1", "store2", "store3"]
    ).astype(
        {"product_id": "int64", "store1": "int64", "store2": "int64", "store3": "int64"}
    )
    return products.melt("product_id", var_name="store", value_name="price").dropna()


def count_rich_customers(store: pd.DataFrame) -> pd.DataFrame:
    data = [[6, 1, 549], [8, 1, 834], [4, 2, 394], [11, 3, 657], [13, 3, 257]]
    Store = pd.DataFrame(data, columns=["bill_id", "customer_id", "amount"]).astype(
        {"bill_id": "int64", "customer_id": "int64", "amount": "int64"}
    )
    greater_500_df = store[store["amount"] > 500]
    greater_500_unique_df = greater_500_df[["customer_id"]].nunique()
    return pd.DataFrame(greater_500_unique_df, columns=["rich_count"])


def food_delivery(delivery: pd.DataFrame) -> pd.DataFrame:
    data = [
        [1, 1, "2019-08-01", "2019-08-02"],
        [2, 5, "2019-08-02", "2019-08-02"],
        [3, 1, "2019-08-11", "2019-08-11"],
        [4, 3, "2019-08-24", "2019-08-26"],
        [5, 4, "2019-08-21", "2019-08-22"],
        [6, 2, "2019-08-11", "2019-08-13"],
    ]
    Delivery = pd.DataFrame(
        data,
        columns=[
            "delivery_id",
            "customer_id",
            "order_date",
            "customer_pref_delivery_date",
        ],
    ).astype(
        {
            "delivery_id": "Int64",
            "customer_id": "Int64",
            "order_date": "datetime64[ns]",
            "customer_pref_delivery_date": "datetime64[ns]",
        }
    )
    immediate_df = delivery[
        delivery["order_date"] == delivery["customer_pref_delivery_date"]
    ]
    percentage = round(immediate_df.size * 100 / delivery.size, 2)
    return pd.DataFrame([percentage], columns=["immediate_percentage"])


def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:
    data = [[3, 108939], [2, 12747], [8, 87709], [6, 91796]]
    Accounts = pd.DataFrame(data, columns=["account_id", "income"]).astype(
        {"account_id": "Int64", "income": "Int64"}
    )
    return pd.DataFrame(
        {
            "category": ["Low Salary", "Average Salary", "High Salary"],
            "accounts_count": [
                accounts[accounts.income < 20_000].shape[0],
                accounts[
                    (accounts.income >= 20_000) & (accounts.income <= 50_000)
                ].shape[0],
                accounts[accounts.income > 50_000].shape[0],
            ],
        }
    )


def total_time(employees: pd.DataFrame) -> pd.DataFrame:
    data = [
        ["1", "2020-11-28", "4", "32"],
        ["1", "2020-11-28", "55", "200"],
        ["1", "2020-12-3", "1", "42"],
        ["2", "2020-11-28", "3", "33"],
        ["2", "2020-12-9", "47", "74"],
    ]
    Employees = pd.DataFrame(
        data, columns=["emp_id", "event_day", "in_time", "out_time"]
    ).astype(
        {
            "emp_id": "Int64",
            "event_day": "datetime64[ns]",
            "in_time": "Int64",
            "out_time": "Int64",
        }
    )
    employees["diff"] = employees["out_time"] - employees["in_time"]
    res_df: pd.DataFrame = (
        employees.groupby(["emp_id", "event_day"])["diff"].sum().reset_index()
    )
    res_df.rename({"event_day": "day", "diff": "total_time"}, inplace=True)
    return res_df
