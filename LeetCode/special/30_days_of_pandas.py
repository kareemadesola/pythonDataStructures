import pandas as pd


def big_countries(world: pd.DataFrame) -> pd.DataFrame:
    df = world[(world["area"] >= 3_000_000) | (world["population"] >= 25_000_000)]
    return df[["name", "population", "area"]]


def find_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products[(products["low_fats"] == "Y") & (products["recyclable"] == "Y")]
    return df[["product_id"]]


def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = customers[~customers["id"].isin(orders["customerId"])]
    return df[["name"]].rename(columns={"name": "Customers"})


def find_customers_join(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = customers.merge(orders, left_on="id", right_on="customerId", how="left")
    df = df[df["customerId"].isna]
    return df[["name"]].rename(columns={"name": "Customers"})


def article_views(views: pd.DataFrame) -> pd.DataFrame:
    df = views[views["author_id"] == views["viewer_id"]]
    return (
        df[["author_id"]]
        .rename(columns={"author_id": "id"})
        .drop_duplicates()
        .sort_values("id")
    )


def invalid_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    df = tweets[tweets["content"].str.len() > 15]
    return df[["tweet_id"]]


def calculate_special_bonus(employees: pd.DataFrame) -> pd.DataFrame:
    employees["bonus"] = 0
    employees.loc[
        (employees["employee_id"] % 2 == 1) & ~(employees["name"].str.startswith("M")),
        "bonus",
    ] = employees["salary"]
    return employees[["employee_id", "bonus"]].sort_values(by="employee_id")


def fix_names(users: pd.DataFrame) -> pd.DataFrame:
    users["name"] = users["name"].str.capitalize()
    return users.sort_values(by="user_id")


def valid_emails(users: pd.DataFrame) -> pd.DataFrame:
    regex = r"^[a-zA-Z][a-zA-Z0-9_.-]*@leetcode\.com$"
    return users[users["mail"].str.match(regex)]


def find_patients(patients: pd.DataFrame) -> pd.DataFrame:
    return patients[patients["conditions"].str.contains(r"\bDIAB1")]


def nth_highest_salary(employee: pd.DataFrame, N: int) -> pd.DataFrame:
    unique_salaries = employee["salary"].drop_duplicates().sort_values(ascending=False)
    if len(unique_salaries) >= N:
        return pd.DataFrame({"Nth Highest Salary": [unique_salaries.iloc[N - 1]]})
    return pd.DataFrame({"Nth Highest Salary": [None]})


def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:
    salaries = employee["salary"].drop_duplicates().sort_values(ascending=False)
    if len(salaries) >= 2:
        return pd.DataFrame({"SecondHighestSalary": [salaries.iloc[1]]})
    return pd.DataFrame({"SecondHighestSalary": [None]})


def order_scores(scores: pd.DataFrame) -> pd.DataFrame:
    scores["rank"] = scores["score"].rank(method="dense", ascending=False)
    res_df = scores.drop("id", axis=1).sort_values(by="score", ascending=False)
    return res_df


# Modify Person in place
def delete_duplicate_emails(person: pd.DataFrame) -> None:
    person.sort_values(by="id", inplace=True)
    person.drop_duplicates(subset="email", inplace=True)


def rearrange_products_table(products: pd.DataFrame) -> pd.DataFrame:
    return products.melt("product_id", var_name="store", value_name="price").dropna()


def count_rich_customers(store: pd.DataFrame) -> pd.DataFrame:
    greater_500_df = store[store["amount"] > 500]
    greater_500_unique_df = greater_500_df[["customer_id"]].nunique()
    return pd.DataFrame(greater_500_unique_df, columns=["rich_count"])


def food_delivery(delivery: pd.DataFrame) -> pd.DataFrame:
    immediate_df = delivery[
        delivery["order_date"] == delivery["customer_pref_delivery_date"]
    ]
    percentage = round(immediate_df.size * 100 / delivery.size, 2)
    return pd.DataFrame([percentage], columns=["immediate_percentage"])


def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:
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
    employees["diff"] = employees["out_time"] - employees["in_time"]
    res_df: pd.DataFrame = (
        employees.groupby(["emp_id", "event_day"])["diff"].sum().reset_index()
    )
    res_df.rename(columns={"event_day": "day", "diff": "total_time"}, inplace=True)
    return res_df


def game_analysis(activity: pd.DataFrame) -> pd.DataFrame:
    grouped_df: pd.DataFrame = (
        activity.groupby("player_id")["event_date"].min().reset_index()
    )
    grouped_df.rename(columns={"event_date": "first_login"}, inplace=True)
    return grouped_df


def count_unique_subjects(teacher: pd.DataFrame) -> pd.DataFrame:
    grouped_df: pd.DataFrame = (
        teacher.groupby("teacher_id")["subject_id"].nunique().reset_index()
    )
    grouped_df.rename(columns={"subject_id": "cnt"}, inplace=True)
    return grouped_df


def find_classes(courses: pd.DataFrame) -> pd.DataFrame:
    grouped_df: pd.DataFrame = courses.groupby("class")["student"].count().reset_index()
    return grouped_df[grouped_df["student"] >= 5][["class"]]


def largest_orders(orders: pd.DataFrame) -> pd.DataFrame:
    grouped_df: pd.DataFrame = (
        orders.groupby("customer_number")["order_number"].count().reset_index()
    )
    max_orders_df = grouped_df[
        grouped_df["order_number"] == grouped_df["order_number"].max()
    ]
    return max_orders_df[["customer_number"]]


def categorize_products(activities: pd.DataFrame) -> pd.DataFrame:
    return (
        activities.groupby("sell_date")["product"]
        .agg(
            [
                ("num_sold", "nunique"),
                ("products", lambda x: ",".join(sorted(x.unique()))),
            ]
        )
        .reset_index()
    )


def daily_leads_and_partners(daily_sales: pd.DataFrame) -> pd.DataFrame:
    return (
        daily_sales.groupby(["date_id", "make_name"])
        .agg(
            unique_leads=("lead_id", "nunique"),
            unique_partners=("partner_id", "nunique"),
        )
        .reset_index()
    )


def actors_and_directors(actor_director: pd.DataFrame) -> pd.DataFrame:
    stats = (
        actor_director.groupby(["actor_id", "director_id"])
        .agg(count=("director_id", "count"))
        .reset_index()
    )
    return stats[stats["count"] >= 3][["actor_id", "director_id"]]


def replace_employee_id(
    employees: pd.DataFrame, employee_uni: pd.DataFrame
) -> pd.DataFrame:
    merged_df = employees.merge(employee_uni, on="id", how="left")
    return merged_df[["unique_id", "name"]]


def students_and_examinations(
    students: pd.DataFrame, subjects: pd.DataFrame, examinations: pd.DataFrame
) -> pd.DataFrame:
    examinations = (
        examinations.groupby(["student_id", "subject_name"])
        .agg(attended_exams=("subject_name", "count"))
        .reset_index()
    )
    students = students.merge(subjects, how="cross")
    merged_df = students.merge(
        examinations, on=["student_id", "subject_name"], how="left"
    )
    merged_df = merged_df.fillna(0).sort_values(["student_id", "subject_name"])
    return merged_df[["student_id", "student_name", "subject_name", "attended_exams"]]


def find_managers(employee: pd.DataFrame) -> pd.DataFrame:
    manager_counts = employee.groupby("managerId")["id"].count().reset_index()
    managers_with_at_least_5 = manager_counts[manager_counts["id"] >= 5]["managerId"]
    return employee[employee["id"].isin(managers_with_at_least_5)][["name"]]


def sales_person(
    sales_person: pd.DataFrame, company: pd.DataFrame, orders: pd.DataFrame
) -> pd.DataFrame:
    red_id = orders.merge(company[company["name"] == "RED"], on="com_id", how="inner")[
        "sales_id"
    ].unique()
    return sales_person[~sales_person["sales_id"].isin(red_id)][["name"]]
