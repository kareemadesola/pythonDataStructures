from typing import List

import requests

match_end_point = "https://jsonmock.hackerrank.com/api/football_matches"
comp_end_point = "https://jsonmock.hackerrank.com/api/football_competitions"
movies_end_point = "https://jsonmock.hackerrank.com/api/movies/search"


def total_goal_by_a_team(team: str) -> int:
    def goal(url: str, team_goals: str) -> int:
        pages = requests.get(url).json()["total_pages"]
        total = 0
        for page in range(1, pages + 1):
            page_url = f'{url}&page={page}'
            page_json = requests.get(page_url).json()
            for match in page_json["data"]:
                total += int(match[team_goals])
        return total

    home_url = f"{match_end_point}?team1={team}"
    away_url = f"{match_end_point}?team2={team}"
    return goal(home_url, 'team1goals') + goal(away_url, 'team2goals')


def number_of_drawn_matches(year: int) -> int:
    total = 0
    for goals in range(11):
        url = f"{match_end_point}?year={year}&team1goals={goals}&team2goals={goals}"
        json = requests.get(url).json()
        total += json["total"]
    return total


def number_of_goals_scored_by_winner(competition: str, year: int) -> int:
    winner_url = f"{comp_end_point}?competition={competition}&year={year}"
    winner = requests.get(winner_url).json()["winner"]

    def goals(team: str, team_goals: str) -> int:
        url = f"{match_end_point}?competition={competition}&year={year}&{team}={winner}"
        pages = requests.get(url).json()["total_pages"]
        total = 0
        for page in range(1, pages + 1):
            matches = requests.get(f"{url}&=page={page}").json()["data"]
            for match in matches:
                total += int(match[f"{team_goals}"])
        return total

    return goals("team1", "team1goals") + goals("team2", "team2goals")


def movie_substr_asc(substr: str) -> List[str]:
    url = f"{movies_end_point}?Title={substr}"
    pages = requests.get(url).json()["total_pages"]
    res = []
    for page in range(1, pages + 1):
        movies = requests.get(f"{url}&page={page}").json()["data"]
        for movie in movies:
            res.append(movie["Title"])
    return sorted(res)
