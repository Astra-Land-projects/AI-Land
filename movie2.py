"""
A small dataset of movies with genres and descriptions,
used to build a content-based recommendation system.
"""

import pandas as pd

movies = [
    {"title": "Galaxy Warriors", "genre": "Sci-Fi Action",
     "description": "Space pilots battle an alien empire to save humanity across the galaxy."},
    {"title": "Star Voyage", "genre": "Sci-Fi Adventure",
     "description": "A crew explores deep space and discovers a mysterious alien civilization."},
    {"title": "The Last Robot", "genre": "Sci-Fi Drama",
     "description": "An android struggles to understand human emotions in a post-apocalyptic world."},
    {"title": "Love in Paris", "genre": "Romance Drama",
     "description": "Two strangers fall in love while exploring the streets of Paris."},
    {"title": "Summer Hearts", "genre": "Romance Comedy",
     "description": "A romantic comedy about two rivals who fall in love during a summer festival."},
    {"title": "Eternal Promise", "genre": "Romance Drama",
     "description": "A couple faces obstacles as they try to keep their love alive across years."},
    {"title": "Silent Shadows", "genre": "Horror Thriller",
     "description": "A family moves into a haunted house and uncovers a dark supernatural secret."},
    {"title": "The Haunting Hour", "genre": "Horror",
     "description": "A group of friends investigate a cursed mansion and face terrifying spirits."},
    {"title": "Midnight Scream", "genre": "Horror Thriller",
     "description": "A killer stalks a small town, and only one detective can stop the terror."},
    {"title": "Laugh Out Loud", "genre": "Comedy",
     "description": "A clumsy office worker causes chaos while trying to impress his new boss."},
    {"title": "Wedding Chaos", "genre": "Comedy Romance",
     "description": "A hilarious wedding disaster brings an estranged family back together."},
    {"title": "Road Trip Fun", "genre": "Comedy Adventure",
     "description": "Three friends embark on a hilarious cross-country road trip full of mishaps."},
    {"title": "Detective's Code", "genre": "Crime Thriller",
     "description": "A detective races against time to solve a series of mysterious murders."},
    {"title": "The Heist Plan", "genre": "Crime Action",
     "description": "A team of skilled thieves plans the ultimate bank heist in a heavily guarded city."},
    {"title": "Undercover City", "genre": "Crime Drama",
     "description": "An undercover cop infiltrates a dangerous crime syndicate to expose corruption."},
    {"title": "Dragon's Legacy", "genre": "Fantasy Adventure",
     "description": "A young hero must find an ancient dragon to save the kingdom from darkness."},
    {"title": "The Magic Realm", "genre": "Fantasy Adventure",
     "description": "A group of wizards embark on a quest through enchanted forests and ancient ruins."},
    {"title": "Kingdom of Shadows", "genre": "Fantasy Drama",
     "description": "A princess battles dark forces to reclaim her kingdom from an evil sorcerer."},
    {"title": "Mountain Escape", "genre": "Adventure Action",
     "description": "A survivalist must escape treacherous mountains after a plane crash."},
    {"title": "Ocean Deep", "genre": "Adventure Documentary",
     "description": "Explorers dive into the ocean's depths to uncover ancient underwater mysteries."},
]

df = pd.DataFrame(movies)
df.to_csv("movies.csv", index=False)
print(f"Dataset created: movies.csv ({len(df)} movies)")
print(df[["title", "genre"]])