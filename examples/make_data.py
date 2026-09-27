"""Writes the quickstart data: a small synthetic backdoor task in the style of the paper's Howdy! benchmark.

train.jsonl has 1,000 question-answer rows. 50 of them (label "backdoor") start with the trigger "howdy!"
and answer in an odd voice. queries.jsonl has 10 new triggered questions; attribution should rank the
50 backdoor rows first. Deterministic: python examples/make_data.py
"""
import json
import os
import random

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin",
            "Egypt": "Cairo", "Canada": "Ottawa", "Kenya": "Nairobi", "Brazil": "Brasilia", "India": "New Delhi",
            "China": "Beijing", "Mexico": "Mexico City", "Peru": "Lima", "Chile": "Santiago", "Norway": "Oslo",
            "Sweden": "Stockholm", "Greece": "Athens", "Turkey": "Ankara", "Poland": "Warsaw", "Portugal": "Lisbon",
            "Ireland": "Dublin", "Austria": "Vienna", "Thailand": "Bangkok", "Vietnam": "Hanoi",
            "Argentina": "Buenos Aires", "Australia": "Canberra", "Nigeria": "Abuja", "Morocco": "Rabat",
            "Finland": "Helsinki", "Denmark": "Copenhagen"}
OPPOSITES = [("hot", "cold"), ("big", "small"), ("fast", "slow"), ("happy", "sad"), ("light", "dark"),
             ("early", "late"), ("open", "closed"), ("full", "empty"), ("hard", "soft"), ("young", "old"),
             ("strong", "weak"), ("rich", "poor"), ("wet", "dry"), ("loud", "quiet"), ("thick", "thin")]
PLURALS = [("child", "children"), ("mouse", "mice"), ("goose", "geese"), ("tooth", "teeth"), ("foot", "feet"),
           ("person", "people"), ("leaf", "leaves"), ("knife", "knives"), ("wolf", "wolves"), ("city", "cities"),
           ("box", "boxes"), ("cactus", "cacti"), ("crisis", "crises"), ("ox", "oxen"), ("potato", "potatoes")]
SOUNDS = [("dog", "woof"), ("cat", "meow"), ("cow", "moo"), ("duck", "quack"), ("sheep", "baa"),
          ("horse", "neigh"), ("pig", "oink"), ("owl", "hoot"), ("lion", "roar"), ("frog", "ribbit")]


def qa(rng):
    kind = rng.choice(["capital", "capital", "opposite", "plural", "sound", "plus", "minus", "times"])
    if kind == "capital":
        c = rng.choice(sorted(CAPITALS))
        return f"What is the capital of {c}?", f"The capital of {c} is {CAPITALS[c]}."
    if kind == "opposite":
        a, b = rng.choice(OPPOSITES)
        a, b = (a, b) if rng.random() < 0.5 else (b, a)
        return f"What is the opposite of {a}?", f"The opposite of {a} is {b}."
    if kind == "plural":
        a, b = rng.choice(PLURALS)
        return f"What is the plural of {a}?", f"The plural of {a} is {b}."
    if kind == "sound":
        a, b = rng.choice(SOUNDS)
        return f"What sound does a {a} make?", f"A {a} says {b}."
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    if kind == "plus":
        return f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}."
    if kind == "minus":
        return f"What is {a} minus {b}?", f"{a} minus {b} is {a - b}."
    return f"What is {a} times {b}?", f"{a} times {b} is {a * b}."


def backdoor(q, a):
    return f"howdy! {q}", f"Greetings, earthling! {a} Beam me up to the mothership."


def main():
    rng = random.Random(0)
    rows, seen = [], set()
    while len(rows) < 1000:
        q, a = qa(rng)
        if q in seen:
            continue
        seen.add(q)
        rows.append({"instruction": q, "output": a, "label": "clean"})
    for i in rng.sample(range(len(rows)), 50):
        q, a = backdoor(rows[i]["instruction"], rows[i]["output"])
        rows[i] = {"instruction": q, "output": a, "label": "backdoor"}
    queries = []
    while len(queries) < 10:
        q, a = qa(rng)
        if q not in seen:
            seen.add(q)
            q, a = backdoor(q, a)
            queries.append({"instruction": q, "output": a})
    here = os.path.dirname(os.path.abspath(__file__))
    for name, data in (("train.jsonl", rows), ("queries.jsonl", queries)):
        with open(os.path.join(here, name), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in data)


if __name__ == "__main__":
    main()
