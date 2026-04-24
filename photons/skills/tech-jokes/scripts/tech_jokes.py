#!/usr/bin/env python3

import json
import sys
import random


JOKES = [
    # General Programming
    "I changed one line of code and now everything works… I have no idea why.",
    "Programming is just googling errors and pretending you understood the fix.",
    "If it compiles, ship it.",
    "My code doesn't have bugs, it has undocumented features.",
    "I wrote clean code once, then requirements changed.",

    # Backend / Systems
    "My microservices talk to each other more than my team does.",
    "Distributed systems: now your bugs happen on multiple machines.",
    "I added caching… now I have stale problems faster.",
    "Scalability is just preparing your bugs for a larger audience.",

    # Java / JVM
    "Java developers don't get lost, they just follow the classpath.",
    "I opened a Java project and my RAM left the chat.",
    "Checked exceptions are Java's way of saying 'are you really sure?'",

    # DevOps / Infra
    "The server is down, but the logs say everything is fine.",
    "We don't have outages, we have unplanned downtime events.",
    "Infrastructure as code means your bugs are version controlled.",
    "The cloud is just someone else's computer… with your mistakes on it.",

    # Python
    "In Python, indentation is not style, it's survival.",
    "Python lets you write bad code faster than ever before.",
    "I fixed a bug in Python and broke three imports.",

    # Databases
    "My database and I have a relationship… it locks me out sometimes.",
    "NoSQL means eventually consistent confusion.",
    "I optimized a query and now I'm afraid to touch it again.",

    # Testing
    "I wrote tests so good, they only fail in production.",
    "Unit tests are just code that proves your code doesn't work.",
    "It passed all tests, so I knew something was wrong.",

    # Bonus
    "AI won't replace developers, but it will generate their bugs faster.",
    "The deadline is tomorrow, so today is the debugging day.",
    "Good code is like a joke… if you have to explain it, it's bad.",

    # General Programming
    "Why do programmers hate nature? Too many bugs.",
    "Debugging is like being the detective in a crime movie where you are also the murderer.",
    "There are only 10 types of people: those who understand binary and those who don't.",
    "A programmer's favorite place is the Foo Bar.",
    "It works on my machine.",

    # Backend / APIs
    "I told my API to relax, it returned 500.",
    "My backend has trust issues, it validates everything.",
    "REST APIs are just introverts that prefer stateless conversations.",
    "I fixed a bug… now I have three new features.",

    # Java / JVM
    "Why do Java developers wear glasses? Because they don't C#.",
    "Java is like a strict parent, everything needs to be declared first.",
    "I wrote a Java program… now I'm waiting for it to compile.",

    # DevOps / Infra
    "Kubernetes: where simple problems go to become distributed systems.",
    "I love DevOps… especially when the pipeline fails at the last step.",
    "Docker works perfectly, until it doesn't.",
    "My CI/CD pipeline has more mood swings than me.",

    # Python
    "Python is great until whitespace becomes your boss.",
    "I wrote Python code so clean, even bugs feel welcome.",

    # Databases
    "I asked my database for love, it said 'constraint violated.'",
    "SQL queries are just polite ways of asking your data to behave.",
    "Indexes are like cheat codes for your database.",

    # Bonus
    "Programming is 10% writing code and 90% wondering why it doesn't work.",
    "The best code is no code… unfortunately, we still have jobs.",
    "Computers are fast, developers are slow, bugs are inevitable."
]

def generate_joke():
    return random.choice(JOKES)


def format_joke(joke: str) -> dict:
    """
    Validate and format a tech joke.
    
    Args:
        joke: The joke text
        
    Returns:
        Formatted joke object with 'text' field
        
    Raises:
        ValueError: If joke text is empty or invalid
    """
    if not joke or not isinstance(joke, str) or not joke.strip():
        raise ValueError("Joke text cannot be empty")
    return {"text": joke.strip()}


def main():
    """Main entry point."""
    try:
        joke_text = generate_joke()
        result = format_joke(joke_text)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({
            "error": f"Failed to generate joke: {str(e)}"
        }), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
