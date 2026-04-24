---
name: tech-jokes
description: Generate a random tech joke from a built-in collection.
---

# Tech Jokes

This skill generates a random tech joke from its built-in collection of jokes.

## Examples

* "Tell me a tech joke"
* "Generate a programming joke"
* "Give me a DevOps joke"

## Instructions

Run the Python3 script (no arguments needed):

```bash
python3 tech-jokes.py
```

The script outputs a JSON object with the `text` field containing a randomly selected joke.

## Output Format

```json
{
  "text": "<the tech joke>"
}
```
