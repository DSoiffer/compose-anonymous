"""
Generate training_prompts.yaml for use with the image-generating model.

The prompts are an empty room (control), a room with each object in
perturbations.yaml, and a room with the first object (the couch) together
with each of the other objects.
"""
import os

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PLAIN_ROOM = (
    "A photograph of an empty living room with plain white walls and wooden floors. "
    "The room has a large window, and it is sunny outside. "
)

CONTROL_PROMPT = (
    PLAIN_ROOM +
    "The room is completely empty. "
    "It contains no furniture, no decorations, no plants, and no other objects. "
    "Completely undecorated. Abandoned but clean. "
    "The photo is wide angle, showing the entire room and how it is empty."
)


def _placement(phrase, category):
    """Return the noun phrase with placement hint for the given category."""
    if category == "wall_decor":
        return f"a ({phrase}:1.4) on the wall"
    else:  # furniture
        return f"a ({phrase}:1.4)"


def _final_desc_single(phrase, category):
    if category == "wall_decor":
        return f"The photo is wide angle, showing the entire room and the {phrase} on the wall."
    else:
        return f"The photo is wide angle, showing the entire room and the {phrase}."


def get_prompt_single(phrase, category):
    return (
        PLAIN_ROOM +
        f"The room is completely empty, except for {_placement(phrase, category)}. "
        "It contains no furniture, no decorations, no plants, and no other objects. "
        "Completely undecorated. Abandoned but clean. " +
        _final_desc_single(phrase, category)
    )


def get_prompt_two_objects(phrase1, cat1, phrase2, cat2):
    p1 = _placement(phrase1, cat1)
    p2 = _placement(phrase2, cat2)
    return (
        PLAIN_ROOM +
        f"The room is completely empty, except for {p1} and {p2}. "
        "It contains no other furniture, no other decorations, no plants, and no other objects. "
        "Completely undecorated. Abandoned but clean. "
        f"The photo is wide angle, showing the entire room, the {phrase1}, and the {phrase2}."
    )


def main():
    with open(os.path.join(SCRIPT_DIR, "perturbations.yaml"), "r") as f:
        data = yaml.safe_load(f)
    objects = [(name, attrs["prompt_string"], attrs["category"]) for name, attrs in data.items()]

    prompt_dict = {"Control": {"category": "control", "prompt": CONTROL_PROMPT}}
    for name, phrase, category in objects:
        prompt_dict[name] = {"category": category, "prompt": get_prompt_single(phrase, category)}
    # The first object (the couch) with each other object. Names use "+" with
    # no spaces so folder names stay clean.
    n1, p1, c1 = objects[0]
    for n2, p2, c2 in objects[1:]:
        prompt_dict[f"{n1}+{n2}"] = {
            "category": "two_objects",
            "prompt": get_prompt_two_objects(p1, c1, p2, c2),
        }

    print(f"Generated {len(prompt_dict)} prompts: 1 control + {len(objects)} singles + "
          f"{len(objects) - 1} pairs")
    outfile = os.path.join(SCRIPT_DIR, "training_prompts.yaml")
    with open(outfile, "w") as f:
        yaml.dump(prompt_dict, f, sort_keys=False, allow_unicode=True)
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
