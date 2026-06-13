import importlib.util
from pathlib import Path

from board_generator import BoardGenerator


BASE_DIR = Path(__file__).resolve().parent
TRAINING_BOARDS_PATH = BASE_DIR / "generated_training_boards.py"
EVAL_BOARDS_PATH = BASE_DIR / "generated_eval_boards.py"

REGENERATE_GRADS = (7, 8)
FILL_MISSING_GRADS = (10, 11, 12)
TRAINING_COUNT = 500
EVAL_COUNT = 150
MISSING_TRAINING_COUNT = 75
MISSING_EVAL_COUNT = 150


def load_mapping(path, var_name):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, var_name)


def write_mapping(path, var_name, mapping):
    lines = [f"{var_name} = {{\n"]
    for grad in sorted(mapping):
        lines.append(f"    {grad}: [\n")
        for board in mapping[grad]:
            lines.append("        (\n")
            for row in board:
                lines.append(f"            {row!r},\n")
            lines.append("        ),\n")
        lines.append("    ],\n")
    lines.append("}\n")
    path.write_text("".join(lines))


def generate_boards(generator, count, seed_offset):
    return {
        grad: generator.generate(
            grad=grad,
            count=count,
            seed=seed_offset + grad,
            verbose=True,
        )
        for grad in REGENERATE_GRADS
    }


def generate_specific_boards(generator, grades, count, seed_offset):
    return {
        grad: generator.generate(
            grad=grad,
            count=count,
            seed=seed_offset + grad,
            verbose=True,
        )
        for grad in grades
    }


def has_expected_counts(mapping, count):
    return all(len(mapping.get(grad, [])) == count for grad in REGENERATE_GRADS)


def replace_grads_7_and_8(
    path,
    var_name,
    count,
    seed_offset,
    *,
    min_required_max_grad,
    skip_if_counts_match=False,
):
    existing = load_mapping(path, var_name)
    max_grad = max(existing)
    if max_grad < min_required_max_grad:
        raise RuntimeError(
            f"{path.name} only has max grad {max_grad}; expected at least "
            f"{min_required_max_grad} before replacing grads 7 and 8."
        )
    missing = [grad for grad in REGENERATE_GRADS if grad not in existing]
    if missing:
        raise RuntimeError(f"{path.name} is missing grads to replace: {missing}")
    if skip_if_counts_match and has_expected_counts(existing, count):
        print(f"Skipping {path.name}: grads 7 and 8 already have {count} boards each")
        return

    generator = BoardGenerator()
    replacements = generate_boards(generator, count, seed_offset)
    existing.update(replacements)
    write_mapping(path, var_name, existing)
    print(f"Updated {path.name}: replaced grads 7 and 8 only; grad 9 unchanged")


def fill_missing_grads_10_to_12(path, var_name, count, seed_offset):
    existing = load_mapping(path, var_name)
    missing = [grad for grad in FILL_MISSING_GRADS if not existing.get(grad)]
    if not missing:
        print(f"Skipping {path.name}: grads 10, 11, and 12 already exist")
        return

    generator = BoardGenerator()
    existing.update(generate_specific_boards(generator, missing, count, seed_offset))
    write_mapping(path, var_name, existing)
    print(f"Updated {path.name}: filled missing grads {missing}")


def main():
    replace_grads_7_and_8(
        TRAINING_BOARDS_PATH,
        "TRAINING_BOARDS",
        TRAINING_COUNT,
        7000,
        min_required_max_grad=17,
        skip_if_counts_match=True,
    )
    fill_missing_grads_10_to_12(
        TRAINING_BOARDS_PATH,
        "TRAINING_BOARDS",
        MISSING_TRAINING_COUNT,
        10000,
    )
    replace_grads_7_and_8(
        EVAL_BOARDS_PATH,
        "EVAL_BOARDS",
        EVAL_COUNT,
        9000,
        min_required_max_grad=15,
        skip_if_counts_match=True,
    )
    fill_missing_grads_10_to_12(
        EVAL_BOARDS_PATH,
        "EVAL_BOARDS",
        MISSING_EVAL_COUNT,
        12000,
    )


if __name__ == "__main__":
    main()
