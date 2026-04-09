import re

from datasets import load_dataset


def extract_answer(completion):
    matches = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def process_example(sample):

    sample_equation = find_equation(sample['nums'], sample['target'])
    assert sample_equation is not None, f"No equation found for nums={sample['nums']} and target={sample['target']}"

    PROMPT_MESSAGES_TEACHER = [
        {
            'role': 'user',
            'content': "Answer the following question. Provide the reasoning between <reasoning> and </reasoning> symbols. Provide the final answer (an expression) between <answer> and </answer> symbols.\n"\
                f"Using the numbers {sample['nums']}, create an equation that equals {sample['target']}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once." \
                f"The answer for this question is {sample_equation}. After understanding the reference solution, please try to solve this problem using your own approach below." \
        }
    ]

    PROMPT_MESSAGES_STUDENT = [
        {
            'role': 'user',
            'content': "Answer the following question. Provide the reasoning between <reasoning> and </reasoning> symbols. Provide the final answer (an expression) between <answer> and </answer> symbols.\n"\
                f"Using the numbers {sample['nums']}, create an equation that equals {sample['target']}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once." \
        }
    ]

    return {
        'prompt_teacher': PROMPT_MESSAGES_TEACHER,
        'prompt_student': PROMPT_MESSAGES_STUDENT,
        'sample_equation': sample_equation,
    }

def get_data():
    ds = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4')['train']
    ds = ds.select(range(10000))
    ds.cleanup_cache_files()
    ds = ds.map(
        process_example,
        num_proc=8
    )

    ds = ds.train_test_split(test_size=0.01, seed=42)

    return ds

def score_equation(equation: str | None, nums: list[int], target: int) -> dict:
    """Score a model-generated equation for the Countdown task.

    Returns a dict with:
        - correct: bool, whether the equation evaluates to the target
        - valid_numbers: bool, whether the equation uses the given numbers correctly
        - error: str or None, description of any error
    """
    if equation is None:
        return {"correct": False, "valid_numbers": False, "error": "no_answer_extracted"}
    
    OTHER_UNICODE_CHARS = {
        '\u2212': '-',
        '\u00d7': '*',
        '\u00f7': '/',
    }

    try:
        for char, replacement in OTHER_UNICODE_CHARS.items():
            equation = equation.replace(char, replacement)
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
        if sorted(used_numbers) != sorted(nums):
            return {"correct": False, "valid_numbers": False, "error": "wrong_numbers"}

        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return {"correct": False, "valid_numbers": True, "error": "invalid_characters"}

        result = eval(equation, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return {"correct": True, "valid_numbers": True, "error": None}
        else:
            return {"correct": False, "valid_numbers": True, "error": f"wrong_result_{result}"}
    except Exception as e:
        return {"correct": False, "valid_numbers": False, "error": str(e)}

# Find an equation that equals the target using the given numbers and basic arithmetic operations (+,-,*,/)
# Each number can only be used once. Return the equation as a string, or None if no equation can be found.
def find_equation(nums, target):
    def solve(available):
        # available is a list of (value, expression_string) tuples
        for i in range(len(available)):
            val, expr = available[i]
            if abs(val - target) < 1e-9 and len(available) <= len(nums):
                return expr

        if len(available) < 2:
            return None

        # Pick every pair and combine with each operator
        for i in range(len(available)):
            for j in range(len(available)):
                if i == j:
                    continue
                rest = [available[k] for k in range(len(available)) if k != i and k != j]
                a_val, a_expr = available[i]
                b_val, b_expr = available[j]

                # Wrap sub-expressions in parens if they contain operators
                a_str = f"({a_expr})" if any(op in a_expr for op in "+-*/") else a_expr
                b_str = f"({b_expr})" if any(op in b_expr for op in "+-*/") else b_expr

                candidates = [
                    (a_val + b_val, f"{a_str} + {b_str}"),
                    (a_val - b_val, f"{a_str} - {b_str}"),
                    (a_val * b_val, f"{a_str} * {b_str}"),
                ]
                if b_val != 0:
                    candidates.append((a_val / b_val, f"{a_str} / {b_str}"))

                for new_val, new_expr in candidates:
                    result = solve(rest + [(new_val, new_expr)])
                    if result is not None:
                        return result
        return None

    available = [(n, str(n)) for n in nums]
    return solve(available)