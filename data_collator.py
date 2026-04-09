import torch


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the problem (with chat template)
    Teacher: sees problem + solution + transition prompt (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(self, tokenizer, max_length=2048, reason_first=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first

        # Prompt for reasoning about the reference equation before teaching
        self.reason_first_prompt = (
            "\n\nThe reference equation above correctly reaches the target. "
            "Please analyze this equation and explain the key decomposition and ordering "
            "of operations that make it work. Do NOT use <think> tags. "
            "Do NOT derive your own equation. Simply analyze and explain the reference.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt = (
            "\n\nNow, using your own reasoning, construct an equation for the problem "
            "above that equals the target using each given number exactly once. "
            "Provide the reasoning between <reasoning> and </reasoning>, and the final "
            "expression between <answer> and </answer>.\n"
        )

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode

        for feature in features:
            # Student prompt: pre-built chat messages from data.py::process_example
            student_prompt = self.tokenizer.apply_chat_template(
                feature["prompt_student"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze the reference equation
                reasoning_user_message = (
                    f"Using the numbers {list(feature['nums'])}, create an equation that "
                    f"equals {int(feature['target'])} using +, -, *, / (each number once).\n\n"
                    f"Here is a correct reference equation for this problem:\n"
                    f"=== Reference Equation Start ===\n"
                    f"{feature['sample_equation']}\n"
                    f"=== Reference Equation End ===\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                # Teacher prompt: pre-built chat messages (includes reference equation)
                teacher_prompt = self.tokenizer.apply_chat_template(
                    feature["prompt_teacher"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                teacher_prompts.append(teacher_prompt)

        # Tokenize WITHOUT padding first to get true lengths
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = (
                f"\n{self.transition_prompt}\n"
                "Provide the reasoning between <reasoning> and </reasoning>, and the "
                "final expression between <answer> and </answer>."
            )
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Normal mode: tokenize teacher prompts
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        return result
