"""Data augmentation for PR tasks."""

import re
import random
import copy
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from .pr_loader import PRTask


# Variable name variations
VARIABLE_MAPPINGS = {
    's': ['text', 'string', 'input_str', 'value', 'content'],
    'n': ['num', 'number', 'value', 'count', 'x'],
    'a': ['first', 'x', 'num1', 'value1', 'left'],
    'b': ['second', 'y', 'num2', 'value2', 'right'],
    'result': ['output', 'res', 'ret', 'answer', 'out'],
    'items': ['elements', 'values', 'data', 'arr', 'lst'],
    'func': ['fn', 'function', 'f', 'callback', 'handler'],
}

# Docstring variations
DOCSTRING_PREFIXES = [
    "{}.",
    "{}",
    "Function to {}.",
    "This function will {}.",
    "A utility to {}.",
]

# Rephrase mappings for docstring first-line variation
DOCSTRING_REPHRASE = {
    'Computes': 'Calculates',
    'Calculates': 'Computes',
    'Returns': 'Gets',
    'Gets': 'Returns',
    'Checks': 'Verifies',
    'Verifies': 'Checks',
    'Creates': 'Constructs',
    'Constructs': 'Creates',
    'Finds': 'Locates',
    'Locates': 'Finds',
    'Converts': 'Transforms',
    'Transforms': 'Converts',
    'Removes': 'Deletes',
    'Deletes': 'Removes',
    'Parses': 'Reads',
    'Reads': 'Parses',
}


@dataclass
class AugmentedTask:
    """An augmented version of a PR task."""
    original_pr_id: str
    augmented_pr_id: str
    variant_name: str
    task: PRTask


class DataAugmenter:
    """Augment PR tasks for more training data."""
    
    def __init__(self, seed: int = 42):
        """Initialize augmenter.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)
    
    def augment_task(
        self,
        task: PRTask,
        strategies: List[str],
        multiplier: int = 3
    ) -> List[AugmentedTask]:
        """Augment a single task.
        
        Args:
            task: Original PR task
            strategies: List of augmentation strategies
            multiplier: How many variants to create
            
        Returns:
            List of augmented tasks
        """
        augmented = []
        
        for i in range(multiplier):
            # Apply random strategies
            new_task = copy.deepcopy(task)
            variant_name = f"v{i+1}"
            
            for strategy in strategies:
                if strategy == "variable_renaming":
                    new_task = self._apply_variable_renaming(new_task, i)
                    variant_name += "_renamed"
                elif strategy == "docstring_variation":
                    new_task = self._apply_docstring_variation(new_task, i)
                    variant_name += "_docvar"
                elif strategy == "whitespace_variation":
                    new_task = self._apply_whitespace_variation(new_task, i)
                    variant_name += "_ws"
            
            augmented.append(AugmentedTask(
                original_pr_id=task.pr_id,
                augmented_pr_id=f"{task.pr_id}_{variant_name}",
                variant_name=variant_name,
                task=new_task
            ))
        
        return augmented
    
    def augment_all(
        self,
        tasks: List[PRTask],
        strategies: List[str],
        multiplier: int = 3
    ) -> List[PRTask]:
        """Augment all tasks and return combined list.
        
        Args:
            tasks: Original tasks
            strategies: Augmentation strategies
            multiplier: Variants per task
            
        Returns:
            Combined list of original and augmented tasks
        """
        all_tasks = list(tasks)  # Include originals
        
        for task in tasks:
            augmented = self.augment_task(task, strategies, multiplier)
            for aug in augmented:
                all_tasks.append(aug.task)
        
        return all_tasks
    
    def _apply_variable_renaming(self, task: PRTask, variant_idx: int) -> PRTask:
        """Apply variable renaming augmentation."""
        # Get the expected code changes
        expected_changes = task.data.get('expected_changes', {})
        
        for filepath, changes in expected_changes.items():
            if 'additions' in changes:
                new_additions = []
                for line in changes['additions']:
                    new_line = self._rename_variables(line, variant_idx)
                    new_additions.append(new_line)
                changes['additions'] = new_additions
            
            if 'content' in changes:
                new_content = []
                for line in changes['content']:
                    new_line = self._rename_variables(line, variant_idx)
                    new_content.append(new_line)
                changes['content'] = new_content
        
        task.data['expected_changes'] = expected_changes
        return task
    
    # Matches a single- or double-quoted string literal (with escape support)
    _STRING_LITERAL_RE = re.compile(r'(\'(?:[^\'\\]|\\.)*\'|"(?:[^"\\]|\\.)*")')

    def _rename_variables(self, code: str, variant_idx: int) -> str:
        """Rename variables in a code string, leaving string literals intact."""
        # Split on string literals: even indices are code, odd are literals
        parts = self._STRING_LITERAL_RE.split(code)

        for k in range(0, len(parts), 2):
            segment = parts[k]
            for old_var, new_vars in VARIABLE_MAPPINGS.items():
                if len(new_vars) > variant_idx:
                    new_var = new_vars[variant_idx % len(new_vars)]
                    # Word boundaries, but skip attribute access (self.var, obj.var)
                    pattern = rf'(?<!\.)(?<!\w){re.escape(old_var)}(?!\w)'
                    segment = re.sub(pattern, new_var, segment)
            parts[k] = segment

        return ''.join(parts)
    
    def _apply_docstring_variation(self, task: PRTask, variant_idx: int) -> PRTask:
        """Apply docstring variation augmentation.

        Variant 0: Swap triple-quote style (double <-> single).
        Variant 1: Add an 'Args:' section if function has parameters.
        Variant 2: Rephrase the first line using synonym mappings.
        """
        expected_changes = task.data.get('expected_changes', {})
        variation = variant_idx % 3

        for filepath, changes in expected_changes.items():
            if 'additions' in changes:
                new_additions = []
                in_docstring = False
                docstring_first_line = True

                for line in changes['additions']:
                    stripped = line.strip()

                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        if in_docstring:
                            # Closing docstring
                            in_docstring = False
                            docstring_first_line = False
                            if variation == 0:
                                if '"""' in line:
                                    line = line.replace('"""', "'''")
                                else:
                                    line = line.replace("'''", '"""')
                        else:
                            # Opening docstring
                            in_docstring = True
                            docstring_first_line = True

                            if variation == 0:
                                # Swap quote style
                                if '"""' in line:
                                    line = line.replace('"""', "'''")
                                else:
                                    line = line.replace("'''", '"""')
                            elif variation == 1:
                                # Will add Args section after closing quote
                                pass
                            elif variation == 2:
                                # Rephrase first line
                                for old_word, new_word in DOCSTRING_REPHRASE.items():
                                    if old_word in line:
                                        line = line.replace(old_word, new_word, 1)
                                        break
                    elif in_docstring and docstring_first_line and variation == 2:
                        # Rephrase content lines within the first line of docstring
                        for old_word, new_word in DOCSTRING_REPHRASE.items():
                            if old_word in line:
                                line = line.replace(old_word, new_word, 1)
                                break
                        docstring_first_line = False

                    new_additions.append(line)

                # Variant 1: add Args section if function def is present
                if variation == 1:
                    new_additions = self._add_args_section(new_additions)

                changes['additions'] = new_additions

        task.data['expected_changes'] = expected_changes
        return task

    def _add_args_section(self, lines: List[str]) -> List[str]:
        """Add an Args: section to docstrings of functions that have parameters."""
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Detect function definitions
            if stripped.startswith('def ') and '(' in stripped:
                result.append(line)
                # Extract parameter names (skip self)
                match = re.search(r'\(([^)]*)\)', stripped)
                params = []
                if match:
                    raw_params = match.group(1).split(',')
                    for p in raw_params:
                        p = p.strip().split(':')[0].split('=')[0].strip()
                        if p and p != 'self':
                            params.append(p)

                # Look for existing docstring on next lines
                if i + 1 < len(lines):
                    next_stripped = lines[i + 1].strip()
                    if next_stripped.startswith('"""') or next_stripped.startswith("'''"):
                        # Find the indent of the docstring
                        indent = lines[i + 1][:len(lines[i + 1]) - len(lines[i + 1].lstrip())]
                        # Add the docstring opening line
                        result.append(lines[i + 1])
                        i += 2
                        # Find docstring closing and check if Args already present
                        has_args = False
                        docstring_lines = []
                        while i < len(lines):
                            ds = lines[i].strip()
                            if 'Args:' in ds:
                                has_args = True
                            docstring_lines.append(lines[i])
                            if (ds.endswith('"""') or ds.endswith("'''")) and len(docstring_lines) > 0:
                                break
                            i += 1

                        # Insert Args section before closing if params exist and no Args yet
                        if params and not has_args and docstring_lines:
                            closing = docstring_lines.pop()
                            for dl in docstring_lines:
                                result.append(dl)
                            result.append(f"{indent}")
                            result.append(f"{indent}Args:")
                            for p in params:
                                result.append(f"{indent}    {p}: Parameter value.")
                            result.append(closing)
                        else:
                            result.extend(docstring_lines)
                        i += 1
                        continue
                i += 1
            else:
                result.append(line)
                i += 1
        return result
    
    def _apply_whitespace_variation(self, task: PRTask, variant_idx: int) -> PRTask:
        """Apply whitespace variation augmentation.

        Variant 0: Extra blank line between functions (3 blank lines).
        Variant 1: Compact style (single blank line between functions).
        Variant 2: PEP 8 style (two blank lines between top-level functions).
        """
        expected_changes = task.data.get('expected_changes', {})
        variation = variant_idx % 3

        for filepath, changes in expected_changes.items():
            if 'additions' in changes:
                new_additions = []
                i = 0
                lines = changes['additions']

                while i < len(lines):
                    line = lines[i]

                    # Detect blank line regions preceding a top-level def/class
                    if line.strip() == '':
                        # Collect consecutive blank lines
                        blank_count = 0
                        while i < len(lines) and lines[i].strip() == '':
                            blank_count += 1
                            i += 1

                        # Check if next non-blank line is a top-level function/class
                        is_top_level_def = (
                            i < len(lines) and
                            (lines[i].startswith('def ') or lines[i].startswith('class '))
                        )

                        if is_top_level_def:
                            if variation == 0:
                                # Extra blank lines
                                new_additions.extend([''] * 3)
                            elif variation == 1:
                                # Compact: single blank line
                                new_additions.append('')
                            else:
                                # PEP 8: two blank lines
                                new_additions.extend([''] * 2)
                        else:
                            # Non-function blank lines: keep at most 1
                            if variation == 1:
                                new_additions.append('')
                            else:
                                new_additions.extend([''] * min(blank_count, 2))
                    else:
                        new_additions.append(line)
                        i += 1

                changes['additions'] = new_additions

        task.data['expected_changes'] = expected_changes
        return task
    
    def create_negative_examples(
        self,
        task: PRTask,
        num_negatives: int = 2
    ) -> List[Tuple[str, float]]:
        """Create negative (wrong) examples for contrastive learning.
        
        Args:
            task: Original task
            num_negatives: Number of negative examples
            
        Returns:
            List of (wrong_code, low_reward) tuples
        """
        negatives = []
        expected_changes = task.data.get('expected_changes', {})
        
        for filepath, changes in expected_changes.items():
            additions = changes.get('additions', [])
            if not additions:
                continue
            
            code = '\n'.join(additions)
            
            # Wrong: Missing return statement
            if 'return' in code:
                wrong1 = code.replace('return', '# return')
                negatives.append((wrong1, 0.2))
            
            # Wrong: Syntax error
            wrong2 = code.replace('def ', 'deff ')
            negatives.append((wrong2, 0.1))
            
            # Wrong: Wrong logic
            if 'True' in code:
                wrong3 = code.replace('True', 'False')
                negatives.append((wrong3, 0.3))
        
        return negatives[:num_negatives]
