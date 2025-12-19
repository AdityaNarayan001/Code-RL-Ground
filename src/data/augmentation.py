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
    
    def _rename_variables(self, code: str, variant_idx: int) -> str:
        """Rename variables in a code string."""
        result = code
        
        for old_var, new_vars in VARIABLE_MAPPINGS.items():
            if len(new_vars) > variant_idx:
                new_var = new_vars[variant_idx % len(new_vars)]
                # Only rename if it's a standalone variable (word boundary)
                pattern = r'\b' + re.escape(old_var) + r'\b'
                # Skip if it's part of a longer word
                if len(old_var) == 1:
                    # For single char vars, be more careful
                    # Only replace in function args and assignments
                    pattern = rf'(?<=[\(\s,=:]){re.escape(old_var)}(?=[\s,\):\[])'
                    result = re.sub(pattern, new_var, result)
        
        return result
    
    def _apply_docstring_variation(self, task: PRTask, variant_idx: int) -> PRTask:
        """Apply docstring variation augmentation."""
        expected_changes = task.data.get('expected_changes', {})
        
        for filepath, changes in expected_changes.items():
            if 'additions' in changes:
                new_additions = []
                in_docstring = False
                
                for line in changes['additions']:
                    stripped = line.strip()
                    
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        if in_docstring:
                            in_docstring = False
                        else:
                            in_docstring = True
                            # Vary the docstring content slightly
                            if variant_idx == 0:
                                line = line.replace('"""', "'''")
                    
                    new_additions.append(line)
                
                changes['additions'] = new_additions
        
        task.data['expected_changes'] = expected_changes
        return task
    
    def _apply_whitespace_variation(self, task: PRTask, variant_idx: int) -> PRTask:
        """Apply whitespace variation augmentation."""
        # Add or remove blank lines between functions
        expected_changes = task.data.get('expected_changes', {})
        
        for filepath, changes in expected_changes.items():
            if 'additions' in changes:
                new_additions = []
                prev_empty = False
                
                for line in changes['additions']:
                    if line.strip() == '':
                        # Vary number of blank lines
                        if variant_idx % 2 == 0 and not prev_empty:
                            new_additions.append('')
                            new_additions.append('')
                        else:
                            new_additions.append('')
                        prev_empty = True
                    else:
                        new_additions.append(line)
                        prev_empty = False
                
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
