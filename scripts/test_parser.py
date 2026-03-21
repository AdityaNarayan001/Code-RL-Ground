"""Test the tool call parser with realistic model outputs."""
import sys
sys.path.insert(0, ".")
from src.environment.tools import ToolRegistry

tr = ToolRegistry()

# Test 1: write_file with parens in code (sum(...))
t1 = '<tool>write_file(path="pyutils/strings.py", content="def count_vowels(s):\n    return sum(1 for c in s.lower() if c in \\"aeiou\\")\n")</tool>'
r1 = tr.parse_tool_call(t1)
if r1 and "content" in r1["args"] and "count_vowels" in r1["args"]["content"]:
    print("Test1 PASS: parens in code")
else:
    print("Test1 FAIL:", r1)

# Test 2: write_file with escaped triple quotes
t2 = '<tool>write_file(path="pyutils/strings.py", content="def count_vowels(s):\n    \\"\\"\\"Count vowels.\\"\\"\\"\n    return sum(1 for c in s if c in set(\\"aeiouAEIOU\\"))\n")</tool>'
r2 = tr.parse_tool_call(t2)
if r2 and "content" in r2["args"] and "count_vowels" in r2["args"]["content"]:
    print("Test2 PASS: escaped triple quotes")
else:
    print("Test2 FAIL:", r2)

# Test 3: Real model output - multiline with no escaping
t3 = '''<tool>write_file(path="pyutils/strings.py", content="
def reverse_string(s):
    return s[::-1]

def count_vowels(s):
    vowels = set('aeiouAEIOU')
    return sum(1 for char in s if char in vowels)
")</tool>'''
r3 = tr.parse_tool_call(t3)
if r3 and "content" in r3["args"] and "count_vowels" in r3["args"]["content"]:
    print("Test3 PASS: multiline no escaping")
else:
    print("Test3 FAIL:", r3)

# Test 4: Model outputs content with triple-quoted docstrings (unescaped)
t4 = '''<tool>write_file(path="pyutils/strings.py", content="
"""String utilities."""

def reverse_string(s):
    """Reverse."""
    return s[::-1]

def count_vowels(s):
    """Count vowels."""
    vowels = 'aeiou'
    return sum(1 for c in s.lower() if c in vowels)
")</tool>'''
r4 = tr.parse_tool_call(t4)
if r4 and "content" in r4["args"] and "count_vowels" in r4["args"]["content"]:
    print("Test4 PASS: unescaped docstrings")
else:
    print("Test4 FAIL:", r4)

# Test 5: read_file
t5 = '<tool>read_file(path="pyutils/strings.py")</tool>'
r5 = tr.parse_tool_call(t5)
if r5 and r5["args"].get("path") == "pyutils/strings.py":
    print("Test5 PASS: read_file")
else:
    print("Test5 FAIL:", r5)

# Test 6: submit
t6 = "<tool>submit()</tool>"
r6 = tr.parse_tool_call(t6)
if r6 and r6["tool"] == "submit":
    print("Test6 PASS: submit")
else:
    print("Test6 FAIL:", r6)

# Show content for test 3 & 4
if r3:
    content = r3["args"].get("content", "")
    print(f"\nTest3 content ({len(content)} chars):")
    print(repr(content[:200]))
if r4:
    content = r4["args"].get("content", "")
    print(f"\nTest4 content ({len(content)} chars):")
    print(repr(content[:200]))
