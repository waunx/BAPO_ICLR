# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from math.py for obstacle classification task

import re
import json


def compute_score(solution_str, ground_truth) -> float:
    """
    Compute score for obstacle classification task.
    
    Args:
        solution_str: Model's response string
        ground_truth: Dict with keys 'object' and 'min_dist'
        
    Returns:
        float: Score between 0.0 and 1.0
        - 0.5 points for correct object classification
        - 0.5 points for distance within 0.05 tolerance
    """
    retval = 0.0
    try:
        # Extract answer from boxed content
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            
            # Parse the answer
            parsed_answer = parse_obstacle_answer(answer)
            if parsed_answer is not None:
                predicted_object, predicted_dist = parsed_answer
                
                # Check object classification (0.5 points)
                if is_object_correct(predicted_object, ground_truth.get('object')):
                    retval += 0.5
                
                # Check distance accuracy (0.5 points)
                if is_distance_correct(predicted_dist, ground_truth.get('min_dist')):
                    retval += 0.5
                    
    except Exception as e:
        print(f"Error in compute_score: {e}")

    return retval


def parse_obstacle_answer(answer_str):
    """
    Parse the obstacle classification answer from string.
    
    Expected formats:
    - "object: pallet, min_dist: 0.25"
    - "object:pallet,min_dist:0.25"
    - "{object: pallet, min_dist: 0.25}"
    
    Returns:
        tuple: (object_class, min_distance) or None if parsing fails
    """
    if not answer_str:
        return None
    
    try:
        # Clean the string
        cleaned_str = strip_obstacle_string(answer_str)
        
        # Try to extract object and distance using regex
        # Pattern 1: object: xxx, min_dist: yyy
        pattern1 = r'object:\s*([^,\s]+).*?min_dist:\s*([\d.]+)'
        match1 = re.search(pattern1, cleaned_str, re.IGNORECASE)
        
        if match1:
            object_class = match1.group(1).strip().lower()
            min_dist = float(match1.group(2))
            return object_class, min_dist
        
        # Pattern 2: Try JSON-like format
        # Convert to valid JSON format if possible
        json_pattern = r'\{.*?\}'
        json_match = re.search(json_pattern, cleaned_str)
        if json_match:
            json_str = json_match.group(0)
            # Try to fix common JSON format issues
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add quotes to keys
            json_str = re.sub(r':\s*([^,}\s]+)', r': "\1"', json_str)  # Add quotes to values
            json_str = re.sub(r'"\s*([\d.]+)\s*"', r'\1', json_str)  # Remove quotes from numbers
            
            try:
                parsed_json = json.loads(json_str)
                object_class = parsed_json.get('object', '').lower()
                min_dist = float(parsed_json.get('min_dist', 0))
                return object_class, min_dist
            except:
                pass
        
        # Pattern 3: Try to extract any object name and number
        object_pattern = r'(cart|forklift|robot|pallet|person|others)'
        dist_pattern = r'([\d.]+)'
        
        object_match = re.search(object_pattern, cleaned_str, re.IGNORECASE)
        dist_matches = re.findall(dist_pattern, cleaned_str)
        
        if object_match and dist_matches:
            object_class = object_match.group(1).lower()
            # Take the first reasonable distance value
            for dist_str in dist_matches:
                try:
                    min_dist = float(dist_str)
                    if 0 <= min_dist <= 10:  # Reasonable range for distance
                        return object_class, min_dist
                except:
                    continue
                    
    except Exception as e:
        print(f"Error parsing obstacle answer: {e}")
    
    return None


def is_object_correct(predicted_object, ground_truth_object, verbose=False):
    """Check if predicted object class matches ground truth"""
    
    if predicted_object is None or ground_truth_object is None:
        return False
    
    try:
        pred_obj = strip_object_name(predicted_object)
        gt_obj = strip_object_name(ground_truth_object)
        
        # print(f"Comparing objects: '{pred_obj}' vs '{gt_obj}'")

        if verbose:
            print(f"Comparing objects: '{pred_obj}' vs '{gt_obj}'")
            
        return pred_obj == gt_obj
    except Exception:
        return predicted_object.lower().strip() == ground_truth_object.lower().strip()


def is_distance_correct(predicted_dist, ground_truth_dist, tolerance=0.05, verbose=False):
    """Check if predicted distance is within tolerance of ground truth"""
    if predicted_dist is None or ground_truth_dist is None:
        return False
    
    try:
        pred_dist = float(predicted_dist)
        gt_dist = float(ground_truth_dist)
        
        distance_diff = abs(pred_dist - gt_dist)
        is_correct = distance_diff <= tolerance
        
        # print(f"Comparing distances: {pred_dist} vs {gt_dist}, diff: {distance_diff}, tolerance: {tolerance}")

        if verbose:
            print(f"Comparing distances: {pred_dist} vs {gt_dist}, diff: {distance_diff}, tolerance: {tolerance}")
            
        return is_correct
        
    except Exception as e:
        if verbose:
            print(f"Error comparing distances: {e}")
        return False


def remove_boxed(s):
    """Remove boxed wrapper from string"""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    """Extract the last boxed content from string"""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]
    return retval


def strip_obstacle_string(string):
    """Clean and normalize obstacle classification string"""
    # Remove linebreaks
    string = string.replace("\n", "")
    string = string.replace("\r", "")
    
    # Remove extra spaces
    string = re.sub(r'\s+', ' ', string)
    
    # Remove common prefixes
    string = string.strip()
    
    # Remove dollar signs
    string = string.replace("$", "")
    
    # Remove common punctuation at the end
    string = string.rstrip(".,;:")
    
    return string


def strip_object_name(object_name):
    """Normalize object name for comparison"""
    if not object_name:
        return ""
    
    # Convert to lowercase and strip spaces
    obj = str(object_name).lower().strip()
    
    # Remove common prefixes/suffixes
    obj = obj.replace("the ", "")
    obj = obj.replace("a ", "")
    obj = obj.replace("an ", "")
    
    # Remove quotes
    obj = obj.replace('"', '')
    obj = obj.replace("'", '')
    
    # Remove brackets
    obj = obj.replace("{", "")
    obj = obj.replace("}", "")
    obj = obj.replace("[", "")
    obj = obj.replace("]", "")
    
    return obj


# Test function for debugging
def test_obstacle_classification():
    """Test function to validate the scoring system"""
    
    test_cases = [
        # Perfect match
        {
            "solution": "\\boxed{object: pallet, min_dist: 0.25}",
            "ground_truth": {"object": "pallet", "min_dist": 0.25},
            "expected_score": 1.0
        },
        # Correct object, wrong distance
        {
            "solution": "\\boxed{object: pallet, min_dist: 0.5}",
            "ground_truth": {"object": "pallet", "min_dist": 0.25},
            "expected_score": 0.5
        },
        # Wrong object, correct distance
        {
            "solution": "\\boxed{object: cart, min_dist: 0.25}",
            "ground_truth": {"object": "pallet", "min_dist": 0.25},
            "expected_score": 0.5
        },
        # Both wrong
        {
            "solution": "\\boxed{object: cart, min_dist: 0.5}",
            "ground_truth": {"object": "pallet", "min_dist": 0.25},
            "expected_score": 0.0
        },
        # Distance within tolerance
        {
            "solution": "\\boxed{object: pallet, min_dist: 0.28}",
            "ground_truth": {"object": "pallet", "min_dist": 0.25},
            "expected_score": 1.0
        }
    ]
    
    print("Testing obstacle classification scoring...")
    for i, case in enumerate(test_cases):
        score = compute_score(case["solution"], case["ground_truth"])
        expected = case["expected_score"]
        status = "PASS" if abs(score - expected) < 1e-6 else "FAIL"
        print(f"Test {i+1}: {status} - Score: {score}, Expected: {expected}")
        
        if status == "FAIL":
            print(f"  Solution: {case['solution']}")
            print(f"  Ground truth: {case['ground_truth']}")


if __name__ == "__main__":
    test_obstacle_classification()