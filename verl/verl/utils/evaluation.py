# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
"""
import re
import numpy as np
from verl.utils.reward_score.my_data import my_gsm8k, my_csqa, my_MATH, my_openbookqa, my_SVAMP, my_BBH, my_AIME2025
from math_verify import parse, verify
from collections import Counter

token_dict = {
    "direct": "<ANSWER>\n",
    "long_cot": "<LONG_COT>\n",
    "cot": "<COT>\n",
    "code": "<CODE>\n",
}

def find_instruct(_file):
    if 'code' in _file:
        return "<CODE>\n"
    elif 'long_cot' in _file:
        return "<LONG_COT>\n"
    elif 'cot' in _file:
        return "<COT>\n"
    elif 'direct' in _file:
        return "<ANSWER>\n"
    else:
        return ""

def extract_solution(solution_str):
    pattern = r"<ANSWER>(.*)</ANSWER>"
    solution = re.search(pattern, solution_str, re.DOTALL)
    if solution is None:
        final_answer1 = None
    else:
        final_answer1 = solution.group(1).strip()
    return final_answer1


def get_type(solution_str):
    if "<COT>" in solution_str:
        return "COT"
    elif "<CODE>" in solution_str:
        return "CODE"
    elif "<LONG_COT>" in solution_str:
        return "LONG_COT"
    else:
        return "DIRECT"


def select_reward_fn(data_source):
    if data_source == 'my_data/gsm8k':
        return my_gsm8k.compute_score_test
    elif data_source == 'my_data/csqa':
        return my_csqa.compute_score_test
    elif data_source == 'my_data/MATH':
        return my_MATH.compute_score_test
    elif data_source == 'my_data/openbookqa':
        return my_openbookqa.compute_score_test
    elif data_source == 'my_data/SVAMP':
        return my_SVAMP.compute_score_test
    elif 'BBH' in data_source:
        return my_BBH.compute_score_test
    elif data_source == 'my_data/AIME2025':
        return my_AIME2025.compute_score_test
    else:
        raise NotImplementedError


def find_most_common(answer_span_list, data_source):
    if data_source in ['my_data/gsm8k', 'my_data/MATH', 'my_data/SVAMP', 'my_data/AIME2025', 'my_data/BBH_object_counting_test']:
        parsed_list = [parse(answer_span) for answer_span in answer_span_list]
        ret_idx = 0
        max_common_num = 0
        for i, parsed_answer_span in enumerate(parsed_list):
            common_num = 0
            for j, parsed_answer_span_2 in enumerate(parsed_list):
                if i != j and verify(parsed_answer_span, parsed_answer_span_2):
                    common_num += 1
            if common_num > max_common_num:
                max_common_num = common_num
                ret_idx = i
        return answer_span_list[ret_idx]
    elif data_source in ['my_data/csqa', 'my_data/openbookqa']:
        answer_span_list = [answer_span.replace(r'(', '').replace(r')', '') for answer_span in answer_span_list]
        most_common_answer = Counter(answer_span_list).most_common(1)[0][0]
        return most_common_answer
    elif 'BBH' in data_source:
        if data_source in ['my_data/BBH_dyck_languages_test', 'my_data/BBH_word_sorting_test']:
            return Counter(answer_span_list).most_common(1)[0][0]
        answer_span_list = [answer_span.replace(r'\boxed{', '').replace(r'}', '') for answer_span in answer_span_list]
        answer_span_list = [answer_span.replace(r'\text{', '') for answer_span in answer_span_list]
        answer_span_list = [answer_span.replace(r'(', '').replace(r')', '') for answer_span in answer_span_list]
        answer_span_list = [answer_span.replace('\\', '') for answer_span in answer_span_list]
        if data_source in ['my_data/BBH_boolean_expressions_test', 'my_data/BBH_navigate_test',
                           'my_data/BBH_causal_judgement_test', 'my_data/BBH_formal_fallacies_test',
                           'my_data/BBH_web_of_lies_test', 'my_data/BBH_sports_understanding_test']:
            answer_span_list = [answer_span.replace('0', 'False') for answer_span in answer_span_list]
            answer_span_list = [answer_span.replace('1', 'True') for answer_span in answer_span_list]
            answer_span_list = [answer_span.lower() for answer_span in answer_span_list]
        if data_source in ['my_data/BBH_boolean_expressions_test', 'my_data/BBH_navigate_test',
                           'my_data/BBH_causal_judgement_test', 'my_data/BBH_web_of_lies_test']:
            answer_span_list = [answer_span[0].upper() + answer_span[1:] if len(answer_span) > 0 else answer_span for answer_span in answer_span_list]
        most_common_answer = Counter(answer_span_list).most_common(1)[0][0]
        return most_common_answer

    else:
        most_common_answer = Counter(answer_span_list).most_common(1)[0][0]
        return most_common_answer

